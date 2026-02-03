"""
API endpoints for document ingestion and management
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import os
from collections import Counter

from app.schemas.ingestion import (
    IngestionRequest,
    IngestionResponse,
    IngestionStats,
    DocumentListResponse,
    DocumentInfo,
    PaginatedChunksResponse,
    ChunkResponse,
    ChunkMetadata,
    EntityInfo
)
from app.ingestion.advanced_ingestion import AdvancedDocumentIngestion
from app.milvus import MilvusClient
from app.core.settings import settings
from pymilvus import Collection, connections

router = APIRouter(prefix="/ingestion", tags=["Document Ingestion"])


@router.delete("/clear", response_model=dict)
async def clear_database():
    """
    Clear all data from the Milvus collection.
    
    WARNING: This will delete all ingested documents and cannot be undone!
    """
    try:
        # Connect to Milvus
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        
        # Drop collection if it exists
        from pymilvus import utility
        if utility.has_collection(settings.MILVUS_COLLECTION_NAME):
            utility.drop_collection(settings.MILVUS_COLLECTION_NAME)
            
            # Recreate empty collection
            milvus_client = MilvusClient(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                collection_name=settings.MILVUS_COLLECTION_NAME,
                dim=settings.EMBEDDING_DIMENSION
            )
            milvus_client.create_collection(drop_existing=False)
            
            connections.disconnect("default")
            
            return {
                "success": True,
                "message": f"Successfully cleared collection '{settings.MILVUS_COLLECTION_NAME}'"
            }
        else:
            connections.disconnect("default")
            return {
                "success": True,
                "message": f"Collection '{settings.MILVUS_COLLECTION_NAME}' does not exist"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(request: IngestionRequest):
    """
    Ingest a document into the vector database.
    
    - **file_path**: Path to the document file (must exist)
    - **clear_existing**: Whether to clear existing data first
    - **chunk_size**: Size of text chunks (1000-10000 chars)
    - **overlap**: Overlap between chunks (0-1000 chars)
    """
    try:
        # Validate file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Run ingestion
        ingester = AdvancedDocumentIngestion()
        stats = ingester.ingest_document(
            file_path=request.file_path,
            clear_existing=request.clear_existing
        )
        
        # Convert top_keywords from tuples to dictionaries
        top_keywords_dict = [
            {"keyword": keyword, "count": count} 
            for keyword, count in stats.get('top_keywords', [])
        ]
        
        # Convert stats to response model
        ingestion_stats = IngestionStats(
            total_chunks=stats['total_chunks'],
            total_embeddings=stats['total_embeddings'],
            milvus_ids_sample=stats['milvus_ids_sample'],
            chunk_types=stats['chunk_types'],
            unique_sections=stats['unique_sections'],
            page_range=stats['page_range'],
            top_keywords=top_keywords_dict,
            source_document=stats['source_document']
        )
        
        return IngestionResponse(
            success=True,
            message=f"Successfully ingested {stats['total_chunks']} chunks from {request.file_path}",
            stats=ingestion_stats
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all ingested documents with summary statistics.
    
    Returns a list of documents with their metadata summary.
    """
    try:
        # Connect to Milvus
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        
        # Load collection
        collection = Collection(settings.MILVUS_COLLECTION_NAME)
        collection.load()
        
        # Query all metadata - use 'id >= 0' since id is an actual field
        results = collection.query(
            expr="id >= 0",
            output_fields=["metadata"],
            limit=16384  # Milvus max limit
        )
        
        # Group by source document
        documents_map = {}
        for result in results:
            metadata = result.get('metadata', {})
            source_doc = metadata.get('source_document', 'unknown')
            
            if source_doc not in documents_map:
                documents_map[source_doc] = {
                    'chunks': [],
                    'chunk_types': [],
                    'pages': [],
                    'sections': []
                }
            
            documents_map[source_doc]['chunks'].append(metadata)
            documents_map[source_doc]['chunk_types'].append(metadata.get('chunk_type', 'unknown'))
            
            page_num = metadata.get('page_number', 0)
            if page_num > 0:
                documents_map[source_doc]['pages'].append(page_num)
            
            section_num = metadata.get('section_number')
            if section_num:
                documents_map[source_doc]['sections'].append(section_num)
        
        # Build response
        documents = []
        for source_doc, data in documents_map.items():
            chunk_type_counts = Counter(data['chunk_types'])
            pages = data['pages']
            page_range = f"{min(pages)} - {max(pages)}" if pages else "Unknown"
            
            documents.append(DocumentInfo(
                source_document=source_doc,
                total_chunks=len(data['chunks']),
                chunk_types=dict(chunk_type_counts),
                page_range=page_range,
                unique_sections=len(set(data['sections']))
            ))
        
        connections.disconnect("default")
        
        return DocumentListResponse(
            total_documents=len(documents),
            documents=documents
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/chunks", response_model=PaginatedChunksResponse)
async def get_chunks(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    source_document: Optional[str] = Query(None, description="Filter by source document"),
    chunk_type: Optional[str] = Query(None, description="Filter by chunk type"),
    section_number: Optional[str] = Query(None, description="Filter by section number")
):
    """
    Get paginated chunks with metadata for verification.
    
    - **page**: Page number (starting from 1)
    - **page_size**: Number of chunks per page (1-100)
    - **source_document**: Optional filter by source document name
    - **chunk_type**: Optional filter by chunk type (table, section_content, etc.)
    - **section_number**: Optional filter by section number (e.g., "2.3")
    """
    try:
        # Connect to Milvus
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        
        # Load collection
        collection = Collection(settings.MILVUS_COLLECTION_NAME)
        collection.load()
        
        # Build filter expression - start with a basic expression using actual fields
        filter_conditions = []
        
        # Note: For JSON field filtering in Milvus, we need to use proper syntax
        # However, complex JSON filtering may not work well, so we'll filter in memory
        # Use a basic expression that always matches
        expr = "id >= 0"
        
        # Get all results (we'll filter in Python)
        all_results = collection.query(
            expr=expr,
            output_fields=["id", "text", "metadata"],
            limit=16384  # Milvus max limit
        )
        
        # Filter in Python
        filtered_results = []
        for result in all_results:
            metadata_dict = result.get('metadata', {})
            
            # Apply filters
            if source_document and metadata_dict.get('source_document') != source_document:
                continue
            if chunk_type and metadata_dict.get('chunk_type') != chunk_type:
                continue
            if section_number and metadata_dict.get('section_number') != section_number:
                continue
            
            filtered_results.append(result)
        
        # Calculate pagination on filtered results
        total = len(filtered_results)
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        
        # Apply pagination
        paginated_results = filtered_results[offset:offset + page_size]
        
        # Build response
        chunks = []
        for result in paginated_results:
            metadata_dict = result.get('metadata', {})
            entities_dict = metadata_dict.get('entities', {})
            
            chunks.append(ChunkResponse(
                id=result.get('id'),
                text=result.get('text'),
                metadata=ChunkMetadata(
                    chunk_index=metadata_dict.get('chunk_index', 0),
                    page_number=metadata_dict.get('page_number', 0),
                    section_number=metadata_dict.get('section_number'),
                    section_title=metadata_dict.get('section_title'),
                    chunk_type=metadata_dict.get('chunk_type', 'unknown'),
                    keywords=metadata_dict.get('keywords', []),
                    entities=EntityInfo(
                        locations=entities_dict.get('locations', []),
                        organizations=entities_dict.get('organizations', []),
                        numbers=entities_dict.get('numbers', []),
                        dates=entities_dict.get('dates', [])
                    ),
                    char_count=metadata_dict.get('char_count', 0),
                    word_count=metadata_dict.get('word_count', 0),
                    has_tables=metadata_dict.get('has_tables', False),
                    has_numbers=metadata_dict.get('has_numbers', False),
                    source_document=metadata_dict.get('source_document', 'unknown')
                )
            ))
        
        connections.disconnect("default")
        
        return PaginatedChunksResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            chunks=chunks
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {str(e)}")


@router.get("/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk_by_id(chunk_id: int):
    """
    Get a specific chunk by its Milvus ID.
    
    - **chunk_id**: The Milvus ID of the chunk
    """
    try:
        # Connect to Milvus
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        
        # Load collection
        collection = Collection(settings.MILVUS_COLLECTION_NAME)
        collection.load()
        
        # Query by ID
        results = collection.query(
            expr=f"id == {chunk_id}",
            output_fields=["id", "text", "metadata"]
        )
        
        if not results:
            connections.disconnect("default")
            raise HTTPException(status_code=404, detail=f"Chunk with ID {chunk_id} not found")
        
        result = results[0]
        metadata_dict = result.get('metadata', {})
        entities_dict = metadata_dict.get('entities', {})
        
        chunk = ChunkResponse(
            id=result.get('id'),
            text=result.get('text'),
            metadata=ChunkMetadata(
                chunk_index=metadata_dict.get('chunk_index', 0),
                page_number=metadata_dict.get('page_number', 0),
                section_number=metadata_dict.get('section_number'),
                section_title=metadata_dict.get('section_title'),
                chunk_type=metadata_dict.get('chunk_type', 'unknown'),
                keywords=metadata_dict.get('keywords', []),
                entities=EntityInfo(
                    locations=entities_dict.get('locations', []),
                    organizations=entities_dict.get('organizations', []),
                    numbers=entities_dict.get('numbers', []),
                    dates=entities_dict.get('dates', [])
                ),
                char_count=metadata_dict.get('char_count', 0),
                word_count=metadata_dict.get('word_count', 0),
                has_tables=metadata_dict.get('has_tables', False),
                has_numbers=metadata_dict.get('has_numbers', False),
                source_document=metadata_dict.get('source_document', 'unknown')
            )
        )
        
        connections.disconnect("default")
        
        return chunk
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunk: {str(e)}")
