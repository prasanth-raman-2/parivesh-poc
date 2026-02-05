"""
Pydantic schemas for Document Ingestion
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class IngestionRequest(BaseModel):
    """Request schema for document ingestion."""
    file_path: str = Field(..., description="Path to the document file to ingest")
    clear_existing: bool = Field(
        default=False, 
        description="Whether to clear existing data before ingestion"
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached metadata and embeddings if available"
    )
    chunk_size: int = Field(default=4000, ge=1000, le=10000, description="Size of text chunks")
    overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")


class EntityInfo(BaseModel):
    """Named entities extracted from text."""
    locations: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    numbers: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk."""
    chunk_index: int
    page_number: int
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    chunk_type: str
    keywords: List[str]
    entities: EntityInfo
    char_count: int
    word_count: int
    has_tables: bool
    has_numbers: bool
    source_document: str


class ChunkResponse(BaseModel):
    """Response schema for a chunk with metadata."""
    id: int = Field(..., description="Milvus ID of the chunk")
    text: str = Field(..., description="Text content of the chunk")
    metadata: ChunkMetadata


class PaginatedChunksResponse(BaseModel):
    """Paginated response for chunks."""
    total: int = Field(..., description="Total number of chunks")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    chunks: List[ChunkResponse]


class IngestionStats(BaseModel):
    """Statistics from document ingestion."""
    total_chunks: int
    total_embeddings: int
    milvus_ids_sample: List[int]
    chunk_types: Dict[str, int]
    unique_sections: int
    page_range: str
    top_keywords: List[Dict[str, Any]]
    source_document: str


class IngestionResponse(BaseModel):
    """Response schema for document ingestion."""
    success: bool
    message: str
    stats: Optional[IngestionStats] = None


class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    source_document: str
    total_chunks: int
    chunk_types: Dict[str, int]
    page_range: str
    unique_sections: int


class DocumentListResponse(BaseModel):
    """Response schema for listing documents."""
    total_documents: int
    documents: List[DocumentInfo]
