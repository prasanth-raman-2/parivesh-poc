from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "document_embeddings",
        dim: int = 3072,  # dimension for text-embedding-3-large
    ):
        """
        Initialize Milvus client.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to store embeddings
            dim: Dimension of embeddings (3072 for text-embedding-3-large)
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        
    def connect(self):
        """Establish connection to Milvus server."""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"Successfully connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Milvus server."""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
    
    def create_collection(self, drop_existing: bool = False):
        """
        Create a collection in Milvus for storing document embeddings.
        
        Args:
            drop_existing: If True, drop existing collection with same name
        """
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                if drop_existing:
                    logger.info(f"Dropping existing collection: {self.collection_name}")
                    utility.drop_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    self.collection = Collection(self.collection_name)
                    return
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document embeddings collection"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            
            # Create index for vector field
            self.create_index()
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def create_index(self):
        """Create IVF_FLAT index for the embedding field."""
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info("Created index for embedding field")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def insert_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]] = None
    ) -> List[int]:
        """
        Insert embeddings into Milvus collection.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional list of metadata dictionaries
            
        Returns:
            List of inserted IDs
        """
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            if metadata is None:
                metadata = [{"source": "unknown"} for _ in range(len(texts))]
            
            # Prepare data
            data = [
                texts,
                embeddings,
                metadata
            ]
            
            # Insert data
            insert_result = self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Inserted {len(texts)} embeddings into Milvus")
            return insert_result.primary_keys
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            raise
    
    def load_collection(self):
        """Load collection into memory for searching."""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            self.collection.load()
            logger.info(f"Loaded collection {self.collection_name} into memory")
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise
    
    def search(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        output_fields: List[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embeddings: List of query embedding vectors
            top_k: Number of top results to return
            output_fields: Fields to include in results
            
        Returns:
            Search results
        """
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            if output_fields is None:
                output_fields = ["text", "metadata"]
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=query_embeddings,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                hit_list = []
                for hit in hits:
                    hit_list.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "text": hit.entity.get("text"),
                        "metadata": hit.entity.get("metadata"),
                    })
                formatted_results.append(hit_list)
            
            logger.info(f"Search completed, found {len(formatted_results)} result sets")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            stats = {
                "name": self.collection.name,
                "num_entities": self.collection.num_entities,
                "description": self.collection.description,
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
