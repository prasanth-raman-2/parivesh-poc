
from litellm import embedding
from app.milvus import MilvusClient
from app.core.settings import settings
from typing import List, Dict, Any, Optional


class DocumentIngestion:
    def __init__(self, use_milvus: bool = True):
        self.configuration={
            "chunk_size": 6000,
            "chunk_overlap": 400,
        }
        self.embedding_model = "text-embedding-3-large"
        self.output_path = "knowledge/knowledge_base.json"
        self.use_milvus = use_milvus
        
        # Initialize Milvus client if enabled
        if self.use_milvus:
            self.milvus_client = MilvusClient(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                collection_name=settings.MILVUS_COLLECTION_NAME,
                dim=settings.EMBEDDING_DIMENSION
            )
    
    def connect_milvus(self):
        """Connect to Milvus and create collection if needed."""
        if self.use_milvus:
            self.milvus_client.connect()
            self.milvus_client.create_collection(drop_existing=False)
            print(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    
    def disconnect_milvus(self):
        """Disconnect from Milvus."""
        if self.use_milvus:
            self.milvus_client.disconnect()
            print("Disconnected from Milvus")
    
    def get_chunks(self, text: str):
        print(f"Ingesting document from text input")

        chunk_size = self.configuration["chunk_size"]
        chunk_overlap = self.configuration["chunk_overlap"]

        # Simple chunking logic
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - chunk_overlap
        
        return chunks
    
    def get_embeddings(self, chunks: List[str]) -> List[List[float]]:
        print(f"Generating embeddings for {len(chunks)} chunks using model {self.embedding_model}")
        embeddings = []
        for chunk in chunks:
            response = embedding(
                model=self.embedding_model,
                input=chunk
            )
            print(f"Embedding response: {response}")
            embeddings.append(response['data'][0]['embedding'])
        return embeddings
    
    def store_embeddings_to_milvus(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Store embeddings to Milvus.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional metadata for each chunk
            
        Returns:
            List of inserted IDs
        """
        if not self.use_milvus:
            raise ValueError("Milvus is not enabled. Set use_milvus=True during initialization.")
        
        if metadata is None:
            # Create default metadata
            metadata = [{"chunk_index": i, "source": "document"} for i in range(len(chunks))]
        
        # Insert embeddings
        ids = self.milvus_client.insert_embeddings(chunks, embeddings, metadata)
        print(f"Successfully stored {len(chunks)} embeddings to Milvus with IDs: {ids[:5]}...")
        
        return ids
    
    def ingest_and_store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline: chunk text, generate embeddings, and store to Milvus.
        
        Args:
            text: Input text to process
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            Dictionary with processing statistics
        """
        # Get chunks
        chunks = self.get_chunks(text)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.get_embeddings(chunks)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Prepare metadata for each chunk
        chunk_metadata = []
        for i in range(len(chunks)):
            meta = {"chunk_index": i}
            if metadata:
                meta.update(metadata)
            chunk_metadata.append(meta)
        
        # Store to Milvus if enabled
        ids = None
        if self.use_milvus:
            ids = self.store_embeddings_to_milvus(chunks, embeddings, chunk_metadata)
        
        return {
            "num_chunks": len(chunks),
            "num_embeddings": len(embeddings),
            "milvus_ids": ids,
            "milvus_enabled": self.use_milvus
        }
    
    def search_similar(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using query text.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with metadata
        """
        if not self.use_milvus:
            raise ValueError("Milvus is not enabled. Set use_milvus=True during initialization.")
        
        # Generate embedding for query
        query_embedding = self.get_embeddings([query_text])[0]
        
        # Load collection and search
        self.milvus_client.load_collection()
        results = self.milvus_client.search(
            query_embeddings=[query_embedding],
            top_k=top_k,
            output_fields=["text", "metadata"]
        )
        
        return results[0] if results else []
