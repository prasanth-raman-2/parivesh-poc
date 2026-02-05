"""
Model Catalogue - Centralized model configuration for LLM and Embedding models
"""

from enum import Enum


class LLMModels(str, Enum):
    """Large Language Models available for use"""
    CLAUDE_3_SONNET = "bedrock/arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_5_SONNET = "bedrock/arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_3_HAIKU = "bedrock/arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
    GPT_5_2 = "gpt-5.2"


class EmbeddingModels(str, Enum):
    """Embedding models available for use"""
    COHERE_EMBED_ENGLISH_V3 = "bedrock/cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "bedrock/cohere.embed-multilingual-v3"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class ModelConfig:
    """Model configuration constants"""
    # Default models
    DEFAULT_LLM = LLMModels.CLAUDE_3_SONNET
    DEFAULT_EMBEDDING = EmbeddingModels.COHERE_EMBED_ENGLISH_V3
    
    # Embedding dimensions
    COHERE_EMBED_DIMENSION = 1024
    TEXT_EMBEDDING_3_LARGE_DIMENSION = 3072
    
    # Collection names
    COHERE_COLLECTION_NAME = "cohere_embeddings"
    OPENAI_COLLECTION_NAME = "openai_embeddings"
    
    # Chunk size configurations
    COHERE_CHUNK_SIZE = 1500
    COHERE_CHUNK_OVERLAP = 150
    TEXT_EMBEDDING_CHUNK_SIZE = 6000
    TEXT_EMBEDDING_CHUNK_OVERLAP = 400
    
    @classmethod
    def get_embedding_dimension(cls, model: EmbeddingModels) -> int:
        """Get embedding dimension for a given model"""
        if model in [EmbeddingModels.COHERE_EMBED_ENGLISH_V3, EmbeddingModels.COHERE_EMBED_MULTILINGUAL_V3]:
            return cls.COHERE_EMBED_DIMENSION
        elif model == EmbeddingModels.TEXT_EMBEDDING_3_LARGE:
            return cls.TEXT_EMBEDDING_3_LARGE_DIMENSION
        return cls.COHERE_EMBED_DIMENSION  # default
    
    @classmethod
    def get_collection_name(cls, model: EmbeddingModels) -> str:
        """Get Milvus collection name for a given embedding model
        
        Returns:
            Collection name string
        """
        if model in [EmbeddingModels.COHERE_EMBED_ENGLISH_V3, EmbeddingModels.COHERE_EMBED_MULTILINGUAL_V3]:
            return cls.COHERE_COLLECTION_NAME
        elif model == EmbeddingModels.TEXT_EMBEDDING_3_LARGE:
            return cls.OPENAI_COLLECTION_NAME
        return cls.COHERE_COLLECTION_NAME  # default
    
    @classmethod
    def get_chunk_config(cls, model: EmbeddingModels) -> tuple[int, int]:
        """Get chunk size and overlap for a given embedding model
        
        Returns:
            Tuple of (chunk_size, overlap)
        """
        if model in [EmbeddingModels.COHERE_EMBED_ENGLISH_V3, EmbeddingModels.COHERE_EMBED_MULTILINGUAL_V3]:
            return (cls.COHERE_CHUNK_SIZE, cls.COHERE_CHUNK_OVERLAP)
        elif model == EmbeddingModels.TEXT_EMBEDDING_3_LARGE:
            return (cls.TEXT_EMBEDDING_CHUNK_SIZE, cls.TEXT_EMBEDDING_CHUNK_OVERLAP)
        return (cls.COHERE_CHUNK_SIZE, cls.COHERE_CHUNK_OVERLAP)  # default
