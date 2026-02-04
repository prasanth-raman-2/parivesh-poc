"""
Model Catalogue - Centralized model configuration for LLM and Embedding models
"""

from enum import Enum


class LLMModels(str, Enum):
    """Large Language Models available for use"""
    CLAUDE_3_SONNET = "bedrock/arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_5_SONNET = "bedrock/arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_3_HAIKU = "bedrock/arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"


class EmbeddingModels(str, Enum):
    """Embedding models available for use"""
    COHERE_EMBED_ENGLISH_V3 = "bedrock/cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "bedrock/cohere.embed-multilingual-v3"


class ModelConfig:
    """Model configuration constants"""
    # Default models
    DEFAULT_LLM = LLMModels.CLAUDE_3_SONNET
    DEFAULT_EMBEDDING = EmbeddingModels.COHERE_EMBED_ENGLISH_V3
    
    # Embedding dimensions
    COHERE_EMBED_DIMENSION = 1024
    
    @classmethod
    def get_embedding_dimension(cls, model: EmbeddingModels) -> int:
        """Get embedding dimension for a given model"""
        if model in [EmbeddingModels.COHERE_EMBED_ENGLISH_V3, EmbeddingModels.COHERE_EMBED_MULTILINGUAL_V3]:
            return cls.COHERE_EMBED_DIMENSION
        return cls.COHERE_EMBED_DIMENSION  # default
