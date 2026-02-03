import logging
from pydantic_settings import BaseSettings ,PydanticBaseSettingsSource
from pydantic import Field

from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    AWS_REGION: str = Field(env="REGION", default="us-east-1", description="Region of the aws")
    AWS_DEPLOYMENT_REGION: str = Field(env="AWS_DEPLOYMENT_REGION", default="us-east-1", description="Region of the aws deployment")
    AWS_ACCESS_KEY_ID: str = Field(env="AWS_ACCESS_KEY_ID", description="Access key of aws account")
    AWS_SECRET_ACCESS_KEY: str = Field(env="AWS_SECRET_ACCESS_KEY", description="Secret key of aws account")
    SECRET_KEY: str = Field(env="SECRET_KEY", default="my-secret-key", description="Secret key for JWT token generation")
    REFRESH_SECRET_KEY: str = Field(env="REFRESH_SECRET_KEY", default="123", description="Refresh secret key for JWT token generation")
    ALGORITHM: str = Field(env="ALGORITHM", default="HS256", description="Algorithm for JWT token generation")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(env="ACCESS_TOKEN_EXPIRE_MINUTES", default=30, description="Access token expiry in minutes")
    
    # Milvus Configuration
    MILVUS_HOST: str = Field(env="MILVUS_HOST", default="localhost", description="Milvus server host")
    MILVUS_PORT: str = Field(env="MILVUS_PORT", default="19530", description="Milvus server port")
    MILVUS_COLLECTION_NAME: str = Field(env="MILVUS_COLLECTION_NAME", default="document_embeddings", description="Milvus collection name")
    EMBEDDING_DIMENSION: int = Field(env="EMBEDDING_DIMENSION", default=3072, description="Embedding dimension for text-embedding-3-large")



settings = Settings()

# Set up basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
