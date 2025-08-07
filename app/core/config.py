from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from functools import lru_cache

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "LLM Query Retrieval System"
    DEBUG: bool = True

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_TOKEN: str

    # OpenAI Configuration - OPTIMIZED FOR ACCURACY
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4.1"       # Better model for accuracy
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    MAX_TOKENS: int = 2000                   # Increased for detailed answers
    TEMPERATURE: float = 0.0                 # Deterministic for consistency

    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "document-embeddings"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./test.db"
    DB_ECHO: bool = False

    # Document Processing - OPTIMIZED FOR ACCURACY
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = "pdf,docx,txt,eml"
    CHUNK_SIZE: int = 1000                   # Larger chunks for more context
    CHUNK_OVERLAP: int = 200                 # More overlap for continuity

    # Vector Store - OPTIMIZED FOR ACCURACY
    VECTOR_STORE_TYPE: str = "faiss"
    EMBEDDING_DIMENSION: int = 1536
    TOP_K_RESULTS: int = 8                   # More results for better context

    # Performance Settings
    REQUEST_TIMEOUT: int = 120               # Longer timeout for accuracy
    MAX_CONCURRENT_REQUESTS: int = 3         # Fewer concurrent for stability
    CACHE_TTL_SECONDS: int = 3600

    # Token Management - OPTIMIZED FOR ACCURACY
    MAX_CONTEXT_TOKENS: int = 6000           # Larger context window
    MAX_PROMPT_TOKENS: int = 8000            # Larger prompt limit
    TOKEN_BUFFER: int = 500
    
    # Batch Processing
    EMBEDDING_BATCH_SIZE: int = 15
    PARALLEL_QUESTIONS: int = 2              # Reduced for better quality

    @property
    def allowed_extensions_list(self) -> List[str]:
        return self.ALLOWED_EXTENSIONS.split(",")

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
