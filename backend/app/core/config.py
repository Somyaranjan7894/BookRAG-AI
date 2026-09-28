"""Application configuration settings for BookRAG AI.

Provides centralized configuration management via environment variables
and pydantic-settings.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    # Application Information
    APP_NAME: str = "BookRAG AI"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = (
        "Research-oriented, production-quality Retrieval-Augmented Generation "
        "application for asking questions about complete books."
    )
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"

    # Server Network Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Text Chunking Defaults (Phase 2)
    CHUNKING_TARGET_SIZE: int = 1200
    CHUNKING_MAX_SIZE: int = 1600
    CHUNKING_OVERLAP: int = 200

    # Semantic Embeddings (Phase 3)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_DEVICE: str = "auto"
    EMBEDDING_DIMENSION: int = 384

    # Vector Retrieval (Phase 4)
    RETRIEVAL_DEFAULT_TOP_K: int = 5
    RETRIEVAL_MAX_TOP_K: int = 100
    INDEX_STORAGE_DIR: str = "data/indexes"



    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()


settings = get_settings()
