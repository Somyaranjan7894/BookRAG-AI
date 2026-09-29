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

    # Cross-Encoder Reranking (Phase 6)
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_MAX_LENGTH: int = 512
    RERANKER_BATCH_SIZE: int = 32
    RERANKER_DEVICE: str = "auto"
    RERANKER_ENABLED: bool = True
    RETRIEVAL_CANDIDATE_K: int = 20

    # Extractive Question Answering (Phase 7)
    QA_MODEL_NAME: str = "deepset/roberta-base-squad2"
    QA_MAX_LENGTH: int = 512
    QA_DOC_STRIDE: int = 128
    QA_MAX_ANSWER_LENGTH: int = 64
    QA_TOP_K_EVIDENCE: int = 5
    QA_N_BEST_SIZE: int = 20
    QA_ENABLED: bool = True
    QA_DEVICE: str = "auto"
    QA_NO_ANSWER_THRESHOLD: float = 0.0

    # Abstractive Question Answering / Generation (Phase 8)
    GENERATION_MODEL_NAME: str = "google/flan-t5-base"
    GENERATION_MAX_INPUT_TOKENS: int = 2048
    GENERATION_MAX_NEW_TOKENS: int = 128
    GENERATION_NUM_BEAMS: int = 4
    GENERATION_DO_SAMPLE: bool = False
    GENERATION_TEMPERATURE: float = 1.0
    GENERATION_TOP_K_EVIDENCE: int = 5
    GENERATION_ENABLED: bool = True
    GENERATION_DEVICE: str = "auto"





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
