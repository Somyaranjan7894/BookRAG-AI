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
