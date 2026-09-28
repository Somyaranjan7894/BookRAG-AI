"""Semantic embeddings service package for BookRAG AI."""

from app.services.embeddings.exceptions import (
    EmbeddingConfigError,
    EmbeddingError,
    InvalidChunkError,
    ModelLoadError,
)
from app.services.embeddings.model import (
    DEFAULT_MODEL_NAME,
    EXPECTED_DIMENSION,
    EmbeddingModel,
    resolve_device,
)
from app.services.embeddings.service import EmbeddingService

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EXPECTED_DIMENSION",
    "EmbeddingConfigError",
    "EmbeddingError",
    "EmbeddingModel",
    "EmbeddingService",
    "InvalidChunkError",
    "ModelLoadError",
    "resolve_device",
]
