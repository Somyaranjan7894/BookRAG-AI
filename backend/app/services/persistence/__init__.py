"""Persistence service package for BookRAG AI."""

from app.services.persistence.exceptions import (
    ConstraintViolationError,
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    PersistenceError,
    ReferentialIntegrityError,
)
from app.services.persistence.embedding_service import EmbeddingPersistenceService
from app.services.persistence.service import DocumentPersistenceService

__all__ = [
    "ConstraintViolationError",
    "DocumentAlreadyExistsError",
    "DocumentNotFoundError",
    "DocumentPersistenceService",
    "EmbeddingPersistenceService",
    "PersistenceError",
    "ReferentialIntegrityError",
]
