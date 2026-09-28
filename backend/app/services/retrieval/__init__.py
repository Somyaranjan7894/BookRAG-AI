"""Vector retrieval service package for BookRAG AI."""

from app.services.retrieval.exceptions import (
    CorruptedIndexError,
    EmptyIndexError,
    IndexDimensionMismatchError,
    IndexNotFoundError,
    IndexPersistenceError,
    InvalidQueryError,
    RetrievalError,
)
from app.services.retrieval.index import VectorIndex
from app.services.retrieval.mapping import VectorToChunkMapping
from app.services.retrieval.service import RetrievalService

__all__ = [
    "CorruptedIndexError",
    "EmptyIndexError",
    "IndexDimensionMismatchError",
    "IndexNotFoundError",
    "IndexPersistenceError",
    "InvalidQueryError",
    "RetrievalError",
    "RetrievalService",
    "VectorIndex",
    "VectorToChunkMapping",
]
