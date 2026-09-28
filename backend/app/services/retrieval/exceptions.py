"""Domain-specific exceptions for vector retrieval and FAISS operations."""


class RetrievalError(Exception):
    """Base exception for all vector retrieval and indexing operations."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class IndexDimensionMismatchError(RetrievalError):
    """Raised when an embedding or query vector dimension does not match the index dimension."""


class EmptyIndexError(RetrievalError):
    """Raised when attempting to search an empty index or retrieve from an unpopulated state."""


class IndexNotFoundError(RetrievalError):
    """Raised when a requested index identifier cannot be found in the registry or storage."""


class InvalidQueryError(RetrievalError):
    """Raised when an input search query is invalid, empty, or whitespace-only."""


class IndexPersistenceError(RetrievalError):
    """Raised when saving or loading FAISS index or metadata files fails."""


class CorruptedIndexError(RetrievalError):
    """Raised when persisted index artifacts and metadata files are inconsistent or corrupted."""
