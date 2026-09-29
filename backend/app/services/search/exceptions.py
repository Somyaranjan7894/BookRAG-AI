"""Domain-specific exceptions for search operations in BookRAG AI."""


class SearchError(Exception):
    """Base exception for all search domain operations."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class InvalidSearchQueryError(SearchError):
    """Raised when an input search query is invalid, empty, or whitespace-only."""


class InvalidTopKError(SearchError):
    """Raised when top_k is non-positive or exceeds maximum allowed bounds."""


class DocumentNotFoundError(SearchError):
    """Raised when a requested document ID does not exist in the search index."""


class IndexNotInitializedError(SearchError):
    """Raised when search is executed against an uninitialized or missing index."""


class SearchEmbeddingError(SearchError):
    """Raised when generating an embedding vector for the search query fails."""


class SearchRetrievalError(SearchError):
    """Raised when vector retrieval execution encounters an internal failure."""
