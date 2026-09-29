"""Search service module for BookRAG AI."""

from app.services.search.exceptions import (
    DocumentNotFoundError,
    IndexNotInitializedError,
    InvalidSearchQueryError,
    InvalidTopKError,
    SearchEmbeddingError,
    SearchError,
    SearchRetrievalError,
)
from app.services.search.query_search import QuerySearchService
from app.services.search.service import SearchService

__all__ = [
    "DocumentNotFoundError",
    "IndexNotInitializedError",
    "InvalidSearchQueryError",
    "InvalidTopKError",
    "QuerySearchService",
    "SearchEmbeddingError",
    "SearchError",
    "SearchRetrievalError",
    "SearchService",
]

