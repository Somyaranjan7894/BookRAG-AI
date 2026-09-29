"""Domain exceptions for Phase 11 Query Understanding and Query Planning."""

from typing import Optional


class QueryUnderstandingError(Exception):
    """Base exception for all query understanding domain errors."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class InvalidQueryError(QueryUnderstandingError):
    """Raised when an input query is empty, whitespace-only, or invalid."""
