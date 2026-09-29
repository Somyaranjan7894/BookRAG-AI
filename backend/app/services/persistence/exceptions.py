"""Exceptions for Phase 13 PostgreSQL persistence operations."""

from typing import Any, Dict, Optional


class PersistenceError(Exception):
    """Base exception for all persistence layer operations."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentAlreadyExistsError(PersistenceError):
    """Raised when attempting to insert a document with an ID that already exists."""
    pass


class DocumentNotFoundError(PersistenceError):
    """Raised when an operation targets a nonexistent document."""
    pass


class ReferentialIntegrityError(PersistenceError):
    """Raised when a foreign key or relational constraint is violated."""
    pass


class ConstraintViolationError(PersistenceError):
    """Raised when a unique or check constraint is violated."""
    pass
