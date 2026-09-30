"""Domain exceptions for asynchronous document processing."""

from typing import Optional


class DocumentProcessingError(Exception):
    """Base exception for document processing pipeline errors."""

    def __init__(
        self,
        message: str,
        is_retryable: bool = False,
        details: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.is_retryable = is_retryable
        self.details = details

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(message='{self.message}', retryable={self.is_retryable})>"


class TransientProcessingError(DocumentProcessingError):
    """Retryable failure (e.g. database connectivity timeout, temporary worker resource error)."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message, is_retryable=True, details=details)


class PermanentProcessingError(DocumentProcessingError):
    """Non-retryable deterministic failure (e.g. file not found, corrupt PDF, invalid schema)."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message, is_retryable=False, details=details)
