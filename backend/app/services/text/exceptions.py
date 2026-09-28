"""Domain-specific exceptions for text cleaning and chunking operations."""


class TextProcessingError(Exception):
    """Base exception for all text processing and chunking failures."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class InvalidChunkingConfigError(TextProcessingError):
    """Raised when chunking configuration parameters are invalid or contradictory."""
