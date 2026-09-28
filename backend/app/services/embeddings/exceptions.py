"""Domain-specific exceptions for semantic embedding generation."""


class EmbeddingError(Exception):
    """Base exception for all embedding-related failures."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ModelLoadError(EmbeddingError):
    """Raised when the SentenceTransformer model fails to load or initialize."""


class InvalidChunkError(EmbeddingError):
    """Raised when an input chunk is invalid, None, or contains empty/whitespace-only text."""


class EmbeddingConfigError(EmbeddingError):
    """Raised when embedding configuration parameters are invalid."""
