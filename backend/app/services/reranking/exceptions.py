"""Domain-specific exceptions for Phase 6 Cross-Encoder reranking operations."""


class RerankingError(Exception):
    """Base exception for all cross-encoder reranking domain operations."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class RerankerModelLoadError(RerankingError):
    """Raised when the CrossEncoder transformer model fails to load or initialize."""


class RerankingExecutionError(RerankingError):
    """Raised when cross-encoder inference fails or candidate inputs are malformed."""


class ScoreAlignmentError(RerankingError):
    """Raised when cross-encoder prediction count does not strictly match candidate count."""


class InvalidRerankingConfigError(RerankingError):
    """Raised when reranking parameters or configuration bounds are invalid."""
