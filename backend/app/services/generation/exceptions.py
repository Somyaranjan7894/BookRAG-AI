"""Domain exception hierarchy for Phase 8 Abstractive Question Answering (Generation)."""

from typing import Optional


class GenerationError(Exception):
    """Base exception for all generation domain errors."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class GenerationModelLoadError(GenerationError):
    """Raised when the Seq2Seq generation model or tokenizer fails to load."""


class GenerationInferenceError(GenerationError):
    """Raised when FLAN-T5 generation forward pass or beam search fails."""


class InvalidGenerationQueryError(GenerationError):
    """Raised when the question query is empty, whitespace-only, or invalid."""


class InvalidGenerationEvidenceError(GenerationError):
    """Raised when evidence items are malformed or missing required content."""


class InvalidGenerationConfigError(GenerationError):
    """Raised when generation configuration parameters are out of allowable bounds."""


class ContextBudgetExceededError(GenerationError):
    """Raised when query or fixed prompt overhead exceeds the maximum allowed token budget."""
