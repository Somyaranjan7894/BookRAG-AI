"""Domain exceptions for Phase 12 Question Generation in BookRAG AI."""

from typing import Any, Optional


class QuestionGenerationError(Exception):
    """Base domain exception for question generation failures."""

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class QuestionModelLoadError(QuestionGenerationError):
    """Raised when the question generation model or tokenizer fails to load."""
    pass


class QuestionInferenceError(QuestionGenerationError):
    """Raised when question generation model forward inference fails."""
    pass


class AnswerExtractionError(QuestionGenerationError):
    """Raised when candidate answer extraction fails unexpectedly."""
    pass


class QuestionValidationError(QuestionGenerationError):
    """Raised when question validation encounters an internal error."""
    pass


class InsufficientCandidatesError(QuestionGenerationError):
    """Raised when not enough candidate answers could be extracted from evidence."""
    pass
