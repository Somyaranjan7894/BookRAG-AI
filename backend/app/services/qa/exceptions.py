"""Domain exception classes for Phase 7 Extractive Question Answering."""

from typing import Optional


class QAError(Exception):
    """Base domain exception for all Question Answering errors."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class QAModelLoadError(QAError):
    """Raised when the extractive QA model or tokenizer cannot be loaded."""


class QAInferenceError(QAError):
    """Raised when forward-pass inference or sliding-window evaluation fails."""


class InvalidQAQueryError(QAError):
    """Raised when a question string is empty, whitespace-only, or invalid."""


class InvalidQAEvidenceError(QAError):
    """Raised when evidence candidates are empty, malformed, or missing required fields."""


class InvalidAnswerSpanError(QAError):
    """Raised when an answer span violates boundary, context, or length constraints."""


class InvalidQAConfigError(QAError):
    """Raised when QA configuration parameters are invalid or out of bounds."""
