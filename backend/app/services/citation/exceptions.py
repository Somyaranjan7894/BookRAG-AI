"""Domain exception hierarchy for Phase 10 Citation and Provenance Mapping."""

from typing import Optional


class CitationError(Exception):
    """Base exception for all citation domain errors."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class DocumentIsolationError(CitationError):
    """Raised when evidence from an unauthorized or mismatched document is detected during citation mapping."""


class InvalidCitationInputError(CitationError):
    """Raised when input grounding results or claims are malformed."""
