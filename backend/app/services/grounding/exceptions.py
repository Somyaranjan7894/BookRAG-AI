"""Domain exception hierarchy for Phase 9 Groundedness & Hallucination Control."""

from typing import Optional


class GroundingError(Exception):
    """Base exception for all groundedness and NLI domain errors."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class NLIModelLoadError(GroundingError):
    """Raised when the CrossEncoder NLI model fails to load."""


class NLILabelMappingError(GroundingError):
    """Raised when the model label mapping (id2label) cannot safely resolve entailment/contradiction/neutral."""


class NLIInferenceError(GroundingError):
    """Raised when NLI scoring or batch inference encounters a runtime failure."""


class InvalidGroundingInputError(GroundingError):
    """Raised when grounding input queries, claims, or evidence are malformed."""


class InvalidGroundingConfigError(GroundingError):
    """Raised when grounding configuration thresholds or parameters are invalid."""
