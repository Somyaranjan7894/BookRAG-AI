"""Phase 8 Abstractive Question Answering (Generation) package.

Provides controlled abstractive text generation using google/flan-t5-base
conditioned strictly on retrieved and reranked book evidence.
"""

from app.services.generation.evidence import BuiltPrompt, EvidenceBuilder
from app.services.generation.exceptions import (
    ContextBudgetExceededError,
    GenerationError,
    GenerationInferenceError,
    GenerationModelLoadError,
    InvalidGenerationConfigError,
    InvalidGenerationEvidenceError,
    InvalidGenerationQueryError,
)
from app.services.generation.model import GenerationModel
from app.services.generation.service import GenerationService

__all__ = [
    "GenerationModel",
    "EvidenceBuilder",
    "BuiltPrompt",
    "GenerationService",
    "GenerationError",
    "GenerationModelLoadError",
    "GenerationInferenceError",
    "InvalidGenerationQueryError",
    "InvalidGenerationEvidenceError",
    "InvalidGenerationConfigError",
    "ContextBudgetExceededError",
]
