"""Grounding and hallucination control package for BookRAG AI Phase 9."""

from app.schemas.grounding import (
    ClaimEvidenceProvenance,
    ClaimResult,
    ExtractedClaim,
    GroundedAnswerRequest,
    GroundedAnswerResponse,
    GroundingReport,
)
from app.services.grounding.claims import ClaimDecomposer
from app.services.grounding.exceptions import (
    GroundingError,
    InvalidGroundingConfigError,
    InvalidGroundingInputError,
    NLIInferenceError,
    NLILabelMappingError,
    NLIModelLoadError,
)
from app.services.grounding.model import NLIModel, NLIScores
from app.services.grounding.orchestrator import GroundedAnswerService
from app.services.grounding.service import GroundingService

__all__ = [
    "GroundingService",
    "GroundedAnswerService",
    "NLIModel",
    "NLIScores",
    "ClaimDecomposer",
    "ExtractedClaim",
    "ClaimResult",
    "GroundingReport",
    "ClaimEvidenceProvenance",
    "GroundedAnswerRequest",
    "GroundedAnswerResponse",
    "GroundingError",
    "NLIModelLoadError",
    "NLILabelMappingError",
    "NLIInferenceError",
    "InvalidGroundingInputError",
    "InvalidGroundingConfigError",
]
