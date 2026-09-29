"""Question generation package for BookRAG AI Phase 12."""

from app.services.question_generation.answers import AnswerCandidateExtractor
from app.services.question_generation.deduplication import (
    QuestionDeduplicator,
    normalize_question_for_dedup,
)
from app.services.question_generation.exceptions import (
    AnswerExtractionError,
    InsufficientCandidatesError,
    QuestionGenerationError,
    QuestionInferenceError,
    QuestionModelLoadError,
    QuestionValidationError,
)
from app.services.question_generation.model import QuestionGenerationModel
from app.services.question_generation.service import QuestionGenerationService
from app.services.question_generation.validator import (
    QuestionValidator,
    classify_question_difficulty,
    classify_question_type,
    matches_expected_answer,
)

__all__ = [
    "AnswerCandidateExtractor",
    "AnswerExtractionError",
    "InsufficientCandidatesError",
    "QuestionDeduplicator",
    "QuestionGenerationError",
    "QuestionGenerationModel",
    "QuestionGenerationService",
    "QuestionInferenceError",
    "QuestionModelLoadError",
    "QuestionValidationError",
    "QuestionValidator",
    "classify_question_difficulty",
    "classify_question_type",
    "matches_expected_answer",
    "normalize_question_for_dedup",
]
