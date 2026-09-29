"""Phase 7 Extractive Question Answering package.

Provides extractive span selection using deepset/roberta-base-squad2 over
retrieved and reranked evidence chunks without generative hallucination.
"""

from app.services.qa.exceptions import (
    InvalidAnswerSpanError,
    InvalidQAConfigError,
    InvalidQAEvidenceError,
    InvalidQAQueryError,
    QAError,
    QAInferenceError,
    QAModelLoadError,
)
from app.services.qa.model import QAModel
from app.services.qa.service import QAService

__all__ = [
    "QAModel",
    "QAService",
    "QAError",
    "QAModelLoadError",
    "QAInferenceError",
    "InvalidQAQueryError",
    "InvalidQAEvidenceError",
    "InvalidAnswerSpanError",
    "InvalidQAConfigError",
]
