"""Pydantic schemas for Phase 7 Extractive Question Answering.

Defines API contracts for asking natural language questions against indexed books
and receiving extracted answer spans with complete provenance back to the source text.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_QA_TOP_K: int = 5
MAX_QA_TOP_K: int = 50
DEFAULT_QA_CANDIDATE_K: int = 20
MAX_QA_CANDIDATE_K: int = 500


class QARequest(BaseModel):
    """Request payload for extractive question answering against indexed documents."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language question string. Must not be empty or whitespace-only.",
        examples=["What is backpropagation?"],
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional source document identifier for document-level isolation / filtering.",
        examples=["doc_123"],
    )
    top_k: int = Field(
        default=DEFAULT_QA_TOP_K,
        ge=1,
        le=MAX_QA_TOP_K,
        description=f"Number of top retrieved/reranked evidence chunks to evaluate (between 1 and {MAX_QA_TOP_K}).",
        examples=[5],
    )
    candidate_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_QA_CANDIDATE_K,
        description=(
            "Optional first-stage candidate pool size retrieved from FAISS before cross-encoder reranking. "
            "Must be greater than or equal to top_k."
        ),
        examples=[20],
    )
    enable_reranking: Optional[bool] = Field(
        default=None,
        description="Optional override to enable or disable Phase 6 Cross-Encoder reranking.",
        examples=[True],
    )
    no_answer_threshold: Optional[float] = Field(
        default=None,
        description=(
            "Optional threshold for SQuAD 2.0 no-answer decision. If (best_span_score - null_score) <= threshold, "
            "the question is treated as unanswerable. Defaults to QA_NO_ANSWER_THRESHOLD."
        ),
        examples=[0.0],
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        """Ensure the query is not empty or whitespace-only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace-only.")
        return stripped

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: Optional[str]) -> Optional[str]:
        """Ensure document_id, if provided, is not empty or whitespace-only."""
        if v is not None:
            stripped = v.strip()
            if not stripped:
                raise ValueError("document_id cannot be empty or whitespace-only if provided.")
            return stripped
        return v

    @model_validator(mode="after")
    def validate_candidate_k_bounds(self) -> "QARequest":
        """Verify candidate_k >= top_k when explicitly supplied."""
        if self.candidate_k is not None and self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be greater than or equal to top_k ({self.top_k})."
            )
        return self


class QAResponse(BaseModel):
    """Structured response for extractive Question Answering with complete provenance.

    SEMANTIC SCORING MEANING:
    - similarity_score: First-stage dense retrieval relevance (cosine inner product in dense vector space).
    - reranker_score: Second-stage Cross-Encoder relevance score output by joint transformer over (query, passage).
    - qa_score: Extractive span logit score (start_logits + end_logits) output by the QA model.
    - no_answer_score: Logit score for the unanswerable/null token (<s> / [CLS]).

    CRITICAL RULES:
    NONE of these scores represents:
    - a probability of truth or factual correctness
    - an answer confidence percentage
    - a hallucination probability metric
    The retrieved document/evidence is the sole source of truth; the model extracts a text span.
    """

    query: str = Field(description="Normalized question query string")
    answer: Optional[str] = Field(
        default=None,
        description="Extracted answer span from the evidence text, or null if unanswerable.",
    )
    answerable: bool = Field(
        description="Whether a sufficiently supported answer span was identified within the retrieved evidence.",
    )
    qa_score: Optional[float] = Field(
        default=None,
        description=(
            "Extractive model span logit score (start_logits + end_logits). "
            "NOTE: This is NOT a probability, factual correctness, or answer confidence percentage."
        ),
    )
    no_answer_score: Optional[float] = Field(
        default=None,
        description="Model logit score for the no-answer/null token in the evaluated context window.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Source document identifier from which the answer was extracted.",
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Unique chunk identifier from which the answer span was extracted.",
    )
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based page number where the answer chunk is located.",
    )
    chunk_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Sequential index of the chunk within the document.",
    )
    answer_start: Optional[int] = Field(
        default=None,
        ge=0,
        description="0-based character start offset of the answer span within the source chunk text.",
    )
    answer_end: Optional[int] = Field(
        default=None,
        ge=0,
        description="0-based character end offset (exclusive) of the answer span within the source chunk text.",
    )
    source_text: Optional[str] = Field(
        default=None,
        description="Full text of the retrieved evidence chunk from which the answer was extracted.",
    )
    evidence_rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based rank of the supporting chunk in the retrieved/reranked evidence list.",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="First-stage dense retrieval cosine similarity score for the supporting chunk.",
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Second-stage Cross-Encoder relevance score for the supporting chunk.",
    )
    total_evidence_evaluated: int = Field(
        default=0,
        ge=0,
        description="Total number of evidence chunks evaluated by the extractive QA model.",
    )
