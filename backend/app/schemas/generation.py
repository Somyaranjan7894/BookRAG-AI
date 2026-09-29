"""Pydantic schemas for Phase 8 Abstractive Question Answering (Generation).

Defines API contracts for submitting natural language questions against indexed books
and receiving grounded answers synthesized by FLAN-T5 exclusively from retrieved evidence,
preserving complete evidence provenance.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_GENERATION_TOP_K: int = 5
MAX_GENERATION_TOP_K: int = 50
DEFAULT_GENERATION_CANDIDATE_K: int = 20
MAX_GENERATION_CANDIDATE_K: int = 500


class GenerationRequest(BaseModel):
    """Request payload for abstractive question answering using retrieved book evidence."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language question string. Must not be empty or whitespace-only.",
        examples=["Why was backpropagation introduced in neural networks?"],
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional source document identifier for document-level isolation / filtering.",
        examples=["deep_learning_handbook"],
    )
    top_k: int = Field(
        default=DEFAULT_GENERATION_TOP_K,
        ge=1,
        le=MAX_GENERATION_TOP_K,
        description=f"Number of top retrieved/reranked evidence chunks to evaluate (between 1 and {MAX_GENERATION_TOP_K}).",
        examples=[5],
    )
    candidate_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_GENERATION_CANDIDATE_K,
        description=(
            "Optional first-stage candidate pool size retrieved from FAISS before cross-encoder reranking. "
            "Must be greater than or equal to top_k."
        ),
        examples=[20],
    )
    enable_reranking: Optional[bool] = Field(
        default=None,
        description="Optional override to enable or disable Phase 6 Cross-Encoder reranking before generation.",
        examples=[True],
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=512,
        description="Maximum number of new tokens the generative model may produce.",
        examples=[128],
    )
    num_beams: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Beam search width for deterministic controlled generation.",
        examples=[4],
    )
    do_sample: Optional[bool] = Field(
        default=None,
        description="Whether to use random sampling. Defaults to False (deterministic).",
        examples=[False],
    )
    temperature: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=2.0,
        description="Sampling temperature if sampling is enabled.",
        examples=[1.0],
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        """Ensure the question query is not empty or whitespace-only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question query cannot be empty or whitespace-only.")
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
    def validate_candidate_k_bounds(self) -> "GenerationRequest":
        """Verify candidate_k >= top_k when explicitly supplied."""
        if self.candidate_k is not None and self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be greater than or equal to top_k ({self.top_k})."
            )
        return self


class GenerationEvidenceItem(BaseModel):
    """Retrieved evidence passage used as grounded context for FLAN-T5 generation."""

    rank: int = Field(ge=1, description="1-based ranking position of the chunk in the evidence set")
    chunk_id: str = Field(description="Unique source chunk identifier")
    document_id: str = Field(description="Source document identifier")
    page_number: int = Field(ge=1, description="1-based page number where the chunk appears")
    chunk_index: int = Field(ge=0, description="Sequential index of the chunk within the document")
    source_text: str = Field(description="Full text of the retrieved supporting passage")
    similarity_score: Optional[float] = Field(
        default=None,
        description="First-stage dense retrieval cosine similarity score.",
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Second-stage Cross-Encoder relevance score.",
    )


class GenerationResponse(BaseModel):
    """API response contract for abstractive question answering.

    SEMANTIC PRINCIPLES:
    - The generated answer is synthesized strictly from the supplied evidence context.
    - The retrieved evidence remains the sole source of truth.
    - FLAN-T5 generates answers based on provided text; it does NOT establish factual veracity.
    - No artificial confidence, factual correctness, or hallucination probabilities are invented.
      (Formal groundedness validation is introduced in Phase 9).
    """

    query: str = Field(description="Normalized input question string")
    answer: Optional[str] = Field(
        default=None,
        description="Generated abstractive answer synthesized from evidence, or null if unanswerable.",
    )
    answerable: bool = Field(
        description="Whether sufficient evidence was available to produce a grounded answer.",
    )
    model_name: str = Field(
        description="Hugging Face model identifier used for abstractive generation.",
    )
    evidence: List[GenerationEvidenceItem] = Field(
        default_factory=list,
        description="Supporting evidence chunks formatted and provided to the generation model.",
    )
    evidence_count: int = Field(
        default=0,
        ge=0,
        description="Total number of evidence chunks included in the generation context.",
    )
