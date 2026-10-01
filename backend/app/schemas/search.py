"""Pydantic schemas for Phase 5 & Phase 6 semantic search request and response contracts."""

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

# Architectural decisions:
# DEFAULT_TOP_K: final number of high-precision candidates returned to caller.
# MAX_TOP_K: upper bound on final returned results to prevent runaway network payloads.
# DEFAULT_CANDIDATE_K: high-recall first-stage candidate pool size retrieved from FAISS before reranking.
DEFAULT_TOP_K: int = 5
MAX_TOP_K: int = 100
DEFAULT_CANDIDATE_K: int = 20
MAX_CANDIDATE_K: int = 500


class SearchRequest(BaseModel):
    """Request payload for semantic vector search query with optional cross-encoder reranking."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query string. Must not be empty or whitespace-only.",
        examples=["What is backpropagation?"],
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description=f"Number of final nearest candidates to return (between 1 and {MAX_TOP_K}).",
        examples=[5],
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional source document identifier for document-level isolation / filtering.",
        examples=["doc_123"],
    )
    candidate_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_CANDIDATE_K,
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

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        """Ensure the search query is not empty or whitespace-only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Search query cannot be empty or whitespace-only.")
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
    def validate_candidate_k_bounds(self) -> "SearchRequest":
        """Verify candidate_k >= top_k when explicitly supplied."""
        if self.candidate_k is not None and self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be greater than or equal to top_k ({self.top_k})."
            )
        return self


class SearchResult(BaseModel):
    """Ranked search candidate preserving complete provenance back to source document and page.

    SEMANTIC SCORING MEANING:
    - similarity_score: First-stage dense retrieval relevance (cosine inner product between L2-normalized
      query and chunk embeddings in dense vector space).
    - reranker_score: Second-stage Cross-Encoder relevance score output by the joint transformer over
      (query, passage).
    NEITHER score represents:
    - a probability
    - an answer confidence score
    - a factual correctness score
    - a hallucination probability metric
    """

    rank: int = Field(ge=1, description="1-based final retrieval ranking position (1 = highest relevance)")
    original_rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Original 1-based candidate ranking position from first-stage FAISS retrieval",
    )
    chunk_id: str = Field(description="Unique source chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    page_number: int = Field(ge=1, description="1-based source page number")
    text: str = Field(description="Text content of the retrieved chunk")
    similarity_score: float = Field(
        description=(
            "Cosine similarity score representing first-stage dense retrieval relevance in vector space. "
            "NOTE: This is NOT a probability, answer confidence, or factual correctness metric."
        )
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description=(
            "Cross-Encoder relevance score from second-stage joint transformer scoring. "
            "NOTE: This is NOT a probability, answer confidence, or factual correctness metric."
        ),
    )
    chunk_index: int = Field(default=0, ge=0, description="Sequential index of the chunk within the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provenance context")


class SearchResponse(BaseModel):
    """API response payload for semantic search containing ranked candidates and provenance metadata."""

    query: str = Field(description="Normalized search query string")
    results: List[SearchResult] = Field(description="Ranked candidate chunks ordered by relevance score")
    total_results: int = Field(ge=0, description="Total number of candidates returned")
    document_id: Optional[str] = Field(default=None, description="Applied document filter ID, if any")
    reranking_applied: bool = Field(
        default=False,
        description="Whether second-stage Cross-Encoder reranking was applied to these results",
    )
    candidate_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total candidate pool size retrieved during first-stage dense retrieval before reranking.",
    )
