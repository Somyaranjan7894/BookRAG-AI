"""Pydantic schemas for Phase 5 semantic search request and response contracts."""

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator

# Architectural decision: MAX_TOP_K is set to 100 to balance multi-chunk retrieval context
# synthesis for downstream generation while bounding vector distance computation, memory usage,
# and network payload sizes in latency-sensitive API operations.
DEFAULT_TOP_K: int = 5
MAX_TOP_K: int = 100


class SearchRequest(BaseModel):
    """Request payload for semantic vector search query."""

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
        description=f"Number of top nearest candidates to retrieve (between 1 and {MAX_TOP_K}).",
        examples=[5],
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional source document identifier for document-level isolation / filtering.",
        examples=["doc_123"],
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


class SearchResult(BaseModel):
    """Ranked search candidate preserving complete provenance back to source document and page.

    SEMANTIC MEANING:
    similarity_score represents semantic retrieval relevance (cosine similarity inner product
    between L2-normalized query and chunk embeddings in dense vector space).
    It is NOT:
    - a probability
    - an answer confidence score
    - a factual correctness score
    - a hallucination probability metric
    """

    rank: int = Field(ge=1, description="1-based retrieval ranking position (1 = nearest/most similar)")
    chunk_id: str = Field(description="Unique source chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    page_number: int = Field(ge=1, description="1-based source page number")
    text: str = Field(description="Text content of the retrieved chunk")
    similarity_score: float = Field(
        description=(
            "Cosine similarity score representing semantic retrieval relevance in vector space. "
            "NOTE: This is NOT a probability, answer confidence, or factual correctness metric."
        )
    )
    chunk_index: int = Field(default=0, ge=0, description="Sequential index of the chunk within the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provenance context")


class SearchResponse(BaseModel):
    """API response payload for semantic search containing ranked candidates and provenance metadata."""

    query: str = Field(description="Normalized search query string")
    results: List[SearchResult] = Field(description="Ranked candidate chunks ordered by similarity score")
    total_results: int = Field(ge=0, description="Total number of candidates returned")
    document_id: Optional[str] = Field(default=None, description="Applied document filter ID, if any")
