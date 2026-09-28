"""Pydantic schemas and models for vector index metadata and retrieval results."""

from datetime import datetime, timezone
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class VectorMappingItem(BaseModel):
    """Mapping between a FAISS internal vector index position and chunk provenance."""

    vector_index: int = Field(ge=0, description="0-based FAISS vector slot position")
    chunk_id: str = Field(description="Unique source chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    page_number: int = Field(ge=1, description="1-based source page number")
    chunk_index: int = Field(default=0, ge=0, description="Sequential index of chunk within document")
    text: str = Field(description="Source text corresponding to this vector")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provenance context")


class IndexMetadata(BaseModel):
    """Metadata describing a FAISS vector index instance."""

    index_id: str = Field(default="default", description="Unique identifier for the index")
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model used to generate indexed vectors",
    )
    dimension: int = Field(default=384, ge=1, description="Embedding vector dimensionality")
    total_vectors: int = Field(default=0, ge=0, description="Total number of vectors stored in the index")
    document_ids: List[str] = Field(default_factory=list, description="List of unique document IDs indexed")
    normalized: bool = Field(default=True, description="Whether vectors are L2-normalized")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp",
    )


class RetrievalResult(BaseModel):
    """Ranked vector search result containing chunk text and provenance metadata."""

    rank: int = Field(ge=1, description="1-based retrieval ranking position (1 = nearest/most similar)")
    chunk_id: str = Field(description="Unique source chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    page_number: int = Field(ge=1, description="1-based source page number")
    chunk_index: int = Field(default=0, ge=0, description="Sequential index of chunk")
    text: str = Field(description="Text content of the retrieved chunk")
    similarity_score: float = Field(
        description="Cosine-style inner product similarity score between query and chunk vector"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provenance context")


class RetrievalQueryRequest(BaseModel):
    """Request payload for semantic vector search query."""

    query: str = Field(min_length=1, description="User search query text")
    top_k: int = Field(default=5, gt=0, le=100, description="Number of top candidates to retrieve")
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID for document-level filtering/isolation",
    )
    index_id: Optional[str] = Field(
        default=None,
        description="Optional target index ID (defaults to active index)",
    )


class RetrievalQueryResponse(BaseModel):
    """Response payload containing ranked retrieval candidate results."""

    query: str = Field(description="Original search query string")
    top_k: int = Field(description="Requested maximum number of results")
    total_results: int = Field(description="Actual number of candidates retrieved")
    document_id: Optional[str] = Field(default=None, description="Applied document filter ID, if any")
    results: List[RetrievalResult] = Field(description="Ranked candidate chunks ordered by similarity score")


class BuildIndexRequest(BaseModel):
    """Request payload to construct a vector index from serialized embedding records."""

    index_id: str = Field(default="default", description="Target index identifier")
    document_id: Optional[str] = Field(default=None, description="Primary document ID")
    records: List[dict] = Field(description="List of serialized EmbeddingRecord dictionaries")


class BuildIndexResponse(BaseModel):
    """Response payload after building or updating a vector index."""

    index_id: str
    total_vectors: int
    dimension: int
    model_name: str
    document_ids: List[str]
