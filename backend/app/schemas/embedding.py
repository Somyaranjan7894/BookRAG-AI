"""Pydantic schemas and models for semantic embeddings and embedding configurations."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Configuration controlling embedding model inference and batch execution."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face identifier for the SentenceTransformer model.",
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Number of text passages encoded concurrently in a single batch.",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to normalize vectors to unit length (L2 norm = 1.0).",
    )
    device: str = Field(
        default="auto",
        description="Execution device: 'auto' (CUDA if available else CPU), 'cpu', or 'cuda'.",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device format string."""
        allowed = {"auto", "cpu", "cuda"}
        v_clean = v.strip().lower()
        if v_clean not in allowed and not v_clean.startswith("cuda:"):
            raise ValueError(f"Invalid device '{v}'. Must be 'auto', 'cpu', or 'cuda'.")
        return v_clean


class EmbeddingRecord(BaseModel):
    """Structured representation of a vector embedding linked with chunk provenance."""

    chunk_id: str = Field(description="Unique identifier of the source text chunk")
    document_id: str = Field(description="Identifier of the parent document")
    page_number: int = Field(ge=1, description="1-based page number from the source PDF")
    text: str = Field(description="Original chunk text corresponding to this embedding")
    embedding: List[float] = Field(description="Normalized dense vector representation")
    dimension: int = Field(description="Vector dimension (e.g. 384)")
    model_name: str = Field(description="Name of the model used to generate this vector")
    device: str = Field(description="Hardware device used for inference (e.g. 'cpu' or 'cuda')")
    normalized: bool = Field(description="Indicates whether the vector has been L2 normalized")


class EmbedChunkRequest(BaseModel):
    """Request payload for single chunk embedding development endpoint."""

    chunk_id: str
    document_id: str
    page_number: int = Field(ge=1)
    text: str
    config: Optional[EmbeddingConfig] = None


class EmbedChunksRequest(BaseModel):
    """Request payload for multi-chunk batch embedding development endpoint."""

    chunks: List[dict] = Field(description="List of serialized chunk dictionaries")
    config: Optional[EmbeddingConfig] = None


class EmbedChunksResponse(BaseModel):
    """Response payload containing generated embedding records."""

    total_embedded: int
    dimension: int
    model_name: str
    device: str
    records: List[EmbeddingRecord]
