"""Pydantic models and configuration schemas for text chunks and chunking operations."""

from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class ChunkingConfig(BaseModel):
    """Configuration settings controlling text chunking behavior."""

    target_size: int = Field(
        default=1200,
        gt=0,
        description="Target soft maximum character count for a chunk before starting a new one.",
    )
    max_size: int = Field(
        default=1600,
        gt=0,
        description="Hard maximum character count limit for a chunk.",
    )
    overlap: int = Field(
        default=200,
        ge=0,
        description="Character count overlap from previous chunk to maintain semantic continuity.",
    )

    @model_validator(mode="after")
    def validate_sizes(self) -> "ChunkingConfig":
        """Ensure chunk size configuration parameters are logically consistent."""
        if self.target_size > self.max_size:
            raise ValueError(
                f"target_size ({self.target_size}) cannot exceed max_size ({self.max_size})."
            )
        if self.overlap >= self.target_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be strictly less than target_size ({self.target_size})."
            )
        return self


class Chunk(BaseModel):
    """Structured representation of a text chunk with strict page provenance."""

    chunk_id: str = Field(description="Unique, deterministic identifier for the chunk")
    document_id: str = Field(description="Identifier of the parent document")
    page_number: int = Field(ge=1, description="1-based source page number")
    chunk_index: int = Field(ge=0, description="Sequential index of chunk within the document")
    text: str = Field(description="Cleaned text content of the chunk")
    char_count: int = Field(ge=0, description="Character count of chunk text")
    word_count: int = Field(ge=0, description="Word count of chunk text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional extra provenance context")


class TextCleanRequest(BaseModel):
    """Request payload for text cleaning development endpoint."""

    text: str = Field(description="Raw text to be cleaned")


class TextCleanResponse(BaseModel):
    """Response payload for text cleaning development endpoint."""

    original_length: int
    cleaned_length: int
    cleaned_text: str


class ChunkPageRequest(BaseModel):
    """Request payload for single-page chunking development endpoint."""

    document_id: str = Field(description="Document ID")
    page_number: int = Field(ge=1, description="1-based page number")
    text: str = Field(description="Page text content to chunk")
    config: Optional[ChunkingConfig] = Field(default=None, description="Optional chunking configuration")


class ChunkPageResponse(BaseModel):
    """Response payload for chunking endpoint."""

    document_id: str
    page_number: int
    total_chunks: int
    chunks: list[Chunk]


class DocumentChunksResponse(BaseModel):
    """Response payload for document-level chunking."""

    document_id: str
    total_pages_processed: int
    total_chunks: int
    chunks: list[Chunk]
