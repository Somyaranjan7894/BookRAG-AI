"""Pydantic schemas for Phase 10 Citation and Provenance Mapping.

Defines API contracts and data models for mapping verified grounded claims
to exact book evidence chunks with deterministic citation identifiers.

KEY ARCHITECTURAL PRINCIPLE:
"The generator generates the answer. The system assigns citations from verified evidence."
FLAN-T5 is never asked to generate citations. Citations are constructed solely from evidence
verified by Phase 9 NLI groundedness validation.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """A unique, deduplicated citation referencing an exact book evidence chunk."""

    citation_id: str = Field(
        description="Response-local deterministic citation identifier (e.g. cite_1, cite_2).",
        examples=["cite_1"],
    )
    document_id: str = Field(
        description="Source book document identifier.",
        examples=["deep_learning_handbook"],
    )
    chunk_id: str = Field(
        description="Durable unique source chunk identifier.",
        examples=["deep_learning_handbook_p012_c0003"],
    )
    page_number: int = Field(
        ge=1,
        description="1-based book page number where the supporting text appears.",
        examples=[12],
    )
    chunk_index: int = Field(
        ge=0,
        description="Sequential index of the chunk within the document.",
        examples=[3],
    )
    source_text: str = Field(
        description="Exact unedited text passage from the supporting chunk.",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="Dense retrieval similarity score from first-stage FAISS retrieval.",
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance score from second-stage reranking.",
    )
    evidence_rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Rank position of the chunk in the retrieved evidence set.",
    )


class ClaimCitationRef(BaseModel):
    """Citation reference attached to an individual decomposed claim."""

    citation_id: str = Field(
        description="Reference to unique citation_id in the top-level citations collection.",
        examples=["cite_1"],
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Source book document identifier.",
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Source chunk identifier.",
    )
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based book page number.",
    )
    chunk_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Chunk index within document.",
    )
    source_text: Optional[str] = Field(
        default=None,
        description="Supporting passage text snippet or full text.",
    )
    relation: Literal["supports", "contradicts"] = Field(
        default="supports",
        description="Semantic relation between the evidence and the claim.",
    )


class CitationMappingResult(BaseModel):
    """Internal container for the output of CitationService.build_citations."""

    citations: List[Citation] = Field(
        default_factory=list,
        description="Top-level deduplicated citations ordered by first verified appearance.",
    )
    claim_citations: Dict[int, List[ClaimCitationRef]] = Field(
        default_factory=dict,
        description="Mapping from claim_index to supporting citation references.",
    )
    contradicting_claim_citations: Dict[int, List[ClaimCitationRef]] = Field(
        default_factory=dict,
        description="Mapping from claim_index to contradicting citation references for conflicts.",
    )
