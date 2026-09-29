"""Pydantic schemas for Phase 9 Groundedness & Hallucination Control.

Defines API contracts, claim decomposition data models, NLI provenance records,
and safe final decision reporting.

CRITICAL DISTINCTION:
The groundedness validation in this system evaluates:
"Is the generated answer supported by the retrieved book evidence?"
It does NOT determine whether the book itself is objectively or factually true in the real world.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.generation import (
    DEFAULT_GENERATION_CANDIDATE_K,
    DEFAULT_GENERATION_TOP_K,
    MAX_GENERATION_CANDIDATE_K,
    MAX_GENERATION_TOP_K,
    GenerationEvidenceItem,
)


class ExtractedClaim(BaseModel):
    """A single deterministic claim decomposed from a generated answer."""

    claim_index: int = Field(ge=0, description="0-based sequential index of the claim in the answer.")
    claim_text: str = Field(min_length=1, description="Normalized text of the extracted claim sentence.")


class ClaimEvidenceProvenance(BaseModel):
    """Evidence passage metadata retained when evaluating a claim against a passage."""

    chunk_id: str = Field(description="Unique source chunk identifier")
    document_id: str = Field(description="Source document identifier")
    page_number: int = Field(ge=1, description="1-based page number where the chunk appears")
    chunk_index: int = Field(ge=0, description="Sequential index of the chunk within the document")
    source_text: str = Field(description="Full text of the retrieved supporting passage")
    rank: Optional[int] = Field(default=None, ge=1, description="1-based ranking position of the chunk")
    similarity_score: Optional[float] = Field(default=None, description="Dense retrieval similarity score")
    reranker_score: Optional[float] = Field(default=None, description="Cross-encoder reranking score")
    nli_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized NLI score for this chunk")


class ClaimResult(BaseModel):
    """Validation outcome for a single decomposed claim evaluated against evidence."""

    claim_index: int = Field(ge=0, description="0-based sequential index of the claim")
    claim_text: str = Field(description="Text of the evaluated claim")
    status: Literal["entailed", "contradicted", "unsupported", "conflicted"] = Field(
        description="Grounding status derived from NLI thresholds."
    )
    entailment_score: float = Field(ge=0.0, le=1.0, description="Highest entailment probability across evidence")
    contradiction_score: float = Field(ge=0.0, le=1.0, description="Highest contradiction probability across evidence")
    neutral_score: float = Field(ge=0.0, le=1.0, description="Neutral score corresponding to strongest evidence")
    supporting_evidence: Optional[ClaimEvidenceProvenance] = Field(
        default=None,
        description="Provenance of the evidence chunk with the strongest entailment score.",
    )
    contradicting_evidence: List[ClaimEvidenceProvenance] = Field(
        default_factory=list,
        description="Provenance of evidence chunks with strong contradiction scores.",
    )


class GroundingReport(BaseModel):
    """Aggregated groundedness evaluation report for an answer."""

    groundedness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of supported claims to total claims (system evaluation metric, NOT factual truth probability).",
    )
    total_claims: int = Field(ge=0, description="Total number of substantive claims decomposed from the answer.")
    supported_claims: int = Field(ge=0, description="Number of claims classified as 'entailed'.")
    unsupported_claims: int = Field(ge=0, description="Number of claims classified as 'unsupported'.")
    contradicted_claims: int = Field(ge=0, description="Number of claims classified as 'contradicted'.")
    conflicted_claims: int = Field(ge=0, description="Number of claims classified as 'conflicted'.")
    overall_status: Literal["grounded", "unsupported", "contradicted", "conflicted", "empty"] = Field(
        description="Overall status of the generated answer."
    )
    claim_results: List[ClaimResult] = Field(
        default_factory=list,
        description="Detailed validation result for each individual claim.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of grounding evaluation or safe refusal reason.",
    )


class GroundedAnswerRequest(BaseModel):
    """Request payload for grounded abstractive question answering with NLI validation."""

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
        description="Optional first-stage candidate pool size retrieved from FAISS before cross-encoder reranking.",
        examples=[20],
    )
    enable_reranking: Optional[bool] = Field(
        default=None,
        description="Optional override to enable or disable Phase 6 Cross-Encoder reranking.",
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
    require_all_claims_supported: Optional[bool] = Field(
        default=None,
        description="Whether to enforce that all claims must be supported without contradiction to accept the answer.",
        examples=[True],
    )
    entailment_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum entailment probability required for an evidence chunk to support a claim.",
        examples=[0.80],
    )
    contradiction_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum contradiction probability required to flag an evidence chunk as contradicting a claim.",
        examples=[0.80],
    )
    top_k_evidence: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of evidence chunks to evaluate against each claim.",
        examples=[5],
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
    def validate_candidate_k_bounds(self) -> "GroundedAnswerRequest":
        """Verify candidate_k >= top_k when explicitly supplied."""
        if self.candidate_k is not None and self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be greater than or equal to top_k ({self.top_k})."
            )
        return self


class GroundedAnswerResponse(BaseModel):
    """API response contract for grounded abstractive question answering.

    SEMANTIC PRINCIPLES:
    - The generated answer is synthesized strictly from retrieved evidence and verified with NLI.
    - If the answer contains unsupported, contradicted, or conflicted claims under the safe policy,
      the answer text is suppressed (answer=None, answerable=False, grounded=False) with a safe refusal reason.
    - Complete claim-level validation results and evidence provenance are preserved for transparency and debugging.
    - groundedness_score is a system evaluation metric, NOT factual truth probability.
    """

    query: str = Field(description="Normalized input question string")
    answer: Optional[str] = Field(
        default=None,
        description="Safe final answer if validated as grounded, or null if refused/unsupported.",
    )
    answerable: bool = Field(
        description="Whether sufficient evidence was available to produce a valid grounded answer.",
    )
    grounded: bool = Field(
        description="Whether the answer passed groundedness validation.",
    )
    groundedness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of supported claims to total claims.",
    )
    grounding_status: str = Field(
        description="Overall grounding status (grounded, unsupported, contradicted, conflicted, empty, disabled).",
    )
    claims: List[ClaimResult] = Field(
        default_factory=list,
        description="Decomposed claim validation results with provenance.",
    )
    evidence: List[GenerationEvidenceItem] = Field(
        default_factory=list,
        description="Retrieved evidence chunks evaluated during generation and grounding.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Safe refusal explanation or grounding evaluation summary.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Generation model identifier (e.g. google/flan-t5-base).",
    )
    grounding_model_name: Optional[str] = Field(
        default=None,
        description="NLI model identifier (e.g. cross-encoder/nli-deberta-v3-base).",
    )
