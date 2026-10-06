"""Schema definitions for BookRAG AI Evaluation Datasets.

Defines standardized Pydantic models for evaluation queries, ground truth
provenance mappings, answerability labels, and dataset envelopes.
Distinguishes SAMPLE / DEVELOPMENT DATA from HUMAN-VERIFIED GOLDEN DATA.
Explicitly enforces verification status: VERIFIED, NOT_VERIFIED, BLOCKED.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


VerificationStatus = Literal["VERIFIED", "NOT_VERIFIED", "BLOCKED"]
EvaluationDatasetType = Literal["golden", "sample", "human_verified"]

SUPPORTED_CATEGORIES = (
    "direct_fact",
    "definition",
    "explanation",
    "multi_page",
    "multi_page_synthesis",
    "comparison",
    "numerical_fact",
    "reasoning",
    "unanswerable",
    "ambiguous",
    # Legacy categories preserved for backwards compatibility with dev_sample.json
    "factual",
    "causal",
    "unanswerable_out_of_scope",
)


class EvaluationItem(BaseModel):
    """Single question evaluation item with verifiable relevance annotations."""

    id: str = Field(description="Unique evaluation question identifier")
    question: str = Field(description="Natural language evaluation prompt or question")
    document_id: str = Field(description="Target book or document identifier")
    relevant_page_numbers: List[int] = Field(
        default_factory=list,
        description="List of 1-based page numbers containing direct answer evidence",
    )
    relevant_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of specific chunk IDs containing supporting evidence",
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description="Human-authored reference answer or key factual points (if answerable)",
    )
    answerable: bool = Field(
        default=True,
        description="Whether this question can be answered from the book contents alone",
    )
    dataset_type: str = Field(
        default="golden",
        description="Dataset type: 'golden', 'sample', or 'human_verified'",
    )
    verification_status: VerificationStatus = Field(
        default="NOT_VERIFIED",
        description="Dataset verification status: 'VERIFIED', 'NOT_VERIFIED', or 'BLOCKED'",
    )
    category: str = Field(
        default="direct_fact",
        description=(
            "Question category: direct_fact, definition, explanation, multi_page, multi_page_synthesis, "
            "comparison, numerical_fact, reasoning, unanswerable, ambiguous (or legacy factual/causal)"
        ),
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Exact textual evidence, citation quotes, or reference excerpts from the source document",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Evaluation rationale, edge case notes, or test objectives",
    )

    @model_validator(mode="before")
    @classmethod
    def map_phase23_aliases(cls, data: Any) -> Any:
        """Map Phase 23 alias names to canonical EvaluationItem schema attributes."""
        if isinstance(data, dict):
            # Map question_id -> id
            if "question_id" in data and "id" not in data:
                data["id"] = data["question_id"]
            # Map page_numbers -> relevant_page_numbers
            if "page_numbers" in data and "relevant_page_numbers" not in data:
                data["relevant_page_numbers"] = data["page_numbers"]
            # Map chunk_ids -> relevant_chunk_ids
            if "chunk_ids" in data and "relevant_chunk_ids" not in data:
                data["relevant_chunk_ids"] = data["chunk_ids"]
            # Map source_evidence -> evidence
            if "source_evidence" in data and "evidence" not in data:
                data["evidence"] = data["source_evidence"]
            # Map query_type -> category
            if "query_type" in data and "category" not in data:
                data["category"] = data["query_type"]
            # Map answerability -> answerable
            if "answerability" in data and "answerable" not in data:
                data["answerable"] = data["answerability"]
        return data

    @property
    def question_id(self) -> str:
        """Phase 23 alias for id."""
        return self.id

    @property
    def page_numbers(self) -> List[int]:
        """Phase 23 alias for relevant_page_numbers."""
        return self.relevant_page_numbers

    @property
    def chunk_ids(self) -> List[str]:
        """Phase 23 alias for relevant_chunk_ids."""
        return self.relevant_chunk_ids

    @property
    def source_evidence(self) -> Optional[str]:
        """Phase 23 alias for evidence."""
        return self.evidence

    @property
    def query_type(self) -> str:
        """Phase 23 alias for category."""
        return self.category

    @property
    def answerability(self) -> bool:
        """Phase 23 alias for answerable."""
        return self.answerable

    @model_validator(mode="after")
    def validate_answerability_and_evidence(self) -> "EvaluationItem":
        """Validate logical consistency between answerability, evidence, and answers."""
        if not self.answerable:
            # For unanswerable or ambiguous queries, we should not have positive relevant evidence
            if self.relevant_chunk_ids and self.category in ("unanswerable", "unanswerable_out_of_scope"):
                raise ValueError(
                    f"Item '{self.id}' is marked unanswerable but has non-empty relevant_chunk_ids."
                )
        return self


class EvaluationDataset(BaseModel):
    """Envelope for a collection of evaluation items against a target document."""

    name: str = Field(description="Name of the evaluation dataset")
    version: str = Field(default="1.0.0", description="Dataset semantic version")
    dataset_type: str = Field(
        default="golden",
        description="Dataset verification level: 'golden', 'sample', or 'human_verified'",
    )
    verification_status: VerificationStatus = Field(
        default="NOT_VERIFIED",
        description="Benchmark verification status: 'VERIFIED', 'NOT_VERIFIED', or 'BLOCKED'",
    )
    description: str = Field(description="Overview of the dataset contents and methodology")
    document_id: str = Field(description="Target document ID evaluated")
    document_title: str = Field(description="Title of the target document")
    items: List[EvaluationItem] = Field(
        default_factory=list,
        description="List of evaluation items in this benchmark",
    )

    @property
    def total_count(self) -> int:
        """Total number of evaluation items."""
        return len(self.items)

    @property
    def answerable_count(self) -> int:
        """Count of answerable evaluation items."""
        return sum(1 for item in self.items if item.answerable)

    @property
    def unanswerable_count(self) -> int:
        """Count of unanswerable evaluation items."""
        return sum(1 for item in self.items if not item.answerable)

    @property
    def category_counts(self) -> Dict[str, int]:
        """Distribution of questions across categories."""
        counts: Dict[str, int] = {}
        for item in self.items:
            counts[item.category] = counts.get(item.category, 0) + 1
        return counts

    @property
    def verification_status_counts(self) -> Dict[str, int]:
        """Distribution of items across verification statuses."""
        counts: Dict[str, int] = {"VERIFIED": 0, "NOT_VERIFIED": 0, "BLOCKED": 0}
        for item in self.items:
            counts[item.verification_status] = counts.get(item.verification_status, 0) + 1
        return counts

    def get_item(self, item_id: str) -> Optional[EvaluationItem]:
        """Retrieve evaluation item by identifier."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
