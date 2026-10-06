"""Pydantic schemas and enums for Phase 22 Controlled Question Generation in BookRAG AI.

Provides internal contracts, candidate representation, validated question schemas
with complete provenance, difficulty heuristics, diversity support, and diagnostic reporting.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class QuestionType(str, Enum):
    """Controlled taxonomy of question types matching Step 3 and analytical categories."""

    DIRECT_FACT = "direct_fact"
    FACTUAL = "factual"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    NUMERICAL = "numerical"
    NUMERICAL_FACT = "numerical_fact"
    REASONING = "reasoning"
    MULTI_PAGE_SYNTHESIS = "multi_page_synthesis"

    # Wh-question subtypes for backward compatibility & fine-grained classification
    WHO = "who"
    WHAT = "what"
    WHEN = "when"
    WHERE = "where"
    WHY = "why"
    HOW = "how"
    HOW_MANY = "how_many"


class QuestionDifficulty(str, Enum):
    """Defensible heuristic difficulty level for generated questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AnswerCandidate(BaseModel):
    """An extracted candidate answer span grounded directly in source book evidence."""

    answer_text: str = Field(..., min_length=1, description="Exact text of the candidate answer.")
    source_text: str = Field(..., min_length=1, description="Source passage containing the answer.")
    start_offset: Optional[int] = Field(default=None, ge=0, description="Character start offset in source_text.")
    end_offset: Optional[int] = Field(default=None, ge=0, description="Character end offset in source_text.")
    document_id: str = Field(..., min_length=1, description="Document identifier.")
    chunk_id: str = Field(..., min_length=1, description="Unique chunk identifier.")
    chunk_ids: List[str] = Field(default_factory=list, description="All source chunk IDs supporting the answer.")
    page_number: int = Field(..., ge=1, description="1-based book page number.")
    page_numbers: List[int] = Field(default_factory=list, description="All source page numbers supporting the answer.")
    chunk_index: Optional[int] = Field(default=None, ge=0, description="0-based chunk index within document.")
    candidate_type: QuestionType = Field(default=QuestionType.FACTUAL, description="Inferred candidate question type.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional candidate metadata.")

    @model_validator(mode="before")
    @classmethod
    def populate_lists(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("chunk_ids") and data.get("chunk_id"):
                data["chunk_ids"] = [data["chunk_id"]]
            if not data.get("page_numbers") and data.get("page_number"):
                data["page_numbers"] = [data["page_number"]]
        return data


class QuestionCandidate(BaseModel):
    """Unvalidated question candidate generated for an answer candidate."""

    candidate_id: Optional[str] = Field(default=None, description="Deterministic candidate identifier.")
    question_text: str = Field(..., min_length=1, description="Generated question text.")
    answer_candidate: AnswerCandidate = Field(..., description="Target answer candidate.")
    question_type: QuestionType = Field(default=QuestionType.FACTUAL, description="Classified question type.")
    difficulty: QuestionDifficulty = Field(default=QuestionDifficulty.EASY, description="Heuristic difficulty level.")
    raw_model_output: Optional[str] = Field(default=None, description="Raw generation output string.")


class RejectedCandidateInfo(BaseModel):
    """Diagnostic detail for a candidate question rejected during validation or deduplication."""

    candidate_id: str = Field(..., description="Unique candidate identifier.")
    question_text: str = Field(..., description="Generated candidate question text.")
    answer_text: str = Field(..., description="Target candidate answer span.")
    rejection_reason: str = Field(..., description="Diagnostic reason for rejection.")
    rejection_category: str = Field(default="validation_failure", description="Category: quality, answerability, duplicate, outside_knowledge.")
    chunk_id: str = Field(..., description="Source chunk ID.")
    chunk_ids: List[str] = Field(default_factory=list, description="All source chunk IDs.")
    page_number: int = Field(..., description="Source page number.")
    page_numbers: List[int] = Field(default_factory=list, description="All source page numbers.")
    question_type: QuestionType = Field(default=QuestionType.FACTUAL, description="Candidate question type.")


class GeneratedQuestion(BaseModel):
    """A fully validated, evidence-grounded question with complete book provenance."""

    question_id: Optional[str] = Field(default=None, description="Unique deterministic question identifier.")
    question: str = Field(..., min_length=1, description="Validated natural-language question.")
    answer: str = Field(..., min_length=1, description="Verified target answer.")
    question_type: QuestionType = Field(..., description="Question taxonomy type.")
    difficulty: QuestionDifficulty = Field(..., description="Heuristic difficulty level.")
    document_id: str = Field(..., min_length=1, description="Book document identifier.")
    chunk_id: str = Field(..., min_length=1, description="Durable chunk ID containing the source evidence.")
    chunk_ids: List[str] = Field(default_factory=list, description="All source chunk IDs supporting the evidence.")
    page_number: int = Field(..., ge=1, description="1-based page number where evidence appears.")
    page_numbers: List[int] = Field(default_factory=list, description="All source page numbers supporting the evidence.")
    source_text: str = Field(..., min_length=1, description="Source book passage supporting the Q&A pair.")
    start_offset: Optional[int] = Field(default=None, ge=0, description="Answer start offset in source passage.")
    end_offset: Optional[int] = Field(default=None, ge=0, description="Answer end offset in source passage.")
    qa_predicted_answer: Optional[str] = Field(default=None, description="Extractive QA model predicted answer.")
    qa_confidence_score: Optional[float] = Field(default=None, ge=0.0, description="Extractive QA confidence score.")
    validation_status: str = Field(default="validated", description="Validation status (e.g. 'validated').")
    rejection_reason: Optional[str] = Field(default=None, description="Diagnostic rejection reason if candidate failed.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic and provenance metadata.")

    @model_validator(mode="before")
    @classmethod
    def populate_provenance_lists(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("chunk_ids") and data.get("chunk_id"):
                data["chunk_ids"] = [data["chunk_id"]]
            if not data.get("page_numbers") and data.get("page_number"):
                data["page_numbers"] = [data["page_number"]]
        return data


class QuestionGenerationRequest(BaseModel):
    """Request payload for generating validated questions from a book document."""

    document_id: str = Field(..., min_length=1, description="Target document ID to generate questions from.")
    count: int = Field(default=5, ge=1, le=50, description="Desired number of validated questions.")
    difficulty: Optional[QuestionDifficulty] = Field(default=None, description="Target difficulty filter.")
    question_type: Optional[QuestionType] = Field(default=None, description="Optional question type filter.")
    chapter: Optional[int] = Field(default=None, ge=1, description="Optional explicit chapter constraint.")
    page_number: Optional[int] = Field(default=None, ge=1, description="Optional explicit page number constraint.")
    include_rejected: bool = Field(default=False, description="Whether to include rejected candidate diagnostics in response.")
    ensure_diversity: bool = Field(default=True, description="Whether to balance across distinct question types where evidence permits.")


class QuestionGenerationResponse(BaseModel):
    """Response payload containing validated, deduplicated questions with provenance."""

    document_id: str = Field(..., description="Target document ID.")
    requested_count: int = Field(..., ge=1, description="Number of questions requested.")
    generated_candidates: int = Field(..., ge=0, description="Total raw candidate questions generated.")
    validated_count: int = Field(..., ge=0, description="Count of candidates that passed QA validation.")
    returned_count: int = Field(..., ge=0, description="Number of validated, deduplicated questions returned.")
    questions: List[GeneratedQuestion] = Field(default_factory=list, description="Validated questions.")
    rejected_candidates: List[RejectedCandidateInfo] = Field(default_factory=list, description="Diagnostic rejected candidates.")
    rejection_summary: Dict[str, int] = Field(default_factory=dict, description="Rejection reason counts.")
    latency_ms: Optional[float] = Field(default=None, description="Total generation and validation latency in milliseconds.")
