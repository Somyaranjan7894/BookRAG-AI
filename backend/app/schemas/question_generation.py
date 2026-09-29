"""Pydantic schemas and enums for Phase 12 Question Generation in BookRAG AI."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Controlled taxonomy of question types."""

    FACTUAL = "factual"
    WHO = "who"
    WHAT = "what"
    WHEN = "when"
    WHERE = "where"
    WHY = "why"
    HOW = "how"
    HOW_MANY = "how_many"
    DEFINITION = "definition"
    COMPARISON = "comparison"


class QuestionDifficulty(str, Enum):
    """Heuristic difficulty level for generated questions."""

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
    page_number: int = Field(..., ge=1, description="1-based book page number.")
    chunk_index: Optional[int] = Field(default=None, ge=0, description="0-based chunk index within document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional candidate metadata.")


class QuestionCandidate(BaseModel):
    """Unvalidated question candidate generated for an answer candidate."""

    question_text: str = Field(..., min_length=1, description="Generated question text.")
    answer_candidate: AnswerCandidate = Field(..., description="Target answer candidate.")
    question_type: QuestionType = Field(default=QuestionType.FACTUAL, description="Classified question type.")
    difficulty: QuestionDifficulty = Field(default=QuestionDifficulty.EASY, description="Heuristic difficulty level.")
    raw_model_output: Optional[str] = Field(default=None, description="Raw generation output string.")


class GeneratedQuestion(BaseModel):
    """A fully validated, evidence-grounded question with complete book provenance."""

    question: str = Field(..., min_length=1, description="Validated natural-language question.")
    answer: str = Field(..., min_length=1, description="Verified target answer.")
    question_type: QuestionType = Field(..., description="Question taxonomy type.")
    difficulty: QuestionDifficulty = Field(..., description="Heuristic difficulty level.")
    document_id: str = Field(..., min_length=1, description="Book document identifier.")
    chunk_id: str = Field(..., min_length=1, description="Durable chunk ID containing the source evidence.")
    page_number: int = Field(..., ge=1, description="1-based page number where evidence appears.")
    source_text: str = Field(..., min_length=1, description="Source book passage supporting the Q&A pair.")
    start_offset: Optional[int] = Field(default=None, ge=0, description="Answer start offset in source passage.")
    end_offset: Optional[int] = Field(default=None, ge=0, description="Answer end offset in source passage.")
    qa_predicted_answer: Optional[str] = Field(default=None, description="Extractive QA model predicted answer.")
    qa_confidence_score: Optional[float] = Field(default=None, ge=0.0, description="Extractive QA confidence score.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic and provenance metadata.")


class QuestionGenerationRequest(BaseModel):
    """Request payload for generating validated questions from a book document."""

    document_id: str = Field(..., min_length=1, description="Target document ID to generate questions from.")
    count: int = Field(default=5, ge=1, le=50, description="Desired number of validated questions.")
    difficulty: Optional[QuestionDifficulty] = Field(default=None, description="Target difficulty filter.")
    chapter: Optional[int] = Field(default=None, ge=1, description="Optional explicit chapter constraint.")
    page_number: Optional[int] = Field(default=None, ge=1, description="Optional explicit page number constraint.")


class QuestionGenerationResponse(BaseModel):
    """Response payload containing validated, deduplicated questions with provenance."""

    document_id: str = Field(..., description="Target document ID.")
    requested_count: int = Field(..., ge=1, description="Number of questions requested.")
    generated_candidates: int = Field(..., ge=0, description="Total raw candidate questions generated.")
    validated_count: int = Field(..., ge=0, description="Count of candidates that passed QA validation.")
    returned_count: int = Field(..., ge=0, description="Number of validated, deduplicated questions returned.")
    questions: List[GeneratedQuestion] = Field(default_factory=list, description="Validated questions.")
