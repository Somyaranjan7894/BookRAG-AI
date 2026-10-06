"""Pydantic schemas for Phase 11 Query Understanding and Query Planning.

Defines the typed contracts between the user's natural language question and
subsequent retrieval / reasoning stages.

The QueryPlan provides retrieval strategy metadata. It does not determine
factual truth or replace answer validation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class QueryType(str, Enum):
    """Taxonomy of user query intents for retrieval and reasoning planning."""

    DIRECT_FACT = "direct_fact"
    FACTUAL = "factual"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    MULTI_PART = "multi_part"
    NUMERICAL_FACT = "numerical_fact"
    MULTI_PAGE = "multi_page"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"
    LIST = "list"
    CAUSAL = "causal"
    PROCEDURAL = "procedural"
    LOCATION = "location"
    SUMMARY = "summary"
    MULTI_HOP = "multi_hop"
    UNKNOWN = "unknown"


class ExpectedAnswerType(str, Enum):
    """Metadata describing the anticipated semantic category of the answer."""

    PERSON_ENTITY = "person/entity"
    DATE_YEAR = "date/year"
    LOCATION = "location"
    NUMBER = "number"
    EXPLANATION = "explanation"
    LIST = "list"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    UNKNOWN = "unknown"


class QueryConstraints(BaseModel):
    """Explicit constraints extracted from the user query.

    Constraints are ONLY populated when explicitly mentioned in the query text.
    They are NEVER inferred or invented.
    """

    chapter: Optional[int] = Field(
        default=None,
        description="Explicit chapter number stated in the question (e.g., Chapter 7).",
        examples=[7],
    )
    year: Optional[int] = Field(
        default=None,
        description="Explicit calendar year mentioned in the question (e.g., 2020, 1998).",
        examples=[2020],
    )
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Explicit 1-based page number stated in the question.",
        examples=[12],
    )
    page_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Explicit page range (start_page, end_page) stated in the question.",
        examples=[(10, 15)],
    )
    quoted_phrases: List[str] = Field(
        default_factory=list,
        description="Exact quoted phrases extracted from the query (e.g. 'quantum tunneling').",
    )
    named_entities: List[str] = Field(
        default_factory=list,
        description="Explicit named entities detected in the query.",
    )


class QueryPlan(BaseModel):
    """Typed query execution plan produced by QueryUnderstandingService."""

    original_query: str = Field(
        description="Unmodified raw user query string.",
        examples=["Compare the population of India and China in 2020."],
    )
    normalized_query: Optional[str] = Field(
        default=None,
        description="Safely normalized query string preserving technical symbols and numbers.",
        examples=["Compare the population of India and China in 2020."],
    )

    @model_validator(mode="before")
    @classmethod
    def populate_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("normalized_query"):
                data["normalized_query"] = data.get("original_query", "")
        return data

    query_type: QueryType = Field(
        description="Classified query taxonomy intent.",
        examples=[QueryType.COMPARISON],
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Key entities, technical terms, and concepts extracted from the query.",
        examples=[["India", "China", "2020"]],
    )
    constraints: QueryConstraints = Field(
        default_factory=QueryConstraints,
        description="Explicit constraints detected from query text.",
    )
    retrieval_queries: List[str] = Field(
        default_factory=list,
        description="Targeted retrieval queries (1 to 3 non-redundant queries).",
        examples=[["India population 2020", "China population 2020"]],
    )
    expected_answer_type: str = Field(
        default=ExpectedAnswerType.EXPLANATION.value,
        description="Approximate expected answer type for downstream metadata.",
        examples=["comparison"],
    )
    requires_multiple_evidence: bool = Field(
        default=False,
        description="Whether answering the query likely requires evidence across multiple chunks.",
    )
    comparison_aspects: List[str] = Field(
        default_factory=list,
        description="Extracted comparison subjects [A, B] for comparison queries.",
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Extracted sub-question components for multi-part queries.",
    )
    is_numerical: bool = Field(
        default=False,
        description="Whether query asks for a numerical value, derivative, or formula.",
    )


class QueryPlanRequest(BaseModel):
    """Request payload for standalone query planning endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language question string to analyze.",
        examples=["What is the difference between CNN and RNN?"],
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Query cannot be empty or whitespace-only.")
        return s
