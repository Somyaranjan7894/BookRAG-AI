"""Application service for Phase 11 Query Understanding.

Provides the public QueryUnderstandingService for converting natural language user
questions into structured QueryPlan execution contracts.
"""

from typing import Optional

from app.core.logging import get_logger
from app.schemas.query_plan import QueryPlan
from app.services.query_understanding.exceptions import InvalidQueryError
from app.services.query_understanding.planner import QueryPlanner

logger = get_logger(__name__)


class QueryUnderstandingService:
    """Service orchestrating query validation, analysis, and query plan creation."""

    def __init__(self, planner: Optional[QueryPlanner] = None) -> None:
        """Initialize QueryUnderstandingService with optional planner injection."""
        self.planner = planner or QueryPlanner()

    def analyze_query(self, query: str) -> QueryPlan:
        """Analyze a user query and produce a validated, typed QueryPlan.

        Args:
            query: Natural language question string.

        Returns:
            Structured QueryPlan with query type, entities, constraints, and retrieval queries.

        Raises:
            InvalidQueryError: If query is None, empty, or whitespace-only.
        """
        if query is None or not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("Question query cannot be empty or whitespace-only.")

        plan = self.planner.plan(query)

        logger.info(
            "Constructed QueryPlan for query='%s' (type=%s, queries=%d, multi_evidence=%s)",
            plan.normalized_query,
            plan.query_type.value,
            len(plan.retrieval_queries),
            plan.requires_multiple_evidence,
        )

        return plan
