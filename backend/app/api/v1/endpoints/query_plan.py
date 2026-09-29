"""FastAPI endpoint for Phase 11 Query Understanding and Planning.

Provides POST /api/v1/query-plan for analyzing user queries, extracting constraints and entities,
and previewing deterministic retrieval execution plans.
"""

from fastapi import APIRouter, Depends, status

from app.schemas.query_plan import QueryPlan, QueryPlanRequest
from app.services.query_understanding.service import QueryUnderstandingService

router = APIRouter(tags=["Query Planning"])

_query_understanding_service: QueryUnderstandingService | None = None


def get_query_understanding_service() -> QueryUnderstandingService:
    """Dependency provider for QueryUnderstandingService."""
    global _query_understanding_service
    if _query_understanding_service is None:
        _query_understanding_service = QueryUnderstandingService()
    return _query_understanding_service


@router.post(
    "/query-plan",
    response_model=QueryPlan,
    status_code=status.HTTP_200_OK,
    summary="Generate Query Execution Plan",
    description=(
        "Analyzes a natural language question, classifies query intent according to the "
        "Phase 11 taxonomy (factual, definition, list, comparison, causal, procedural, "
        "location, summary, multi_hop, unknown), extracts explicit constraints (chapter, year, page), "
        "identifies key entities, and produces 1 to 3 targeted retrieval queries for the RAG pipeline."
    ),
)
async def generate_query_plan(
    request: QueryPlanRequest,
    service: QueryUnderstandingService = Depends(get_query_understanding_service),
) -> QueryPlan:
    """Analyze query and return structured QueryPlan contract."""
    return service.analyze_query(request.query)
