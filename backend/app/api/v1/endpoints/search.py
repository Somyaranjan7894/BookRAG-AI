"""FastAPI semantic search endpoint for BookRAG AI.

Provides a thin API layer exposing semantic vector retrieval over indexed chunks.
"""

from fastapi import APIRouter, Depends, status

from app.schemas.search import SearchRequest, SearchResponse
from app.services.search.service import SearchService

router = APIRouter(tags=["Search"])

# Cached default search service instance sharing the development retrieval service
_search_service: SearchService | None = None


def get_search_service() -> SearchService:
    """Dependency provider for SearchService.

    Reuses the shared development RetrievalService so indexed chunks from
    /api/v1/retrieval are immediately searchable.
    """
    global _search_service
    if _search_service is None:
        from app.api.v1.endpoints.retrieval import get_dev_retrieval_service

        dev_retrieval_svc = get_dev_retrieval_service()
        _search_service = SearchService(retrieval_service=dev_retrieval_svc)
    return _search_service


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Search",
    description=(
        "Execute natural language semantic search across indexed document chunks. "
        "Returns top-K candidate chunks ranked by cosine similarity score with complete "
        "provenance (chunk_id, document_id, page_number, text). "
        "similarity_score represents vector-space semantic retrieval relevance, "
        "NOT confidence, probability, or factual correctness."
    ),
)
async def semantic_search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Execute semantic vector search for a natural language query."""
    return service.search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
    )
