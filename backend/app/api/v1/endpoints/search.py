"""FastAPI semantic search endpoint for BookRAG AI.

Provides a thin API layer exposing semantic vector retrieval over indexed chunks.
"""

from fastapi import APIRouter, Depends, status

from app.api.v1.dependencies import get_search_service
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search.service import SearchService

router = APIRouter(tags=["Search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Search with Cross-Encoder Reranking",
    description=(
        "Execute two-stage semantic retrieval and precision reranking across indexed chunks. "
        "First stage retrieves candidate_k nearest chunks via dense vector search (FAISS / pgvector); "
        "second stage applies a Cross-Encoder transformer to score joint (query, passage) pairs. "
        "similarity_score represents first-stage dense retrieval relevance; "
        "reranker_score represents second-stage cross-encoder relevance. "
        "NEITHER score represents answer confidence, probability, or factual correctness."
    ),
)
async def semantic_search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Execute semantic vector search and precision cross-encoder reranking."""
    return service.search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        candidate_k=request.candidate_k,
        enable_reranking=request.enable_reranking,
    )
