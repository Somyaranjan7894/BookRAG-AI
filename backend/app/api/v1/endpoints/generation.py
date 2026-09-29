"""FastAPI Abstractive Question Answering endpoint for BookRAG AI.

Provides the public POST /api/v1/answer endpoint, orchestrating dense FAISS retrieval,
Cross-Encoder precision reranking, context budgeting, and FLAN-T5 answer generation.
"""

from fastapi import APIRouter, Depends, status

from app.api.v1.endpoints.search import get_search_service
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.services.generation.service import GenerationService
from app.services.search.service import SearchService

router = APIRouter(tags=["Answer Generation"])

_generation_service: GenerationService | None = None


def get_generation_service() -> GenerationService:
    """Dependency provider for GenerationService with singleton model caching."""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service


@router.post(
    "/answer",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Abstractive Question Answering (FLAN-T5)",
    description=(
        "Ask a natural language question against indexed book documents and receive an abstractive answer "
        "synthesized strictly from retrieved and reranked evidence using FLAN-T5. "
        "Pipeline: Query -> FAISS Dense Retrieval -> Candidate Pool -> Cross-Encoder Reranking -> "
        "Evidence Context Budgeting -> FLAN-T5 Grounded Generation -> Structured Answer with Provenance. "
        "The retrieved evidence remains the sole source of truth; if evidence is unavailable or empty, "
        "the service returns a structured unanswerable response rather than hallucinating from world knowledge."
    ),
)
async def generate_answer(
    request: GenerationRequest,
    search_service: SearchService = Depends(get_search_service),
    generation_service: GenerationService = Depends(get_generation_service),
) -> GenerationResponse:
    """Execute two-stage evidence retrieval followed by grounded abstractive generation."""
    # 1. Retrieve and rerank top evidence using SearchService
    search_response = search_service.search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        candidate_k=request.candidate_k,
        enable_reranking=request.enable_reranking,
    )

    # 2. Synthesize answer from top evidence context
    return generation_service.generate_answer(
        query=request.query,
        evidence=search_response.results,
        max_new_tokens=request.max_new_tokens,
        num_beams=request.num_beams,
        do_sample=request.do_sample,
        temperature=request.temperature,
    )
