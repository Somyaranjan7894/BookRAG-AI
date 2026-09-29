"""FastAPI Extractive Question Answering endpoint for BookRAG AI.

Provides a clean API layer orchestrating two-stage retrieval (FAISS + Cross-Encoder)
and extractive answer span selection (RoBERTa SQuAD2) over indexed documents.
"""

from fastapi import APIRouter, Depends, status

from app.api.v1.endpoints.search import get_search_service
from app.schemas.qa import QARequest, QAResponse
from app.services.qa.service import QAService
from app.services.search.service import SearchService

router = APIRouter(tags=["Question Answering"])

_qa_service: QAService | None = None


def get_qa_service() -> QAService:
    """Dependency provider for QAService with singleton model caching."""
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service


@router.post(
    "/qa",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK,
    summary="Extractive Question Answering",
    description=(
        "Ask a natural language question against indexed books and extract the best supported answer span. "
        "Pipeline: Dense FAISS Retrieval -> Candidate Pool -> Cross-Encoder Reranker -> Top Evidence -> "
        "RoBERTa SQuAD2 Extractive QA -> Best Answer Span with complete source provenance. "
        "The model extracts text spans from the evidence; it does NOT generate or hallucinate new facts. "
        "If evidence is insufficient or unanswerable, returns a structured no-answer response."
    ),
)
async def ask_question(
    request: QARequest,
    search_service: SearchService = Depends(get_search_service),
    qa_service: QAService = Depends(get_qa_service),
) -> QAResponse:
    """Execute two-stage evidence retrieval followed by extractive answer span extraction."""
    # 1. Retrieve and rerank top evidence using SearchService
    search_response = search_service.search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        candidate_k=request.candidate_k,
        enable_reranking=request.enable_reranking,
    )

    # 2. Extract best supported answer span across reranked evidence chunks
    return qa_service.answer_question(
        query=request.query,
        evidence=search_response.results,
        no_answer_threshold=request.no_answer_threshold,
    )
