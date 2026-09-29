"""FastAPI Grounded Answer Generation endpoint for BookRAG AI Phase 9.

Provides the public POST /api/v1/grounded-answer endpoint, orchestrating dense FAISS retrieval,
Cross-Encoder precision reranking, FLAN-T5 abstractive generation, deterministic claim decomposition,
NLI groundedness validation (cross-encoder/nli-deberta-v3-base), and safe decision policy.
"""

from fastapi import APIRouter, Depends, status

from app.schemas.grounding import GroundedAnswerRequest, GroundedAnswerResponse
from app.services.grounding.orchestrator import GroundedAnswerService

router = APIRouter(tags=["Grounded Answer"])

_grounded_answer_service: GroundedAnswerService | None = None


def get_grounded_answer_service() -> GroundedAnswerService:
    """Dependency provider for GroundedAnswerService with cached singleton components."""
    global _grounded_answer_service
    if _grounded_answer_service is None:
        _grounded_answer_service = GroundedAnswerService()
    return _grounded_answer_service


@router.post(
    "/grounded-answer",
    response_model=GroundedAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Grounded Abstractive Question Answering with NLI Validation",
    description=(
        "Ask a natural language question against indexed book documents and receive a verified, "
        "grounded answer synthesized strictly from retrieved evidence and validated using NLI. "
        "Pipeline: Query -> FAISS Retrieval -> Cross-Encoder Reranker -> FLAN-T5 Generation -> "
        "Sentence-level Claim Decomposition -> DeBERTa-v3 NLI Verification -> Safe Decision Policy. "
        "If any substantive claims are unsupported, contradicted, or conflicting under the safe policy, "
        "the answer is safely suppressed (answer=null, answerable=false, grounded=false) with full "
        "claim-level diagnostic metadata and source evidence provenance retained."
    ),
)
async def generate_grounded_answer(
    request: GroundedAnswerRequest,
    grounded_service: GroundedAnswerService = Depends(get_grounded_answer_service),
) -> GroundedAnswerResponse:
    """Execute end-to-end grounded question answering pipeline with hallucination control."""
    return grounded_service.answer_with_grounding(request)
