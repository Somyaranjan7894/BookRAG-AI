"""FastAPI endpoint for Phase 12 Question Generation in BookRAG AI."""

from typing import Optional
from fastapi import APIRouter, Depends, status

from app.core.logging import get_logger
from app.schemas.question_generation import (
    QuestionGenerationRequest,
    QuestionGenerationResponse,
)
from app.services.question_generation.service import QuestionGenerationService

logger = get_logger(__name__)

router = APIRouter(prefix="/questions", tags=["Question Generation"])

_qg_service: Optional[QuestionGenerationService] = None


def get_question_generation_service() -> QuestionGenerationService:
    """Dependency provider for QuestionGenerationService with singleton model caching."""
    global _qg_service
    if _qg_service is None:
        _qg_service = QuestionGenerationService()
    return _qg_service


@router.post(
    "/generate",
    response_model=QuestionGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate evidence-grounded questions from a book document",
    description=(
        "Extracts candidate answer spans from document evidence, generates targeted reading comprehension "
        "questions with a local model, and verifies each candidate with extractive QA and strict answer matching."
    ),
)
async def generate_questions_endpoint(
    request: QuestionGenerationRequest,
    service: QuestionGenerationService = Depends(get_question_generation_service),
) -> QuestionGenerationResponse:
    """Generate validated, deduplicated questions with full book provenance."""
    logger.info(
        "Question generation request for document '%s' (requested_count=%d, difficulty=%s, chapter=%s, page=%s)",
        request.document_id,
        request.count,
        request.difficulty,
        request.chapter,
        request.page_number,
    )
    return service.generate_questions(request)
