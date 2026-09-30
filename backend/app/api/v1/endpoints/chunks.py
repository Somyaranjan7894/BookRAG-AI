"""Development and testing endpoints for text cleaning and intelligent chunking."""

from fastapi import APIRouter, Depends, status

from app.api.v1.dependencies import get_chunker, get_text_cleaner
from app.schemas.chunk import (
    ChunkPageRequest,
    ChunkPageResponse,
    TextCleanRequest,
    TextCleanResponse,
)
from app.services.text.chunker import Chunker
from app.services.text.cleaner import TextCleaner

router = APIRouter(prefix="/chunks", tags=["Chunks"])


@router.post(
    "/clean",
    response_model=TextCleanResponse,
    status_code=status.HTTP_200_OK,
    summary="Clean Text",
    description="Development endpoint to test conservative text cleaning and normalization.",
)
async def clean_text(
    payload: TextCleanRequest,
    cleaner: type[TextCleaner] = Depends(get_text_cleaner),
) -> TextCleanResponse:
    """Apply deterministic text cleaner to raw text."""
    cleaned = cleaner.clean(payload.text)
    return TextCleanResponse(
        original_length=len(payload.text),
        cleaned_length=len(cleaned),
        cleaned_text=cleaned,
    )


@router.post(
    "/chunk-page",
    response_model=ChunkPageResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk Single Page",
    description="Development endpoint to chunk a single page's text into structured Chunks.",
)
async def chunk_single_page(
    payload: ChunkPageRequest,
    chunker: Chunker = Depends(get_chunker),
) -> ChunkPageResponse:
    """Generate structured Chunks for a single page with provenance."""
    chunks = chunker.chunk_text(
        text=payload.text,
        document_id=payload.document_id,
        page_number=payload.page_number,
        config=payload.config,
    )
    return ChunkPageResponse(
        document_id=payload.document_id,
        page_number=payload.page_number,
        total_chunks=len(chunks),
        chunks=chunks,
    )
