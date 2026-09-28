"""Development and testing endpoints for text cleaning and intelligent chunking."""

from fastapi import APIRouter, status

from app.schemas.chunk import (
    ChunkPageRequest,
    ChunkPageResponse,
    DocumentChunksResponse,
    TextCleanRequest,
    TextCleanResponse,
)
from app.services.pdf.ingestion import PDFIngestionService
from app.services.text.chunker import Chunker
from app.services.text.cleaner import TextCleaner
from app.services.text.processor import TextProcessingService

router = APIRouter(prefix="/chunks", tags=["Chunks"])


@router.post(
    "/clean",
    response_model=TextCleanResponse,
    status_code=status.HTTP_200_OK,
    summary="Clean Text",
    description="Development endpoint to test conservative text cleaning and normalization.",
)
async def clean_text(payload: TextCleanRequest) -> TextCleanResponse:
    """Apply deterministic text cleaner to raw text."""
    cleaned = TextCleaner.clean(payload.text)
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
async def chunk_single_page(payload: ChunkPageRequest) -> ChunkPageResponse:
    """Generate structured Chunks for a single page with provenance."""
    chunker = Chunker()
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
