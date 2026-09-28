"""Development and testing endpoints for semantic vector embedding generation."""

from fastapi import APIRouter, status

from app.schemas.chunk import Chunk
from app.schemas.embedding import (
    EmbedChunkRequest,
    EmbedChunksRequest,
    EmbedChunksResponse,
    EmbeddingRecord,
)
from app.services.embeddings.service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.post(
    "/embed-chunk",
    response_model=EmbeddingRecord,
    status_code=status.HTTP_200_OK,
    summary="Embed Single Chunk",
    description="Development endpoint to generate a dense semantic vector for a single text chunk.",
)
async def embed_single_chunk(payload: EmbedChunkRequest) -> EmbeddingRecord:
    """Generate an embedding for a single chunk with provenance retention."""
    chunk = Chunk(
        chunk_id=payload.chunk_id,
        document_id=payload.document_id,
        page_number=payload.page_number,
        chunk_index=0,
        text=payload.text,
        char_count=len(payload.text),
        word_count=len(payload.text.split()),
    )
    service = EmbeddingService()
    return service.embed_chunk(chunk, config=payload.config)


@router.post(
    "/embed-chunks",
    response_model=EmbedChunksResponse,
    status_code=status.HTTP_200_OK,
    summary="Embed Batch of Chunks",
    description="Development endpoint to batch-embed multiple text chunks.",
)
async def embed_batch_chunks(payload: EmbedChunksRequest) -> EmbedChunksResponse:
    """Generate embeddings for a batch of serialized chunks."""
    chunks = [
        Chunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            page_number=c["page_number"],
            chunk_index=c.get("chunk_index", idx),
            text=c["text"],
            char_count=len(c["text"]),
            word_count=len(c["text"].split()),
        )
        for idx, c in enumerate(payload.chunks)
    ]
    service = EmbeddingService()
    records = service.embed_chunks(chunks, config=payload.config)

    dimension = records[0].dimension if records else 384
    model_name = records[0].model_name if records else "sentence-transformers/all-MiniLM-L6-v2"
    device = records[0].device if records else "cpu"

    return EmbedChunksResponse(
        total_embedded=len(records),
        dimension=dimension,
        model_name=model_name,
        device=device,
        records=records,
    )
