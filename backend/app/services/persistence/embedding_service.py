"""Embedding persistence service connecting EmbeddingService to VectorRepository."""

from typing import Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.chunk import Chunk as DBChunk
from app.repositories.vector_repository import BaseVectorRepository, PGVectorRepository
from app.schemas.chunk import Chunk as DomainChunk
from app.schemas.embedding import EmbeddingConfig, EmbeddingRecord
from app.services.embeddings.service import EmbeddingService

logger = get_logger(__name__)


class EmbeddingPersistenceService:
    """Orchestrates generation of dense vector embeddings and their persistence in PostgreSQL."""

    def __init__(
        self,
        session: Session,
        embedding_service: Optional[EmbeddingService] = None,
        vector_repo: Optional[BaseVectorRepository] = None,
    ) -> None:
        """Initialize EmbeddingPersistenceService with injected dependencies."""
        self.session = session
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_repo = vector_repo or PGVectorRepository(session)

    def embed_and_persist_chunks(
        self,
        chunks: Sequence[DomainChunk],
        batch_size: Optional[int] = None,
    ) -> int:
        """Generate embeddings in batches using EmbeddingService and persist them via VectorRepository.

        Args:
            chunks: Sequence of DomainChunk items needing embedding.
            batch_size: Batch size for model inference (defaults to settings.EMBEDDING_BATCH_SIZE).

        Returns:
            Total count of chunk embeddings persisted.
        """
        if not chunks:
            return 0

        effective_batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        logger.info("Generating and persisting embeddings for %d chunks (batch_size=%d)", len(chunks), effective_batch_size)

        # 1. Generate embeddings in batches via Phase 3 EmbeddingService
        cfg = EmbeddingConfig(batch_size=effective_batch_size)
        records: List[EmbeddingRecord] = self.embedding_service.embed_chunks(
            chunks=list(chunks),
            config=cfg,
        )

        # 2. Map chunk_id to embedding vector
        embeddings_map: Dict[str, Sequence[float]] = {
            rec.chunk_id: rec.embedding
            for rec in records
        }

        # 3. Persist via VectorRepository
        updated_count = self.vector_repo.store_embeddings(embeddings_map)
        self.session.commit()

        logger.info("Successfully persisted %d embeddings to pgvector", updated_count)
        return updated_count

    def embed_unembedded_document_chunks(
        self,
        document_id: str,
        batch_size: Optional[int] = None,
    ) -> int:
        """Find chunks for a given document that have no embeddings, embed and persist them.

        Args:
            document_id: Target document identifier.
            batch_size: Inference batch size.

        Returns:
            Number of newly embedded chunks.
        """
        stmt = (
            select(DBChunk)
            .where(
                DBChunk.document_id == document_id,
                DBChunk.embedding.isnot(None) == False,
            )
            .order_by(DBChunk.page_number.asc(), DBChunk.chunk_index.asc())
        )
        db_chunks = list(self.session.execute(stmt).scalars().all())

        if not db_chunks:
            logger.debug("No unembedded chunks found for document '%s'", document_id)
            return 0

        domain_chunks = [
            DomainChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
                char_count=c.char_count,
                word_count=c.word_count,
            )
            for c in db_chunks
        ]

        return self.embed_and_persist_chunks(domain_chunks, batch_size=batch_size)

    def embed_all_unembedded_chunks(
        self,
        limit: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> int:
        """Find all database chunks lacking embeddings across all documents and persist them.

        Args:
            limit: Maximum chunks to process in this run.
            batch_size: Inference batch size.

        Returns:
            Number of newly embedded chunks.
        """
        stmt = (
            select(DBChunk)
            .where(DBChunk.embedding.isnot(None) == False)
            .order_by(DBChunk.created_at.asc())
        )
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)

        db_chunks = list(self.session.execute(stmt).scalars().all())
        if not db_chunks:
            return 0

        domain_chunks = [
            DomainChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
                char_count=c.char_count,
                word_count=c.word_count,
            )
            for c in db_chunks
        ]

        return self.embed_and_persist_chunks(domain_chunks, batch_size=batch_size)
