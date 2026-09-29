"""Repository for Chunk persistence operations."""

from typing import List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk


class ChunkRepository:
    """Encapsulates persistent database operations for Chunk entities."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, chunk: Chunk) -> Chunk:
        """Persist a single Chunk entity."""
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def create_many(self, chunks: Sequence[Chunk]) -> List[Chunk]:
        """Persist multiple Chunk entities in batch."""
        chunk_list = list(chunks)
        if not chunk_list:
            return []
        self.session.add_all(chunk_list)
        self.session.flush()
        return chunk_list

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a Chunk by its primary identifier."""
        stmt = select(Chunk).where(Chunk.chunk_id == chunk_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_document(self, document_id: str) -> List[Chunk]:
        """Retrieve all chunks for a document ordered by page number and chunk index."""
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.page_number.asc(), Chunk.chunk_index.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_page(self, document_id: str, page_number: int) -> List[Chunk]:
        """Retrieve all chunks for a specific document page ordered by chunk index."""
        stmt = (
            select(Chunk)
            .where(
                Chunk.document_id == document_id,
                Chunk.page_number == page_number,
            )
            .order_by(Chunk.chunk_index.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def count_by_document(self, document_id: str) -> int:
        """Count total chunks associated with a specific document."""
        stmt = (
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == document_id)
        )
        return self.session.execute(stmt).scalar_one() or 0
