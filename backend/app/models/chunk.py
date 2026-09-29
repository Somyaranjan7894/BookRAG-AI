"""Chunk ORM model representing text chunks for retrieval and question answering."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.page import Page


class Chunk(Base):
    """SQLAlchemy model representing a text chunk derived from an extracted page."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        index=True,
        doc="Deterministic chunk identifier preserving Phase 2 identity.",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key to the parent Document.",
    )
    page_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("pages.page_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key to the parent Page.",
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="1-indexed source book page number.",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="0-indexed sequence position of chunk within the page or document.",
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Cleaned chunk text content.",
    )
    char_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Character length of chunk text.",
    )
    word_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Whitespace-delimited word count of chunk text.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Timestamp when the chunk was persisted.",
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )
    page: Mapped["Page"] = relationship(
        "Page",
        back_populates="chunks",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "page_number",
            "chunk_index",
            name="uq_chunks_document_page_chunk",
        ),
        CheckConstraint(
            "page_number >= 1",
            name="ck_chunks_page_number_positive",
        ),
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_chunks_chunk_index_non_negative",
        ),
        Index(
            "ix_chunks_document_page",
            "document_id",
            "page_number",
        ),
        Index(
            "ix_chunks_doc_page_idx",
            "document_id",
            "page_number",
            "chunk_index",
        ),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id='{self.chunk_id}', doc='{self.document_id}', p={self.page_number}, idx={self.chunk_index})>"
