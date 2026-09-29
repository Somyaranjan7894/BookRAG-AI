"""Page ORM model representing individual extracted book pages."""

from datetime import datetime
from typing import TYPE_CHECKING, List

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
    from app.models.chunk import Chunk
    from app.models.document import Document


class Page(Base):
    """SQLAlchemy model representing an extracted page within a document."""

    __tablename__ = "pages"

    page_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        index=True,
        doc="Unique identifier for the page (e.g. {document_id}_p{page_number}).",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key to the parent Document.",
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="1-indexed sequential physical page number in the source book.",
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Exact raw text content extracted from this page.",
    )
    char_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Total character count of the page text.",
    )
    word_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Total whitespace-delimited word count of the page text.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Timestamp when the page was persisted.",
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="pages",
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="page",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "page_number",
            name="uq_pages_document_page_number",
        ),
        CheckConstraint(
            "page_number >= 1",
            name="ck_pages_page_number_positive",
        ),
        Index(
            "ix_pages_document_id_page_number",
            "document_id",
            "page_number",
        ),
    )

    def __repr__(self) -> str:
        return f"<Page(id='{self.page_id}', doc='{self.document_id}', page={self.page_number})>"
