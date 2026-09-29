"""Document ORM model representing ingested book source materials."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.page import Page


class DocumentStatus(str, Enum):
    """Enumeration of persistent document lifecycle states."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Document(Base):
    """SQLAlchemy model representing a persistent book document."""

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        index=True,
        doc="Deterministic document identifier generated from file content.",
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Original uploaded filename.",
    )
    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Optional book or document title metadata.",
    )
    author: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Optional author metadata.",
    )
    page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total number of extracted pages.",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentStatus.UPLOADED.value,
        index=True,
        doc="Current document lifecycle processing state.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Timestamp when the document was initially registered.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Timestamp of the most recent document update.",
    )

    # Relationships
    pages: Mapped[List["Page"]] = relationship(
        "Page",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Page.page_number",
        passive_deletes=True,
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Document(id='{self.document_id}', filename='{self.filename}', status='{self.status}')>"
