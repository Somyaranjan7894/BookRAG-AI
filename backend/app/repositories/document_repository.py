"""Repository for Document persistence operations."""

from typing import List, Optional, Union

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus


class DocumentRepository:
    """Encapsulates persistent database operations for Document entities."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, document: Document) -> Document:
        """Persist a new Document entity."""
        self.session.add(document)
        self.session.flush()
        return document

    def get_by_id(self, document_id: str) -> Optional[Document]:
        """Retrieve a Document by its primary identifier."""
        stmt = select(Document).where(Document.document_id == document_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def update_status(
        self,
        document_id: str,
        status: Union[DocumentStatus, str],
        stage: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Document]:
        """Update the lifecycle status of an existing Document, with optional stage and error message."""
        doc = self.get_by_id(document_id)
        if doc is None:
            return None
        doc.status = status.value if isinstance(status, DocumentStatus) else str(status)
        if stage is not None:
            doc.processing_stage = stage
        if error_message is not None:
            doc.error_message = error_message
        self.session.flush()
        return doc

    def list_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """Retrieve an ordered page of persisted documents."""
        stmt = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def delete(self, document_id: str) -> bool:
        """Delete a Document and cascade deletion to associated pages and chunks."""
        doc = self.get_by_id(document_id)
        if doc is None:
            return False
        stmt = delete(Document).where(Document.document_id == document_id)
        self.session.execute(stmt)
        self.session.flush()
        return True

    def count(self) -> int:
        """Return total count of persisted documents."""
        stmt = select(func.count()).select_from(Document)
        return self.session.execute(stmt).scalar_one() or 0
