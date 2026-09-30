"""Transactional persistence service coordinating Document, Page, and Chunk storage."""

from typing import List, Optional, Sequence

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.page import Page
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.page_repository import PageRepository
from app.schemas.chunk import Chunk as DomainChunk
from app.schemas.document import Document as DomainDocument
from app.services.persistence.exceptions import (
    ConstraintViolationError,
    DocumentAlreadyExistsError,
    PersistenceError,
    ReferentialIntegrityError,
)

logger = get_logger(__name__)


class DocumentPersistenceService:
    """Orchestrates transactional persistence of book documents, pages, and chunks."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.doc_repo = DocumentRepository(session)
        self.page_repo = PageRepository(session)
        self.chunk_repo = ChunkRepository(session)

    def persist_document_and_chunks(
        self,
        document: DomainDocument,
        chunks: Sequence[DomainChunk],
    ) -> Document:
        """Persist a complete document, its pages, and its generated chunks in a single atomic transaction.

        Guarantees:
        - If any step fails, the entire transaction is rolled back.
        - Document is marked PROCESSING during ingestion, then PROCESSED upon commit.
        - No partial state is retained if chunk or page persistence fails.
        """
        logger.info(
            "Persisting document '%s' with %d pages and %d chunks",
            document.document_id,
            document.page_count,
            len(chunks),
        )

        try:
            # Check for existing document
            existing = self.doc_repo.get_by_id(document.document_id)
            if existing is not None:
                if existing.status == DocumentStatus.PROCESSED.value:
                    raise DocumentAlreadyExistsError(
                        f"Document with ID '{document.document_id}' already exists in persistence.",
                        details={"document_id": document.document_id},
                    )
                # Reuse and update existing record (from UPLOADED / QUEUED / FAILED)
                doc_record = existing
                doc_record.filename = document.filename
                doc_record.title = getattr(document, "title", None) or getattr(document.metadata, "title", None)
                doc_record.author = getattr(document, "author", None) or getattr(document.metadata, "author", None)
                doc_record.page_count = document.page_count
                doc_record.status = DocumentStatus.PROCESSING.value
                # Clean up any partial pages/chunks from a previous failed attempt
                self.session.query(Chunk).filter(Chunk.document_id == document.document_id).delete()
                self.session.query(Page).filter(Page.document_id == document.document_id).delete()
                self.session.flush()
            else:
                # 1. Create Document record in PROCESSING status
                doc_record = Document(
                    document_id=document.document_id,
                    filename=document.filename,
                    title=getattr(document, "title", None) or getattr(document.metadata, "title", None),
                    author=getattr(document, "author", None) or getattr(document.metadata, "author", None),
                    page_count=document.page_count,
                    status=DocumentStatus.PROCESSING.value,
                )
                self.doc_repo.create(doc_record)

            # 2. Create Page records
            page_records: List[Page] = []
            for p in document.pages:
                page_id = f"{document.document_id}_p{p.page_number}"
                page_records.append(
                    Page(
                        page_id=page_id,
                        document_id=document.document_id,
                        page_number=p.page_number,
                        text=p.text,
                        char_count=p.char_count,
                        word_count=p.word_count,
                    )
                )
            self.page_repo.create_many(page_records)

            # 3. Create Chunk records
            chunk_records: List[Chunk] = []
            for c in chunks:
                page_id = f"{c.document_id}_p{c.page_number}"
                chunk_records.append(
                    Chunk(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        page_id=page_id,
                        page_number=c.page_number,
                        chunk_index=c.chunk_index,
                        text=c.text,
                        char_count=c.char_count,
                        word_count=c.word_count,
                    )
                )
            self.chunk_repo.create_many(chunk_records)

            # 4. Transition Document to PROCESSED
            doc_record.status = DocumentStatus.PROCESSED.value

            # 5. Commit atomic transaction
            self.session.commit()
            logger.info("Successfully committed persistence for document '%s'", document.document_id)
            return doc_record

        except IntegrityError as exc:
            self.session.rollback()
            msg = str(exc.orig) if hasattr(exc, "orig") else str(exc)
            logger.error("Integrity error during document persistence '%s': %s", document.document_id, msg)
            if "foreign key" in msg.lower():
                raise ReferentialIntegrityError(
                    f"Referential integrity violated during persistence: {msg}",
                    details={"document_id": document.document_id, "error": msg},
                ) from exc
            raise ConstraintViolationError(
                f"Constraint violated during persistence: {msg}",
                details={"document_id": document.document_id, "error": msg},
            ) from exc

        except DocumentAlreadyExistsError:
            self.session.rollback()
            raise

        except SQLAlchemyError as exc:
            self.session.rollback()
            logger.error("SQLAlchemy error during document persistence '%s': %s", document.document_id, exc)
            raise PersistenceError(
                f"Database error during document persistence: {exc}",
                details={"document_id": document.document_id},
            ) from exc

        except Exception as exc:
            self.session.rollback()
            logger.error("Unexpected error during document persistence '%s': %s", document.document_id, exc)
            raise PersistenceError(
                f"Unexpected error during document persistence: {exc}",
                details={"document_id": document.document_id},
            ) from exc

    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a Document with pages and chunks eagerly loaded or queried."""
        return self.doc_repo.get_by_id(document_id)

    def get_document_pages(self, document_id: str) -> List[Page]:
        """Retrieve all pages belonging to a document."""
        return self.page_repo.get_by_document(document_id)

    def get_document_chunks(self, document_id: str) -> List[Chunk]:
        """Retrieve all chunks belonging to a document."""
        return self.chunk_repo.get_by_document(document_id)

    def list_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """List persisted documents with pagination."""
        return self.doc_repo.list_documents(skip=skip, limit=limit)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and cascade delete its pages and chunks."""
        try:
            result = self.doc_repo.delete(document_id)
            self.session.commit()
            return result
        except SQLAlchemyError as exc:
            self.session.rollback()
            logger.error("Failed to delete document '%s': %s", document_id, exc)
            raise PersistenceError(f"Failed to delete document: {exc}") from exc
