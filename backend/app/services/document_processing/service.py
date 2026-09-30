"""Application-level document processing service orchestrating ingestion, chunking, persistence, and indexing."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.document import Document, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.services.document_processing.exceptions import (
    DocumentProcessingError,
    PermanentProcessingError,
    TransientProcessingError,
)
from app.services.embeddings.service import EmbeddingService
from app.services.pdf.exceptions import (
    InvalidPDFError,
    PDFExtractionError,
    PDFNotFoundError,
)
from app.services.pdf.ingestion import PDFIngestionService
from app.services.persistence import DocumentPersistenceService
from app.services.persistence.embedding_service import EmbeddingPersistenceService
from app.services.retrieval.service import RetrievalService
from app.services.text.processor import TextProcessingService

logger = get_logger(__name__)


class DocumentProcessingService:
    """Orchestrates end-to-end background document processing.

    Stages:
    1. File Validation
    2. Document Registration / State Update (PROCESSING)
    3. PDF Ingestion (PyMuPDF)
    4. Text Cleaning & Chunking
    5. PostgreSQL Relational Persistence (Documents, Pages, Chunks)
    6. Vector Indexing (pgvector or FAISS based on settings.VECTOR_BACKEND)
    7. Final Status Update (PROCESSED)
    """

    def __init__(
        self,
        session: Session,
        pdf_service: Optional[PDFIngestionService] = None,
        text_service: Optional[TextProcessingService] = None,
        persistence_service: Optional[DocumentPersistenceService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        embedding_persistence: Optional[EmbeddingPersistenceService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        vector_backend: Optional[str] = None,
    ) -> None:
        self.session = session
        self.doc_repo = DocumentRepository(session)
        self.pdf_service = pdf_service or PDFIngestionService()
        self.text_service = text_service or TextProcessingService()
        self.persistence_service = persistence_service or DocumentPersistenceService(session)
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_backend = (vector_backend or settings.VECTOR_BACKEND).lower()

        if self.vector_backend == "pgvector":
            self.embedding_persistence = embedding_persistence or EmbeddingPersistenceService(
                session=session,
                embedding_service=self.embedding_service,
            )
            self.retrieval_service = None
        else:
            self.embedding_persistence = None
            self.retrieval_service = retrieval_service or RetrievalService(
                embedding_service=self.embedding_service
            )

    def process(self, document_id: str, file_path: str) -> Dict[str, Any]:
        """Execute the multi-stage document processing pipeline.

        Args:
            document_id: Unique deterministic document identifier.
            file_path: Stable filesystem path to source PDF file.

        Returns:
            Dict containing processing summary: document_id, status, page_count, chunk_count.

        Raises:
            TransientProcessingError: For retryable operational errors.
            PermanentProcessingError: For deterministic unretryable failures.
        """
        logger.info(
            "Starting document processing: document_id=%s, file_path=%s, backend=%s",
            document_id,
            file_path,
            self.vector_backend,
        )

        # 1. File validation
        path = Path(file_path)
        if not path.is_file():
            err_msg = f"Source PDF file does not exist at path: '{file_path}'"
            logger.error("Processing failed for '%s': %s", document_id, err_msg)
            self._mark_failed(document_id, stage="validation", error_message=err_msg)
            raise PermanentProcessingError(err_msg)

        if path.suffix.lower() != ".pdf":
            err_msg = f"Invalid file extension for '{file_path}'; expected '.pdf'"
            logger.error("Processing failed for '%s': %s", document_id, err_msg)
            self._mark_failed(document_id, stage="validation", error_message=err_msg)
            raise PermanentProcessingError(err_msg)

        try:
            # 2. Stage: Document Registration / Idempotency Check
            doc = self.doc_repo.get_by_id(document_id)
            if doc is None:
                doc = Document(
                    document_id=document_id,
                    filename=path.name,
                    status=DocumentStatus.PROCESSING.value,
                    processing_stage="ingestion",
                )
                self.doc_repo.create(doc)
                self.session.commit()
            elif doc.status == DocumentStatus.PROCESSED.value:
                # Idempotency check: verify if document already has pages and chunks
                chunk_count = len(doc.chunks)
                if chunk_count > 0:
                    logger.info(
                        "Document '%s' is already processed (%d chunks). Skipping duplicate execution.",
                        document_id,
                        chunk_count,
                    )
                    return {
                        "document_id": document_id,
                        "status": DocumentStatus.PROCESSED.value,
                        "page_count": doc.page_count,
                        "chunk_count": chunk_count,
                        "idempotent": True,
                    }
                else:
                    self.doc_repo.update_status(
                        document_id=document_id,
                        status=DocumentStatus.PROCESSING,
                        stage="ingestion",
                        error_message=None,
                    )
                    self.session.commit()
            else:
                self.doc_repo.update_status(
                    document_id=document_id,
                    status=DocumentStatus.PROCESSING,
                    stage="ingestion",
                    error_message=None,
                )
                self.session.commit()

            # 3. Stage: PDF Ingestion
            try:
                pdf_document = self.pdf_service.ingest_pdf(
                    file_path=str(path),
                    document_id=document_id,
                )
            except (PDFNotFoundError, InvalidPDFError, PDFExtractionError) as exc:
                err_msg = f"PDF ingestion error: {exc.message}"
                logger.error("PDF ingestion failed for '%s': %s", document_id, err_msg)
                self._mark_failed(document_id, stage="ingestion", error_message=err_msg)
                raise PermanentProcessingError(err_msg, details=str(exc)) from exc

            # 4. Stage: Text Cleaning & Chunking
            self.doc_repo.update_status(document_id, DocumentStatus.PROCESSING, stage="chunking")
            self.session.commit()

            chunks = self.text_service.process_document(pdf_document)
            logger.info(
                "Generated %d chunks across %d pages for document '%s'",
                len(chunks),
                pdf_document.page_count,
                document_id,
            )

            # 5. Stage: Relational Persistence (Documents, Pages, Chunks)
            self.doc_repo.update_status(document_id, DocumentStatus.PROCESSING, stage="persistence")
            self.session.commit()

            self.persistence_service.persist_document_and_chunks(pdf_document, chunks)
            self.session.commit()

            # 6. Stage: Vector Generation & Indexing
            if self.vector_backend == "pgvector":
                self.doc_repo.update_status(document_id, DocumentStatus.PROCESSING, stage="embedding")
                self.session.commit()

                assert self.embedding_persistence is not None
                persisted_vectors = self.embedding_persistence.embed_and_persist_chunks(chunks)
                logger.info(
                    "Persisted %d vector embeddings to pgvector for document '%s'",
                    persisted_vectors,
                    document_id,
                )

                self.doc_repo.update_status(document_id, DocumentStatus.PROCESSING, stage="indexing")
                self.session.commit()
            else:
                # FAISS local vector backend
                self.doc_repo.update_status(document_id, DocumentStatus.PROCESSING, stage="embedding")
                self.session.commit()

                assert self.retrieval_service is not None
                self.retrieval_service.build_index_from_chunks(
                    chunks=chunks,
                    index_id=document_id,
                )
                self.retrieval_service.save_index(
                    index_id=document_id,
                    directory=settings.INDEX_STORAGE_DIR,
                )
                logger.info("Built and persisted FAISS index for document '%s'", document_id)

                self.doc_repo.update_status(document_id, DocumentStatus.PROCESSING, stage="indexing")
                self.session.commit()

            # 7. Stage: Completion
            self.doc_repo.update_status(
                document_id=document_id,
                status=DocumentStatus.PROCESSED,
                stage="completed",
                error_message=None,
            )
            self.session.commit()

            logger.info("Successfully completed document processing for '%s'", document_id)
            return {
                "document_id": document_id,
                "status": DocumentStatus.PROCESSED.value,
                "page_count": pdf_document.page_count,
                "chunk_count": len(chunks),
                "backend": self.vector_backend,
            }

        except (PermanentProcessingError, TransientProcessingError):
            raise
        except OperationalError as exc:
            self.session.rollback()
            err_msg = f"Database connectivity error: {exc}"
            logger.exception("Transient database failure processing document '%s': %s", document_id, exc)
            self._mark_failed(document_id, stage="database", error_message="Database connectivity failure")
            raise TransientProcessingError(err_msg, details=str(exc)) from exc
        except Exception as exc:
            self.session.rollback()
            err_msg = f"Unexpected processing error: {exc}"
            logger.exception("Unexpected error processing document '%s': %s", document_id, exc)
            self._mark_failed(document_id, stage="failed", error_message=str(exc)[:500])
            raise PermanentProcessingError(err_msg, details=str(exc)) from exc

    def _mark_failed(self, document_id: str, stage: str, error_message: str) -> None:
        """Helper to safely record failure status in PostgreSQL."""
        try:
            self.session.rollback()
            doc = self.doc_repo.get_by_id(document_id)
            if doc is not None:
                doc.status = DocumentStatus.FAILED.value
                doc.processing_stage = stage
                doc.error_message = error_message[:500]
                self.session.flush()
                self.session.commit()
        except Exception as exc:
            logger.warning("Failed to record failure status for '%s': %s", document_id, exc)
