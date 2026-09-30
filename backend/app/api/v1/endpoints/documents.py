"""Development, ingestion, and persistent document management endpoints."""

from collections.abc import Generator
import hashlib
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.models.document import Document, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    PersistedChunkResponse,
    PersistedDocumentResponse,
    PersistedPageResponse,
)
from app.services.pdf.ingestion import PDFIngestionService
from app.services.persistence import DocumentAlreadyExistsError, DocumentPersistenceService
from app.services.text.processor import TextProcessingService

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_optional_db() -> Generator[Optional[Session], None, None]:
    """Provide a database session if database infrastructure is reachable."""
    try:
        session = get_session_factory()()
    except Exception as exc:
        logger.debug("Database not reachable for optional persistence: %s", exc)
        yield None
        return

    try:
        yield session
    finally:
        session.close()


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload & Enqueue Document for Processing",
    description=(
        "Accepts a PDF upload or JSON file reference, validates presence, registers "
        "the document in status 'queued', enqueues the Celery background task, and returns 202 Accepted."
    ),
)
async def upload_document(
    request: Request,
    db: Optional[Session] = Depends(get_optional_db),
) -> DocumentUploadResponse:
    """Asynchronously enqueue a PDF document for background processing."""
    content_type = request.headers.get("content-type", "")
    os.makedirs(settings.UPLOAD_STORAGE_DIR, exist_ok=True)

    dest_path: str
    filename: str
    custom_doc_id: Optional[str] = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file or not hasattr(uploaded_file, "filename"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Form data missing required 'file' upload",
            )
        filename = uploaded_file.filename or "upload.pdf"
        custom_doc_id = form.get("document_id")
        file_bytes = await uploaded_file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )
        dest_filename = f"{hashlib.sha256(file_bytes).hexdigest()[:12]}_{filename}"
        dest_path = os.path.join(settings.UPLOAD_STORAGE_DIR, dest_filename)
        with open(dest_path, "wb") as f:
            f.write(file_bytes)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request payload; expected multipart form-data or JSON with 'file_path'",
            )
        raw_file_path = body.get("file_path")
        if not raw_file_path or not isinstance(raw_file_path, str) or not raw_file_path.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON payload missing required 'file_path'",
            )
        source_path = Path(raw_file_path.strip())
        if not source_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source file not found at '{raw_file_path}'",
            )
        filename = source_path.name
        dest_path = str(source_path.resolve())
        custom_doc_id = body.get("document_id")

    # Determine stable deterministic document ID if not provided
    if custom_doc_id and custom_doc_id.strip():
        document_id = custom_doc_id.strip()
    else:
        with open(dest_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        base_name = Path(filename).stem
        safe_stem = "".join(c if c.isalnum() else "_" for c in base_name)[:20].strip("_")
        document_id = f"doc_{safe_stem}_{file_hash}"

    # Register initial durable state in PostgreSQL
    if db is not None:
        doc_repo = DocumentRepository(db)
        doc = doc_repo.get_by_id(document_id)
        if doc is None:
            doc = Document(
                document_id=document_id,
                filename=filename,
                status=DocumentStatus.QUEUED.value,
                processing_stage="queued",
            )
            doc_repo.create(doc)
        else:
            doc_repo.update_status(
                document_id,
                status=DocumentStatus.QUEUED,
                stage="queued",
                error_message=None,
            )
        db.commit()

    # Enqueue Celery task
    task_id: Optional[str] = None
    try:
        from app.workers.tasks import process_document_task

        task = process_document_task.delay(document_id=document_id, file_path=dest_path)
        task_id = task.id
        logger.info("Enqueued process_document_task for document '%s' with task_id=%s", document_id, task_id)
    except Exception as exc:
        logger.exception("Failed to enqueue Celery task for document '%s': %s", document_id, exc)
        if db is not None:
            doc_repo = DocumentRepository(db)
            doc_repo.update_status(
                document_id,
                status=DocumentStatus.FAILED,
                stage="queue_failed",
                error_message=str(exc),
            )
            db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue background processing task: {exc}",
        )

    return DocumentUploadResponse(
        document_id=document_id,
        task_id=task_id,
        status=DocumentStatus.QUEUED.value,
        message="Document processing task enqueued successfully.",
    )


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest PDF Document",
    description=(
        "Ingests a PDF from a local filesystem path, extracts page text with complete provenance, "
        "and transactionally persists Document, Pages, and Chunks to PostgreSQL if available."
    ),
)
async def ingest_document(
    payload: DocumentIngestRequest,
    persist: bool = Query(default=True, description="Persist document, pages, and chunks to PostgreSQL"),
    db: Optional[Session] = Depends(get_optional_db),
) -> DocumentIngestResponse:
    """Ingest a PDF from a local filesystem path and return structured document representation."""
    service = PDFIngestionService()
    document = service.ingest_pdf(
        file_path=payload.file_path,
        document_id=payload.document_id,
    )

    # Transactional persistence into PostgreSQL
    if persist and db is not None:
        try:
            processor = TextProcessingService()
            chunks = processor.process_document(document)
            persistence = DocumentPersistenceService(db)
            persistence.persist_document_and_chunks(document, chunks)
            logger.info("Persisted document '%s' with %d chunks to PostgreSQL", document.document_id, len(chunks))
        except DocumentAlreadyExistsError:
            logger.info("Document '%s' already exists in PostgreSQL persistence.", document.document_id)
        except Exception as exc:
            logger.warning("Could not persist document '%s' to database: %s", document.document_id, exc)

    return DocumentIngestResponse(status="success", document=document)


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Persisted Documents",
    description="Retrieve paginated list of documents persisted in PostgreSQL.",
)
async def list_persisted_documents(
    skip: int = Query(default=0, ge=0, description="Offset for pagination"),
    limit: int = Query(default=100, ge=1, le=500, description="Page limit"),
    db: Optional[Session] = Depends(get_optional_db),
) -> DocumentListResponse:
    """Retrieve list of documents from PostgreSQL."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable",
        )
    persistence = DocumentPersistenceService(db)
    docs = persistence.list_documents(skip=skip, limit=limit)
    total = persistence.doc_repo.count()
    return DocumentListResponse(
        total=total,
        documents=[
            PersistedDocumentResponse(
                document_id=d.document_id,
                filename=d.filename,
                title=d.title,
                author=d.author,
                page_count=d.page_count,
                status=d.status,
                processing_stage=d.processing_stage,
                error_message=d.error_message,
                created_at=str(d.created_at) if d.created_at else None,
                updated_at=str(d.updated_at) if d.updated_at else None,
            )
            for d in docs
        ],
    )


@router.get(
    "/{document_id}",
    response_model=PersistedDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Persisted Document",
    description="Retrieve an individual document by identifier from PostgreSQL.",
)
async def get_persisted_document(
    document_id: str,
    db: Optional[Session] = Depends(get_optional_db),
) -> PersistedDocumentResponse:
    """Retrieve document by ID."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable",
        )
    persistence = DocumentPersistenceService(db)
    doc = persistence.get_document(document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in database",
        )
    return PersistedDocumentResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        title=doc.title,
        author=doc.author,
        page_count=doc.page_count,
        status=doc.status,
        processing_stage=doc.processing_stage,
        error_message=doc.error_message,
        created_at=str(doc.created_at) if doc.created_at else None,
        updated_at=str(doc.updated_at) if doc.updated_at else None,
    )


@router.get(
    "/{document_id}/pages",
    response_model=List[PersistedPageResponse],
    status_code=status.HTTP_200_OK,
    summary="Get Document Pages",
    description="Retrieve all persisted pages for a document from PostgreSQL.",
)
async def get_persisted_pages(
    document_id: str,
    db: Optional[Session] = Depends(get_optional_db),
) -> List[PersistedPageResponse]:
    """Retrieve pages belonging to a document."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable",
        )
    persistence = DocumentPersistenceService(db)
    pages = persistence.get_document_pages(document_id)
    return [
        PersistedPageResponse(
            page_id=p.page_id,
            document_id=p.document_id,
            page_number=p.page_number,
            text=p.text,
            char_count=p.char_count,
            word_count=p.word_count,
        )
        for p in pages
    ]


@router.get(
    "/{document_id}/chunks",
    response_model=List[PersistedChunkResponse],
    status_code=status.HTTP_200_OK,
    summary="Get Document Chunks",
    description="Retrieve all persisted text chunks for a document from PostgreSQL.",
)
async def get_persisted_chunks(
    document_id: str,
    db: Optional[Session] = Depends(get_optional_db),
) -> List[PersistedChunkResponse]:
    """Retrieve chunks belonging to a document."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable",
        )
    persistence = DocumentPersistenceService(db)
    chunks = persistence.get_document_chunks(document_id)
    return [
        PersistedChunkResponse(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            page_id=c.page_id,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            text=c.text,
            char_count=c.char_count,
            word_count=c.word_count,
        )
        for c in chunks
    ]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Persisted Document",
    description="Delete a document and cascade deletion to associated pages and chunks in PostgreSQL.",
)
async def delete_persisted_document(
    document_id: str,
    db: Optional[Session] = Depends(get_optional_db),
) -> dict:
    """Delete a document and its cascaded relational data."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable",
        )
    persistence = DocumentPersistenceService(db)
    deleted = persistence.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found for deletion",
        )
    return {"status": "deleted", "document_id": document_id}
