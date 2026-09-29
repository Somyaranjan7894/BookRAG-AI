"""Development, ingestion, and persistent document management endpoints."""

from collections.abc import Generator
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.schemas.document import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentListResponse,
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
