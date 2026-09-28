"""Minimal development and testing endpoint for PDF document ingestion."""

from fastapi import APIRouter, status

from app.schemas.document import DocumentIngestRequest, DocumentIngestResponse
from app.services.pdf.ingestion import PDFIngestionService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest PDF Document",
    description=(
        "Development endpoint to trigger synchronous PDF ingestion from a filesystem path. "
        "Extracts raw page text, preserves page numbers, and calculates page statistics."
    ),
)
async def ingest_document(payload: DocumentIngestRequest) -> DocumentIngestResponse:
    """Ingest a PDF from a local filesystem path and return structured document representation."""
    service = PDFIngestionService()
    document = service.ingest_pdf(
        file_path=payload.file_path,
        document_id=payload.document_id,
    )
    return DocumentIngestResponse(status="success", document=document)
