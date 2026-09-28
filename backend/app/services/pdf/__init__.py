"""PDF service package for document ingestion and page-aware representation."""

from app.services.pdf.exceptions import (
    InvalidPDFError,
    PDFExtractionError,
    PDFIngestionError,
    PDFNotFoundError,
)
from app.services.pdf.ingestion import PDFIngestionService
from app.services.pdf.parser import PDFParser

__all__ = [
    "InvalidPDFError",
    "PDFExtractionError",
    "PDFIngestionError",
    "PDFNotFoundError",
    "PDFIngestionService",
    "PDFParser",
]
