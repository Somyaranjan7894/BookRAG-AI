"""Low-level PDF parser leveraging PyMuPDF for document inspection and text extraction."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator
import pymupdf

from app.core.logging import get_logger
from app.schemas.document import DocumentMetadata
from app.services.pdf.exceptions import InvalidPDFError, PDFExtractionError

logger = get_logger(__name__)


class PDFParser:
    """Encapsulates PyMuPDF operations for safe PDF opening and extraction."""

    @contextmanager
    def open_document(self, file_path: Path) -> Generator[pymupdf.Document, None, None]:
        """Safely open a PDF file and guarantee closure of file descriptors.

        Args:
            file_path: Validated Path to the PDF file.

        Yields:
            pymupdf.Document instance.

        Raises:
            InvalidPDFError: If PyMuPDF fails to parse or decode the PDF structure.
            PDFExtractionError: For unexpected file access or parsing errors.
        """
        doc = None
        try:
            doc = pymupdf.open(str(file_path))
            if not doc.is_pdf:
                raise InvalidPDFError(
                    f"File '{file_path.name}' is not recognized as a valid PDF document."
                )
            yield doc
        except pymupdf.FileDataError as exc:
            logger.error("PyMuPDF FileDataError opening %s: %s", file_path.name, exc)
            raise InvalidPDFError(
                f"File '{file_path.name}' is corrupt or not a readable PDF.",
                details=str(exc),
            ) from exc
        except InvalidPDFError:
            raise
        except Exception as exc:
            logger.error("Unexpected error opening PDF %s: %s", file_path.name, exc)
            raise PDFExtractionError(
                f"Failed to open PDF '{file_path.name}'.",
                details=str(exc),
            ) from exc
        finally:
            if doc is not None:
                doc.close()

    def extract_metadata(self, doc: pymupdf.Document) -> DocumentMetadata:
        """Extract standard PDF metadata fields."""
        raw_meta = doc.metadata or {}
        return DocumentMetadata(
            title=raw_meta.get("title") or None,
            author=raw_meta.get("author") or None,
            subject=raw_meta.get("subject") or None,
            creator=raw_meta.get("creator") or None,
            producer=raw_meta.get("producer") or None,
            creation_date=raw_meta.get("creationDate") or None,
            mod_date=raw_meta.get("modDate") or None,
        )

    def extract_page_text(self, page: pymupdf.Page) -> str:
        """Extract unaltered raw text from a single PDF page."""
        try:
            return page.get_text()
        except Exception as exc:
            logger.warning("Failed to extract text from page %s: %s", page.number + 1, exc)
            raise PDFExtractionError(
                f"Failed to extract text from page {page.number + 1}.",
                details=str(exc),
            ) from exc
