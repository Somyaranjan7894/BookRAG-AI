"""High-level PDF ingestion service for BookRAG AI.

Orchestrates PDF file validation, safe extraction, page-level metadata tracking,
and diagnostic health checks for extracted content.
"""

import hashlib
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.document import Document, Page
from app.services.pdf.exceptions import (
    InvalidPDFError,
    PDFIngestionError,
    PDFNotFoundError,
)
from app.services.pdf.parser import PDFParser

logger = get_logger(__name__)

# Minimum number of non-whitespace characters expected for a standard readable page.
LOW_TEXT_CHAR_THRESHOLD = 50


class PDFIngestionService:
    """Service handling validation and ingestion of PDF book documents."""

    def __init__(self, parser: Optional[PDFParser] = None) -> None:
        self.parser = parser or PDFParser()

    def _generate_deterministic_id(self, file_path: Path) -> str:
        """Compute a deterministic document identifier based on SHA-256 of file content.

        Ensures identical documents consistently receive the same ID across ingestion runs
        without external database dependencies.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return f"doc_{hasher.hexdigest()[:16]}"

    def _validate_file_path(self, raw_path: Path | str) -> Path:
        """Ensure the specified path exists, is a regular file, and has valid sizing."""
        resolved = Path(raw_path).resolve()
        if not resolved.exists():
            logger.error("PDF ingestion failed: File not found at '%s'", raw_path)
            raise PDFNotFoundError(f"File not found at path '{raw_path}'.")

        if not resolved.is_file():
            logger.error("PDF ingestion failed: Path '%s' is not a regular file", raw_path)
            raise InvalidPDFError(f"Path '{raw_path}' is not a regular file.")

        if resolved.stat().st_size == 0:
            logger.error("PDF ingestion failed: File '%s' is 0 bytes", resolved.name)
            raise InvalidPDFError(f"File '{resolved.name}' is empty (0 bytes).")

        if resolved.stat().st_size > settings.MAX_UPLOAD_FILE_SIZE_BYTES:
            logger.error(
                "PDF ingestion failed: File '%s' (%d bytes) exceeds maximum limit (%d bytes)",
                resolved.name,
                resolved.stat().st_size,
                settings.MAX_UPLOAD_FILE_SIZE_BYTES,
            )
            raise InvalidPDFError(
                f"File '{resolved.name}' exceeds maximum allowed upload size of "
                f"{settings.MAX_UPLOAD_FILE_SIZE_BYTES // (1024 * 1024)} MB."
            )

        return resolved

    def ingest_pdf(
        self,
        file_path: Path | str,
        document_id: Optional[str] = None,
    ) -> Document:
        """Ingest a PDF document, extract page-by-page text, and return structured Document.

        Args:
            file_path: Filesystem path to the target PDF document.
            document_id: Optional caller-specified unique ID. If omitted, a deterministic
                         hash of the file content is generated.

        Returns:
            Document: Structured Pydantic representation with page details and diagnostics.

        Raises:
            PDFNotFoundError: If the target file does not exist.
            InvalidPDFError: If the document is malformed, corrupt, or has 0 pages.
            PDFExtractionError: If extraction fails due to internal parser issues.
        """
        path = self._validate_file_path(file_path)
        stable_doc_id = document_id or self._generate_deterministic_id(path)

        logger.info(
            "Starting ingestion for file '%s' (Document ID: %s)",
            path.name,
            stable_doc_id,
        )

        try:
            with self.parser.open_document(path) as doc:
                logger.info("PDF opened successfully: '%s'", path.name)
                page_count = len(doc)
                logger.info("Detected %d page(s) in '%s'", page_count, path.name)

                if page_count == 0:
                    raise InvalidPDFError(
                        f"PDF document '{path.name}' contains 0 pages."
                    )

                if page_count > settings.MAX_UPLOAD_PAGE_COUNT:
                    raise InvalidPDFError(
                        f"PDF document '{path.name}' has {page_count} pages, "
                        f"exceeding the maximum allowed limit of {settings.MAX_UPLOAD_PAGE_COUNT} pages."
                    )

                metadata = self.parser.extract_metadata(doc)
                pages: list[Page] = []
                document_warnings: list[str] = []

                for index, raw_page in enumerate(doc):
                    page_number = index + 1
                    raw_text = self.parser.extract_page_text(raw_page)

                    char_count = len(raw_text)
                    word_count = len(raw_text.split())
                    stripped_text = raw_text.strip()
                    has_text = len(stripped_text) > 0

                    warning: Optional[str] = None
                    if not has_text:
                        warning = f"Page {page_number}: Empty page with no extractable text."
                    elif len(stripped_text) < LOW_TEXT_CHAR_THRESHOLD:
                        warning = (
                            f"Page {page_number}: Low text page with only "
                            f"{len(stripped_text)} non-whitespace characters."
                        )

                    if warning:
                        logger.warning(
                            "Extraction diagnostic [%s]: %s",
                            path.name,
                            warning,
                        )
                        document_warnings.append(warning)

                    pages.append(
                        Page(
                            page_number=page_number,
                            text=raw_text,
                            char_count=char_count,
                            word_count=word_count,
                            has_text=has_text,
                            extraction_warning=warning,
                        )
                    )

                total_chars = sum(p.char_count for p in pages)
                total_words = sum(p.word_count for p in pages)

                document = Document(
                    document_id=stable_doc_id,
                    filename=path.name,
                    source_path=str(path),
                    page_count=page_count,
                    total_characters=total_chars,
                    total_words=total_words,
                    metadata=metadata,
                    pages=pages,
                    warnings=document_warnings,
                )

                logger.info(
                    "Completed ingestion for '%s': %d page(s), %d characters, %d words, %d warning(s)",
                    path.name,
                    page_count,
                    total_chars,
                    total_words,
                    len(document_warnings),
                )
                return document

        except PDFIngestionError:
            # Re-raise known domain exceptions
            raise
        except Exception as exc:
            logger.exception("Unexpected error during PDF ingestion for '%s': %s", path.name, exc)
            raise PDFIngestionError(
                f"Unexpected failure while ingesting '{path.name}'.",
                details=str(exc),
            ) from exc
