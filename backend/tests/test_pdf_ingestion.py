"""Unit and integration tests for Phase 1 PDF ingestion and page representation."""

from pathlib import Path
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.schemas.document import Document
from app.services.pdf.exceptions import (
    InvalidPDFError,
    PDFNotFoundError,
)
from app.services.pdf.ingestion import PDFIngestionService


def test_valid_pdf_opens_and_ingests_successfully(deterministic_sample_pdf: Path) -> None:
    """Requirement 1: Valid PDF opens successfully and returns structured Document."""
    service = PDFIngestionService()
    doc = service.ingest_pdf(deterministic_sample_pdf)

    assert isinstance(doc, Document)
    assert doc.filename == "sample_book.pdf"
    assert doc.source_path == str(deterministic_sample_pdf.resolve())


def test_correct_page_count(deterministic_sample_pdf: Path) -> None:
    """Requirement 2: Accurate total page count is preserved."""
    service = PDFIngestionService()
    doc = service.ingest_pdf(deterministic_sample_pdf)

    assert doc.page_count == 3
    assert len(doc.pages) == 3


def test_one_based_page_numbering(deterministic_sample_pdf: Path) -> None:
    """Requirement 3: Page numbering must be strictly 1-based."""
    service = PDFIngestionService()
    doc = service.ingest_pdf(deterministic_sample_pdf)

    page_numbers = [page.page_number for page in doc.pages]
    assert page_numbers == [1, 2, 3]


def test_text_extracted_from_correct_pages(deterministic_sample_pdf: Path) -> None:
    """Requirement 4: Text content is extracted accurately from each respective page."""
    service = PDFIngestionService()
    doc = service.ingest_pdf(deterministic_sample_pdf)

    assert "BookRAG AI Phase 1" in doc.pages[0].text
    assert "This is a PDF ingestion test." in doc.pages[1].text
    assert "Retrieval comes later." in doc.pages[2].text


def test_character_counts_populated_correctly(deterministic_sample_pdf: Path) -> None:
    """Requirement 5: Character counts reflect extracted text accurately."""
    service = PDFIngestionService()
    doc = service.ingest_pdf(deterministic_sample_pdf)

    for page in doc.pages:
        assert page.char_count == len(page.text)
        assert page.char_count > 0

    assert doc.total_characters == sum(p.char_count for p in doc.pages)


def test_word_counts_populated_correctly(deterministic_sample_pdf: Path) -> None:
    """Requirement 6: Word counts match token counts of page text."""
    service = PDFIngestionService()
    doc = service.ingest_pdf(deterministic_sample_pdf)

    for page in doc.pages:
        expected_words = len(page.text.split())
        assert page.word_count == expected_words

    assert doc.total_words == sum(p.word_count for p in doc.pages)


def test_empty_and_low_text_diagnostics(empty_and_low_text_pdf: Path) -> None:
    """Requirement 7: Empty and low-text pages produce warnings without crashing ingestion."""
    service = PDFIngestionService()
    doc = service.ingest_pdf(empty_and_low_text_pdf)

    assert doc.page_count == 3

    # Page 1: normal text
    assert doc.pages[0].has_text is True
    assert doc.pages[0].extraction_warning is None

    # Page 2: empty page
    assert doc.pages[1].has_text is False
    assert doc.pages[1].extraction_warning is not None
    assert "Empty page" in doc.pages[1].extraction_warning

    # Page 3: low text page (< 50 chars)
    assert doc.pages[2].has_text is True
    assert doc.pages[2].extraction_warning is not None
    assert "Low text page" in doc.pages[2].extraction_warning

    # Document-level aggregated warnings
    assert len(doc.warnings) == 2


def test_missing_file_raises_controlled_error(tmp_path: Path) -> None:
    """Requirement 8: Missing file raises controlled PDFNotFoundError."""
    service = PDFIngestionService()
    nonexistent = tmp_path / "does_not_exist.pdf"

    with pytest.raises(PDFNotFoundError) as exc_info:
        service.ingest_pdf(nonexistent)

    assert "File not found" in str(exc_info.value)


def test_invalid_pdf_raises_controlled_error(corrupt_pdf: Path) -> None:
    """Requirement 9: Corrupt or non-PDF file raises controlled InvalidPDFError."""
    service = PDFIngestionService()

    with pytest.raises(InvalidPDFError) as exc_info:
        service.ingest_pdf(corrupt_pdf)

    assert "corrupt or not a readable PDF" in str(exc_info.value) or "not recognized as a valid PDF" in str(exc_info.value)


def test_zero_page_pdf_raises_controlled_error(zero_page_pdf: Path) -> None:
    """Zero-page PDF raises controlled InvalidPDFError."""
    service = PDFIngestionService()

    with pytest.raises(InvalidPDFError) as exc_info:
        service.ingest_pdf(zero_page_pdf)

    assert "0 pages" in str(exc_info.value)


def test_document_metadata_preserved(deterministic_sample_pdf: Path) -> None:
    """Requirement 10: PDF metadata fields are extracted and preserved."""
    service = PDFIngestionService()
    doc = service.ingest_pdf(deterministic_sample_pdf)

    assert doc.metadata.title == "BookRAG AI Test Document"
    assert doc.metadata.author == "Antigravity Engineering"
    assert doc.metadata.subject == "Phase 1 Ingestion Verification"


def test_deterministic_document_id_generation(deterministic_sample_pdf: Path) -> None:
    """Document ID is generated deterministically from file content and remains stable."""
    service = PDFIngestionService()
    doc1 = service.ingest_pdf(deterministic_sample_pdf)
    doc2 = service.ingest_pdf(deterministic_sample_pdf)

    assert doc1.document_id.startswith("doc_")
    assert doc1.document_id == doc2.document_id


def test_custom_document_id_supported(deterministic_sample_pdf: Path) -> None:
    """Caller can provide a custom document_id which is preserved."""
    service = PDFIngestionService()
    custom_id = "custom_book_12345"
    doc = service.ingest_pdf(deterministic_sample_pdf, document_id=custom_id)

    assert doc.document_id == custom_id


def test_api_ingest_endpoint_success(client: TestClient, deterministic_sample_pdf: Path) -> None:
    """Minimal dev API endpoint ingests PDF and returns 200 with structured Document."""
    response = client.post(
        "/api/v1/documents/ingest",
        json={"file_path": str(deterministic_sample_pdf)},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "success"
    assert data["document"]["page_count"] == 3
    assert len(data["document"]["pages"]) == 3
    assert data["document"]["pages"][0]["page_number"] == 1


def test_api_ingest_endpoint_not_found(client: TestClient, tmp_path: Path) -> None:
    """API endpoint returns 404 for nonexistent file."""
    response = client.post(
        "/api/v1/documents/ingest",
        json={"file_path": str(tmp_path / "missing.pdf")},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 404


def test_api_ingest_endpoint_invalid_pdf(client: TestClient, corrupt_pdf: Path) -> None:
    """API endpoint returns 400 for corrupt PDF."""
    response = client.post(
        "/api/v1/documents/ingest",
        json={"file_path": str(corrupt_pdf)},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 400
