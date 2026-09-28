"""Deterministic PDF test fixture generator using PyMuPDF.

Generates reproducible synthetic PDF files without committing binaries to Git
or requiring external network dependencies.
"""

from pathlib import Path
from typing import Optional
import pymupdf


def create_deterministic_pdf(
    dest_path: Path,
    pages_content: list[str],
    metadata: Optional[dict[str, str]] = None,
) -> Path:
    """Create a multi-page PDF document with specific page strings.

    Args:
        dest_path: Destination path for the generated PDF.
        pages_content: List of strings; each item generates a page with that content.
        metadata: Optional dictionary of PDF metadata attributes.

    Returns:
        Path to the saved PDF file.
    """
    doc = pymupdf.open()
    for text in pages_content:
        page = doc.new_page()
        if text:
            page.insert_text(pymupdf.Point(50, 72), text)

    if metadata:
        doc.set_metadata(metadata)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dest_path))
    doc.close()
    return dest_path


def create_zero_page_pdf(dest_path: Path) -> Path:
    """Create a syntactically valid PDF document containing 0 pages."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    zero_page_pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"xref\n0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer\n<< /Root 1 0 R /Size 3 >>\n"
        b"startxref\n115\n%%EOF"
    )
    with open(dest_path, "wb") as f:
        f.write(zero_page_pdf_bytes)
    return dest_path



def create_corrupt_pdf(dest_path: Path) -> Path:
    """Create a corrupt file that has a .pdf extension but invalid content."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(b"NOT_A_VALID_PDF_HEADER_JUST_RANDOM_GARBAGE_BYTES\x00\x01\x02\x03")
    return dest_path
