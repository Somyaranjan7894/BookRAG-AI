"""Domain-specific exceptions for PDF ingestion and processing."""


class PDFIngestionError(Exception):
    """Base exception for all PDF ingestion failures."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class PDFNotFoundError(PDFIngestionError):
    """Raised when the specified PDF file cannot be found on the filesystem."""


class InvalidPDFError(PDFIngestionError):
    """Raised when the file is not a valid or readable PDF document."""


class PDFExtractionError(PDFIngestionError):
    """Raised when an unrecoverable failure occurs during page text extraction."""
