"""Pydantic schemas for structured document and page representation."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata extracted from the PDF document headers."""

    title: Optional[str] = Field(default=None, description="Document title from PDF metadata")
    author: Optional[str] = Field(default=None, description="Author from PDF metadata")
    subject: Optional[str] = Field(default=None, description="Subject from PDF metadata")
    creator: Optional[str] = Field(default=None, description="Application that created the original document")
    producer: Optional[str] = Field(default=None, description="Producer of the PDF")
    creation_date: Optional[str] = Field(default=None, description="Creation timestamp string")
    mod_date: Optional[str] = Field(default=None, description="Modification timestamp string")
    custom: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata entries")


class Page(BaseModel):
    """Structured representation of a single extracted PDF page."""

    page_number: int = Field(ge=1, description="1-based page number within the document")
    text: str = Field(description="Raw extracted text content for the page")
    char_count: int = Field(ge=0, description="Total character count of extracted text")
    word_count: int = Field(ge=0, description="Total word count of extracted text")
    has_text: bool = Field(description="Indicates whether the page contains extractable text")
    extraction_warning: Optional[str] = Field(default=None, description="Diagnostic warning if text extraction was suspicious or empty")


class Document(BaseModel):
    """Structured, page-aware representation of an ingested book document."""

    document_id: str = Field(description="Unique, stable identifier for the document")
    filename: str = Field(description="Base filename of the ingested document")
    source_path: str = Field(description="Resolved filesystem path to the source file")
    page_count: int = Field(ge=0, description="Total number of pages in the document")
    total_characters: int = Field(ge=0, description="Total character count across all pages")
    total_words: int = Field(ge=0, description="Total word count across all pages")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document-level metadata")
    pages: list[Page] = Field(default_factory=list, description="Ordered list of page objects")
    warnings: list[str] = Field(default_factory=list, description="Document-level extraction and diagnostic warnings")


class DocumentIngestRequest(BaseModel):
    """Request payload for minimal development ingestion endpoint."""

    file_path: str = Field(description="Absolute or relative path to the PDF file to ingest")
    document_id: Optional[str] = Field(default=None, description="Optional custom document ID")


class DocumentIngestResponse(BaseModel):
    """Response payload for document ingestion endpoint."""

    status: str = Field(default="success", description="Status of the ingestion operation")
    document: Document = Field(description="Ingested document representation")


class PersistedDocumentResponse(BaseModel):
    """Response payload representing a document persisted in PostgreSQL."""

    document_id: str
    filename: str
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: int
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Paginated list of persistent documents."""

    total: int
    documents: list[PersistedDocumentResponse]


class PersistedPageResponse(BaseModel):
    """Response payload representing an individual persisted page."""

    page_id: str
    document_id: str
    page_number: int
    text: str
    char_count: int
    word_count: int


class PersistedChunkResponse(BaseModel):
    """Response payload representing an individual persisted chunk."""

    chunk_id: str
    document_id: str
    page_id: str
    page_number: int
    chunk_index: int
    text: str
    char_count: int
    word_count: int

