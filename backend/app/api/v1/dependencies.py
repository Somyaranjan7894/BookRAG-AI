"""Centralized FastAPI dependency injection layer for BookRAG AI.

Provides declarative dependency providers for database sessions, repositories,
and domain services, enabling clean service boundaries and test substitution.
"""

from collections.abc import Generator
from typing import Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.session import get_db, get_session_factory
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.services.citation.service import CitationService
from app.services.document_processing.service import DocumentProcessingService
from app.services.document_processing.upload_service import DocumentUploadService
from app.services.embeddings.service import EmbeddingService
from app.services.generation.service import GenerationService
from app.services.grounding.orchestrator import GroundedAnswerService
from app.services.pdf.ingestion import PDFIngestionService
from app.services.persistence import DocumentPersistenceService
from app.services.qa.service import QAService
from app.services.query_understanding.service import QueryUnderstandingService
from app.services.question_generation.service import QuestionGenerationService
from app.services.reranking.service import RerankerService
from app.services.retrieval.service import RetrievalService
from app.services.search.service import SearchService
from app.services.text.chunker import Chunker
from app.services.text.cleaner import TextCleaner
from app.services.text.processor import TextProcessingService

logger = get_logger(__name__)

# Singletons for cached service instances
_dev_retrieval_service: Optional[RetrievalService] = None
_reranker_service: Optional[RerankerService] = None
_search_service: Optional[SearchService] = None
_qa_service: Optional[QAService] = None
_generation_service: Optional[GenerationService] = None
_grounded_answer_service: Optional[GroundedAnswerService] = None
_question_generation_service: Optional[QuestionGenerationService] = None
_query_understanding_service: Optional[QueryUnderstandingService] = None
_embedding_service: Optional[EmbeddingService] = None
_citation_service: Optional[CitationService] = None


# ------------------------------------------------------------------------------
# Configuration Dependency
# ------------------------------------------------------------------------------

def get_settings_dependency() -> Settings:
    """Provide application settings."""
    return get_settings()


# ------------------------------------------------------------------------------
# Database Session Dependencies
# ------------------------------------------------------------------------------

def get_db_session() -> Generator[Session, None, None]:
    """Provide a request-scoped database session."""
    yield from get_db()


def get_optional_db_session() -> Generator[Optional[Session], None, None]:
    """Provide a database session if database infrastructure is reachable.

    Falls back to yielding None if PostgreSQL is not reachable, enabling
    graceful degradation in environments without active database storage.
    """
    try:
        session = get_session_factory()()
    except Exception as exc:
        logger.debug("Database unreachable for optional persistence: %s", exc)
        yield None
        return

    try:
        yield session
    finally:
        session.close()


def get_required_db_session(
    db: Optional[Session] = Depends(get_optional_db_session),
) -> Session:
    """Provide a required database session, raising 503 if unreachable."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable",
        )
    return db


# ------------------------------------------------------------------------------
# Repositories & Persistence Dependencies
# ------------------------------------------------------------------------------

def get_document_repository(
    db: Session = Depends(get_required_db_session),
) -> DocumentRepository:
    """Provide a DocumentRepository bound to the current database session."""
    return DocumentRepository(db)


def get_document_persistence_service(
    db: Session = Depends(get_required_db_session),
) -> DocumentPersistenceService:
    """Provide a DocumentPersistenceService bound to the active database session."""
    return DocumentPersistenceService(db)


def get_optional_persistence_service(
    db: Optional[Session] = Depends(get_optional_db_session),
) -> Optional[DocumentPersistenceService]:
    """Provide DocumentPersistenceService if database is reachable, else None."""
    if db is None:
        return None
    return DocumentPersistenceService(db)


# ------------------------------------------------------------------------------
# Document Processing & Upload Dependencies
# ------------------------------------------------------------------------------

def get_document_upload_service(
    db: Optional[Session] = Depends(get_optional_db_session),
) -> DocumentUploadService:
    """Provide DocumentUploadService with optional database persistence."""
    return DocumentUploadService(session=db)


def get_document_processing_service(
    db: Session = Depends(get_required_db_session),
) -> DocumentProcessingService:
    """Provide full DocumentProcessingService for backend document pipeline."""
    return DocumentProcessingService(session=db)


# ------------------------------------------------------------------------------
# Ingestion & Text Processing Dependencies
# ------------------------------------------------------------------------------

def get_pdf_ingestion_service() -> PDFIngestionService:
    """Provide PDF ingestion service."""
    return PDFIngestionService()


def get_text_cleaner() -> type[TextCleaner]:
    """Provide TextCleaner class."""
    return TextCleaner


def get_chunker() -> Chunker:
    """Provide Chunker instance."""
    return Chunker()


def get_text_processing_service() -> TextProcessingService:
    """Provide TextProcessingService."""
    return TextProcessingService()


# ------------------------------------------------------------------------------
# Retrieval, Search & Embeddings Dependencies
# ------------------------------------------------------------------------------

def get_embedding_service() -> EmbeddingService:
    """Provide cached EmbeddingService singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_dev_retrieval_service() -> RetrievalService:
    """Retrieve shared development RetrievalService holding in-memory indexes."""
    global _dev_retrieval_service
    if _dev_retrieval_service is None:
        _dev_retrieval_service = RetrievalService()
    return _dev_retrieval_service


def get_reranker_service() -> RerankerService:
    """Provide cached RerankerService singleton."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service


def get_search_service() -> SearchService:
    """Dependency provider for SearchService.

    Reuses the shared development RetrievalService so indexed chunks from
    /api/v1/retrieval are immediately searchable.
    """
    global _search_service
    if _search_service is None:
        dev_retrieval_svc = get_dev_retrieval_service()
        reranker_svc = get_reranker_service()
        _search_service = SearchService(
            retrieval_service=dev_retrieval_svc,
            reranker_service=reranker_svc,
        )
    return _search_service


# ------------------------------------------------------------------------------
# QA, Generation, Grounding & Planning Dependencies
# ------------------------------------------------------------------------------

def get_qa_service() -> QAService:
    """Provide cached QAService singleton."""
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service


def get_generation_service() -> GenerationService:
    """Provide cached GenerationService singleton."""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service


def get_citation_service() -> CitationService:
    """Provide cached CitationService singleton."""
    global _citation_service
    if _citation_service is None:
        _citation_service = CitationService()
    return _citation_service


def get_grounded_answer_service() -> GroundedAnswerService:
    """Provide cached GroundedAnswerService singleton."""
    global _grounded_answer_service
    if _grounded_answer_service is None:
        _grounded_answer_service = GroundedAnswerService()
    return _grounded_answer_service


def get_question_generation_service() -> QuestionGenerationService:
    """Provide cached QuestionGenerationService singleton."""
    global _question_generation_service
    if _question_generation_service is None:
        _question_generation_service = QuestionGenerationService()
    return _question_generation_service


def get_query_understanding_service() -> QueryUnderstandingService:
    """Provide cached QueryUnderstandingService singleton."""
    global _query_understanding_service
    if _query_understanding_service is None:
        _query_understanding_service = QueryUnderstandingService()
    return _query_understanding_service
