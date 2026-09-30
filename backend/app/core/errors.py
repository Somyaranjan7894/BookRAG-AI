"""Application error handling configuration for BookRAG AI.

Ensures structured, sanitized error responses with request correlation tracking
without leaking internal stack traces or secrets to clients.
"""

from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.correlation import get_request_id
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_error_response(
    request: Request,
    status_code: int,
    message: str,
    error_type: Optional[str] = None,
    details: Optional[Any] = None,
) -> JSONResponse:
    """Construct a sanitized, standardized JSON error response.

    Args:
        request: Incoming FastAPI request instance.
        status_code: Numeric HTTP status code.
        message: Human-readable error description.
        error_type: Machine-readable uppercase error identifier.
        details: Optional structured contextual details or schema errors.

    Returns:
        JSONResponse containing sanitized error envelope and X-Request-ID header.
    """
    request_id = getattr(request.state, "request_id", None) or get_request_id() or ""

    error_payload: dict[str, Any] = {
        "code": status_code,
        "message": message,
        "request_id": request_id,
    }
    if error_type:
        error_payload["error_type"] = error_type
    if details is not None:
        error_payload["details"] = details

    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=status_code,
        content={"error": error_payload},
        headers=headers,
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Register core and domain exception handlers with the FastAPI application."""

    # --------------------------------------------------------------------------
    # Standard HTTP & Validation Exceptions
    # --------------------------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle standard HTTP exceptions with structured response."""
        error_type = "HTTP_ERROR"
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            error_type = "NOT_FOUND"
        elif exc.status_code == status.HTTP_400_BAD_REQUEST:
            error_type = "BAD_REQUEST"
        elif exc.status_code == status.HTTP_403_FORBIDDEN:
            error_type = "FORBIDDEN"
        elif exc.status_code == status.HTTP_401_UNAUTHORIZED:
            error_type = "UNAUTHORIZED"
        elif exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            error_type = "SERVICE_UNAVAILABLE"

        return create_error_response(
            request=request,
            status_code=exc.status_code,
            message=str(exc.detail),
            error_type=error_type,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle schema validation errors safely."""
        clean_errors = []
        for err in exc.errors():
            item = dict(err)
            if "ctx" in item and isinstance(item["ctx"], dict):
                item["ctx"] = {k: str(v) for k, v in item["ctx"].items()}
            clean_errors.append(item)

        return create_error_response(
            request=request,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Request validation error",
            error_type="VALIDATION_ERROR",
            details=jsonable_encoder(clean_errors),
        )

    # --------------------------------------------------------------------------
    # PDF Ingestion Exceptions (Phase 1)
    # --------------------------------------------------------------------------
    from app.services.pdf.exceptions import (
        InvalidPDFError,
        PDFIngestionError,
        PDFNotFoundError,
    )

    @app.exception_handler(PDFNotFoundError)
    async def pdf_not_found_handler(request: Request, exc: PDFNotFoundError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_404_NOT_FOUND,
            message=exc.message,
            error_type="PDF_NOT_FOUND",
        )

    @app.exception_handler(InvalidPDFError)
    async def invalid_pdf_handler(request: Request, exc: InvalidPDFError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_PDF",
        )

    @app.exception_handler(PDFIngestionError)
    async def pdf_ingestion_handler(request: Request, exc: PDFIngestionError) -> JSONResponse:
        logger.error("PDF ingestion error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="PDF_INGESTION_ERROR",
        )

    # --------------------------------------------------------------------------
    # Text Processing Exceptions (Phase 2)
    # --------------------------------------------------------------------------
    from app.services.text.exceptions import (
        InvalidChunkingConfigError,
        TextProcessingError,
    )

    @app.exception_handler(InvalidChunkingConfigError)
    async def invalid_chunking_config_handler(request: Request, exc: InvalidChunkingConfigError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_CHUNKING_CONFIG",
            details=exc.details,
        )

    @app.exception_handler(TextProcessingError)
    async def text_processing_handler(request: Request, exc: TextProcessingError) -> JSONResponse:
        logger.error("Text processing error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="TEXT_PROCESSING_ERROR",
            details=exc.details,
        )

    # --------------------------------------------------------------------------
    # Embeddings Exceptions (Phase 3)
    # --------------------------------------------------------------------------
    from app.services.embeddings.exceptions import (
        EmbeddingError,
        InvalidChunkError,
        ModelLoadError,
    )

    @app.exception_handler(InvalidChunkError)
    async def invalid_chunk_handler(request: Request, exc: InvalidChunkError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_CHUNK",
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
        logger.error("Embedding model load error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="MODEL_LOAD_ERROR",
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
        logger.error("Embedding service error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="EMBEDDING_ERROR",
        )

    # --------------------------------------------------------------------------
    # Retrieval Exceptions (Phase 4)
    # --------------------------------------------------------------------------
    from app.services.retrieval.exceptions import (
        CorruptedIndexError,
        IndexDimensionMismatchError,
        IndexNotFoundError,
        IndexPersistenceError,
        InvalidQueryError,
        RetrievalError,
    )

    @app.exception_handler(InvalidQueryError)
    async def invalid_query_handler(request: Request, exc: InvalidQueryError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_QUERY",
        )

    @app.exception_handler(IndexDimensionMismatchError)
    async def dimension_mismatch_handler(request: Request, exc: IndexDimensionMismatchError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="DIMENSION_MISMATCH",
        )

    @app.exception_handler(IndexNotFoundError)
    async def index_not_found_handler(request: Request, exc: IndexNotFoundError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_404_NOT_FOUND,
            message=exc.message,
            error_type="INDEX_NOT_FOUND",
        )

    @app.exception_handler(CorruptedIndexError)
    async def corrupted_index_handler(request: Request, exc: CorruptedIndexError) -> JSONResponse:
        logger.error("Corrupted index detected on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="CORRUPTED_INDEX",
        )

    @app.exception_handler(IndexPersistenceError)
    async def index_persistence_handler(request: Request, exc: IndexPersistenceError) -> JSONResponse:
        logger.error("Index persistence error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="INDEX_PERSISTENCE_ERROR",
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
        logger.error("Retrieval error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="RETRIEVAL_ERROR",
        )

    # --------------------------------------------------------------------------
    # Search Exceptions (Phase 5)
    # --------------------------------------------------------------------------
    from app.services.search.exceptions import (
        DocumentNotFoundError as SearchDocumentNotFoundError,
        IndexNotInitializedError,
        InvalidSearchQueryError,
        InvalidTopKError,
        SearchEmbeddingError,
        SearchError,
        SearchRetrievalError,
    )

    @app.exception_handler(InvalidSearchQueryError)
    async def invalid_search_query_handler(request: Request, exc: InvalidSearchQueryError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_SEARCH_QUERY",
        )

    @app.exception_handler(InvalidTopKError)
    async def invalid_top_k_handler(request: Request, exc: InvalidTopKError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_TOP_K",
        )

    @app.exception_handler(SearchDocumentNotFoundError)
    async def search_document_not_found_handler(request: Request, exc: SearchDocumentNotFoundError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_404_NOT_FOUND,
            message=exc.message,
            error_type="DOCUMENT_NOT_FOUND",
        )

    @app.exception_handler(IndexNotInitializedError)
    async def index_not_initialized_handler(request: Request, exc: IndexNotInitializedError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_404_NOT_FOUND,
            message=exc.message,
            error_type="INDEX_NOT_INITIALIZED",
        )

    @app.exception_handler(SearchEmbeddingError)
    async def search_embedding_error_handler(request: Request, exc: SearchEmbeddingError) -> JSONResponse:
        logger.error("Search embedding error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="SEARCH_EMBEDDING_ERROR",
        )

    @app.exception_handler(SearchRetrievalError)
    async def search_retrieval_error_handler(request: Request, exc: SearchRetrievalError) -> JSONResponse:
        logger.error("Search retrieval error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="SEARCH_RETRIEVAL_ERROR",
        )

    @app.exception_handler(SearchError)
    async def general_search_error_handler(request: Request, exc: SearchError) -> JSONResponse:
        logger.error("General search service error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="SEARCH_ERROR",
        )

    # --------------------------------------------------------------------------
    # Reranking Exceptions (Phase 6)
    # --------------------------------------------------------------------------
    from app.services.reranking.exceptions import (
        InvalidRerankingConfigError,
        RerankerModelLoadError,
        RerankingError,
        RerankingExecutionError,
        ScoreAlignmentError,
    )

    @app.exception_handler(InvalidRerankingConfigError)
    async def invalid_reranking_config_handler(request: Request, exc: InvalidRerankingConfigError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_RERANKING_CONFIG",
        )

    @app.exception_handler(RerankerModelLoadError)
    async def reranker_model_load_handler(request: Request, exc: RerankerModelLoadError) -> JSONResponse:
        logger.error("Reranker model load error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="RERANKER_MODEL_LOAD_ERROR",
        )

    @app.exception_handler(ScoreAlignmentError)
    async def score_alignment_error_handler(request: Request, exc: ScoreAlignmentError) -> JSONResponse:
        logger.error("Score alignment error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="SCORE_ALIGNMENT_ERROR",
        )

    @app.exception_handler(RerankingExecutionError)
    async def reranking_execution_handler(request: Request, exc: RerankingExecutionError) -> JSONResponse:
        logger.error("Reranking execution error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="RERANKING_EXECUTION_ERROR",
        )

    @app.exception_handler(RerankingError)
    async def general_reranking_handler(request: Request, exc: RerankingError) -> JSONResponse:
        logger.error("General reranking error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="RERANKING_ERROR",
        )

    # --------------------------------------------------------------------------
    # QA Exceptions (Phase 7)
    # --------------------------------------------------------------------------
    from app.services.qa.exceptions import (
        InvalidAnswerSpanError,
        InvalidQAConfigError,
        InvalidQAEvidenceError,
        InvalidQAQueryError,
        QAError,
        QAInferenceError,
        QAModelLoadError,
    )

    @app.exception_handler(InvalidQAQueryError)
    async def invalid_qa_query_handler(request: Request, exc: InvalidQAQueryError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_QA_QUERY",
        )

    @app.exception_handler(InvalidQAEvidenceError)
    async def invalid_qa_evidence_handler(request: Request, exc: InvalidQAEvidenceError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_QA_EVIDENCE",
        )

    @app.exception_handler(InvalidAnswerSpanError)
    async def invalid_answer_span_handler(request: Request, exc: InvalidAnswerSpanError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_ANSWER_SPAN",
        )

    @app.exception_handler(InvalidQAConfigError)
    async def invalid_qa_config_handler(request: Request, exc: InvalidQAConfigError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_QA_CONFIG",
        )

    @app.exception_handler(QAModelLoadError)
    async def qa_model_load_handler(request: Request, exc: QAModelLoadError) -> JSONResponse:
        logger.error("QA model load error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=exc.message,
            error_type="QA_MODEL_LOAD_ERROR",
        )

    @app.exception_handler(QAInferenceError)
    async def qa_inference_handler(request: Request, exc: QAInferenceError) -> JSONResponse:
        logger.error("QA inference error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="QA_INFERENCE_ERROR",
        )

    @app.exception_handler(QAError)
    async def general_qa_error_handler(request: Request, exc: QAError) -> JSONResponse:
        logger.error("General QA error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="QA_ERROR",
        )

    # --------------------------------------------------------------------------
    # Generation Exceptions (Phase 8)
    # --------------------------------------------------------------------------
    from app.services.generation.exceptions import (
        ContextBudgetExceededError,
        GenerationError,
        GenerationInferenceError,
        GenerationModelLoadError,
        InvalidGenerationConfigError,
        InvalidGenerationEvidenceError,
        InvalidGenerationQueryError,
    )

    @app.exception_handler(InvalidGenerationQueryError)
    async def invalid_generation_query_handler(request: Request, exc: InvalidGenerationQueryError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_GENERATION_QUERY",
        )

    @app.exception_handler(InvalidGenerationEvidenceError)
    async def invalid_generation_evidence_handler(request: Request, exc: InvalidGenerationEvidenceError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_GENERATION_EVIDENCE",
        )

    @app.exception_handler(InvalidGenerationConfigError)
    async def invalid_generation_config_handler(request: Request, exc: InvalidGenerationConfigError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_GENERATION_CONFIG",
        )

    @app.exception_handler(ContextBudgetExceededError)
    async def context_budget_exceeded_handler(request: Request, exc: ContextBudgetExceededError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="CONTEXT_BUDGET_EXCEEDED",
        )

    @app.exception_handler(GenerationModelLoadError)
    async def generation_model_load_handler(request: Request, exc: GenerationModelLoadError) -> JSONResponse:
        logger.error("Generation model load error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=exc.message,
            error_type="GENERATION_MODEL_LOAD_ERROR",
        )

    @app.exception_handler(GenerationInferenceError)
    async def generation_inference_handler(request: Request, exc: GenerationInferenceError) -> JSONResponse:
        logger.error("Generation inference error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="GENERATION_INFERENCE_ERROR",
        )

    @app.exception_handler(GenerationError)
    async def general_generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
        logger.error("General generation error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="GENERATION_ERROR",
        )

    # --------------------------------------------------------------------------
    # Grounding Exceptions (Phase 9)
    # --------------------------------------------------------------------------
    from app.services.grounding.exceptions import (
        GroundingError,
        InvalidGroundingConfigError,
        InvalidGroundingInputError,
        NLIInferenceError,
        NLILabelMappingError,
        NLIModelLoadError,
    )

    @app.exception_handler(InvalidGroundingInputError)
    async def invalid_grounding_input_handler(request: Request, exc: InvalidGroundingInputError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_GROUNDING_INPUT",
        )

    @app.exception_handler(InvalidGroundingConfigError)
    async def invalid_grounding_config_handler(request: Request, exc: InvalidGroundingConfigError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_GROUNDING_CONFIG",
        )

    @app.exception_handler(NLIModelLoadError)
    async def nli_model_load_handler(request: Request, exc: NLIModelLoadError) -> JSONResponse:
        logger.error("NLI model load error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=exc.message,
            error_type="NLI_MODEL_LOAD_ERROR",
        )

    @app.exception_handler(NLILabelMappingError)
    async def nli_label_mapping_handler(request: Request, exc: NLILabelMappingError) -> JSONResponse:
        logger.error("NLI label mapping error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="NLI_LABEL_MAPPING_ERROR",
        )

    @app.exception_handler(NLIInferenceError)
    async def nli_inference_handler(request: Request, exc: NLIInferenceError) -> JSONResponse:
        logger.error("NLI inference error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="NLI_INFERENCE_ERROR",
        )

    @app.exception_handler(GroundingError)
    async def general_grounding_error_handler(request: Request, exc: GroundingError) -> JSONResponse:
        logger.error("General grounding error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="GROUNDING_ERROR",
        )

    # --------------------------------------------------------------------------
    # Citation Exceptions (Phase 10)
    # --------------------------------------------------------------------------
    from app.services.citation.exceptions import (
        CitationError,
        DocumentIsolationError,
        InvalidCitationInputError,
    )

    @app.exception_handler(InvalidCitationInputError)
    async def invalid_citation_input_handler(request: Request, exc: InvalidCitationInputError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_CITATION_INPUT",
        )

    @app.exception_handler(DocumentIsolationError)
    async def document_isolation_error_handler(request: Request, exc: DocumentIsolationError) -> JSONResponse:
        logger.warning("Document isolation violation on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="DOCUMENT_ISOLATION_ERROR",
        )

    @app.exception_handler(CitationError)
    async def general_citation_error_handler(request: Request, exc: CitationError) -> JSONResponse:
        logger.error("General citation error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="CITATION_ERROR",
        )

    # --------------------------------------------------------------------------
    # Query Understanding Exceptions (Phase 11)
    # --------------------------------------------------------------------------
    from app.services.query_understanding.exceptions import (
        InvalidQueryError as QUInvalidQueryError,
        QueryUnderstandingError,
    )

    @app.exception_handler(QUInvalidQueryError)
    async def qu_invalid_query_error_handler(request: Request, exc: QUInvalidQueryError) -> JSONResponse:
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="INVALID_QUERY_UNDERSTANDING",
        )

    @app.exception_handler(QueryUnderstandingError)
    async def query_understanding_error_handler(request: Request, exc: QueryUnderstandingError) -> JSONResponse:
        logger.error("Query understanding error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="QUERY_UNDERSTANDING_ERROR",
        )

    # --------------------------------------------------------------------------
    # Question Generation Exceptions (Phase 12)
    # --------------------------------------------------------------------------
    from app.services.question_generation.exceptions import (
        QuestionGenerationError,
        QuestionInferenceError,
        QuestionModelLoadError,
    )

    @app.exception_handler(QuestionModelLoadError)
    async def question_model_load_handler(request: Request, exc: QuestionModelLoadError) -> JSONResponse:
        logger.error("Question model load error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=exc.message,
            error_type="QUESTION_MODEL_LOAD_ERROR",
        )

    @app.exception_handler(QuestionInferenceError)
    async def question_inference_handler(request: Request, exc: QuestionInferenceError) -> JSONResponse:
        logger.error("Question inference error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="QUESTION_INFERENCE_ERROR",
        )

    @app.exception_handler(QuestionGenerationError)
    async def question_generation_error_handler(request: Request, exc: QuestionGenerationError) -> JSONResponse:
        logger.error("Question generation error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="QUESTION_GENERATION_ERROR",
        )

    # --------------------------------------------------------------------------
    # Relational Persistence Exceptions (Phase 13 / 14)
    # --------------------------------------------------------------------------
    from app.services.persistence.exceptions import (
        ConstraintViolationError,
        DocumentAlreadyExistsError,
        DocumentNotFoundError as PersistenceDocumentNotFoundError,
        PersistenceError,
        ReferentialIntegrityError,
    )

    @app.exception_handler(PersistenceDocumentNotFoundError)
    async def persistence_document_not_found_handler(
        request: Request, exc: PersistenceDocumentNotFoundError
    ) -> JSONResponse:
        logger.warning("Document not found: %s", exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_404_NOT_FOUND,
            message=exc.message,
            error_type="DOCUMENT_NOT_FOUND",
            details=exc.details,
        )

    @app.exception_handler(DocumentAlreadyExistsError)
    async def document_already_exists_handler(request: Request, exc: DocumentAlreadyExistsError) -> JSONResponse:
        logger.warning("Document already exists: %s", exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_409_CONFLICT,
            message=exc.message,
            error_type="DOCUMENT_ALREADY_EXISTS",
            details=exc.details,
        )

    @app.exception_handler(ConstraintViolationError)
    async def constraint_violation_handler(request: Request, exc: ConstraintViolationError) -> JSONResponse:
        logger.warning("Database constraint violation on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="CONSTRAINT_VIOLATION",
            details=exc.details,
        )

    @app.exception_handler(ReferentialIntegrityError)
    async def referential_integrity_handler(request: Request, exc: ReferentialIntegrityError) -> JSONResponse:
        logger.warning("Database referential integrity violation on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="REFERENTIAL_INTEGRITY_ERROR",
            details=exc.details,
        )

    @app.exception_handler(PersistenceError)
    async def general_persistence_error_handler(request: Request, exc: PersistenceError) -> JSONResponse:
        logger.error("Persistence error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="PERSISTENCE_ERROR",
        )

    # --------------------------------------------------------------------------
    # Document Processing & Background Exceptions (Phase 15)
    # --------------------------------------------------------------------------
    from app.services.document_processing.exceptions import (
        DocumentProcessingError,
        PermanentProcessingError,
        TransientProcessingError,
    )

    @app.exception_handler(PermanentProcessingError)
    async def permanent_processing_handler(request: Request, exc: PermanentProcessingError) -> JSONResponse:
        logger.error("Permanent document processing error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.message,
            error_type="PERMANENT_PROCESSING_ERROR",
            details=exc.details,
        )

    @app.exception_handler(TransientProcessingError)
    async def transient_processing_handler(request: Request, exc: TransientProcessingError) -> JSONResponse:
        logger.error("Transient document processing error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=exc.message,
            error_type="TRANSIENT_PROCESSING_ERROR",
            details=exc.details,
        )

    @app.exception_handler(DocumentProcessingError)
    async def document_processing_handler(request: Request, exc: DocumentProcessingError) -> JSONResponse:
        logger.error("Document processing error on %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=exc.message,
            error_type="DOCUMENT_PROCESSING_ERROR",
            details=exc.details,
        )

    # --------------------------------------------------------------------------
    # Fallback Catch-All Unhandled Exception Handler
    # --------------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected server errors without exposing internal traces to clients."""
        logger.exception("Unhandled server error processing request to %s: %s", request.url.path, exc)
        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="An internal server error occurred.",
            error_type="INTERNAL_SERVER_ERROR",
        )

    app.add_exception_handler(500, unhandled_exception_handler)
