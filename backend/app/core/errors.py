"""Application error handling configuration for BookRAG AI.

Ensures structured error responses without leaking internal stack traces to clients.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Register core exception handlers with the FastAPI application."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle standard HTTP exceptions with structured response."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle schema validation errors safely."""
        from fastapi.encoders import jsonable_encoder

        clean_errors = []
        for err in exc.errors():
            item = dict(err)
            if "ctx" in item and isinstance(item["ctx"], dict):
                item["ctx"] = {k: str(v) for k, v in item["ctx"].items()}
            clean_errors.append(item)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "message": "Request validation error",
                    "details": jsonable_encoder(clean_errors),
                }
            },
        )

    # Late import to avoid circular dependency
    from app.services.pdf.exceptions import (
        InvalidPDFError,
        PDFIngestionError,
        PDFNotFoundError,
    )

    @app.exception_handler(PDFNotFoundError)
    async def pdf_not_found_handler(request: Request, exc: PDFNotFoundError) -> JSONResponse:
        """Handle missing PDF file errors."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "code": status.HTTP_404_NOT_FOUND,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(InvalidPDFError)
    async def invalid_pdf_handler(request: Request, exc: InvalidPDFError) -> JSONResponse:
        """Handle malformed or corrupt PDF errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(PDFIngestionError)
    async def pdf_ingestion_handler(request: Request, exc: PDFIngestionError) -> JSONResponse:
        """Handle unexpected PDF ingestion failures."""
        logger.error("PDF ingestion error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    from app.services.embeddings.exceptions import (
        EmbeddingError,
        InvalidChunkError,
        ModelLoadError,
    )

    @app.exception_handler(InvalidChunkError)
    async def invalid_chunk_handler(request: Request, exc: InvalidChunkError) -> JSONResponse:
        """Handle malformed or empty chunk embedding errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
        """Handle model loading failures."""
        logger.error("Embedding model load error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
        """Handle general embedding failures."""
        logger.error("Embedding service error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    from app.services.retrieval.exceptions import (
        CorruptedIndexError,
        EmptyIndexError,
        IndexDimensionMismatchError,
        IndexNotFoundError,
        IndexPersistenceError,
        InvalidQueryError,
        RetrievalError,
    )

    @app.exception_handler(InvalidQueryError)
    async def invalid_query_handler(request: Request, exc: InvalidQueryError) -> JSONResponse:
        """Handle empty or invalid query strings."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(IndexDimensionMismatchError)
    async def dimension_mismatch_handler(request: Request, exc: IndexDimensionMismatchError) -> JSONResponse:
        """Handle vector dimension mismatches."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(IndexNotFoundError)
    async def index_not_found_handler(request: Request, exc: IndexNotFoundError) -> JSONResponse:
        """Handle missing index requests."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "code": status.HTTP_404_NOT_FOUND,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(CorruptedIndexError)
    async def corrupted_index_handler(request: Request, exc: CorruptedIndexError) -> JSONResponse:
        """Handle corrupted index or metadata files."""
        logger.error("Corrupted index detected on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(IndexPersistenceError)
    async def index_persistence_handler(request: Request, exc: IndexPersistenceError) -> JSONResponse:
        """Handle index save/load failures."""
        logger.error("Index persistence error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
        """Handle general retrieval failures."""
        logger.error("Retrieval error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    from app.services.search.exceptions import (
        DocumentNotFoundError,
        IndexNotInitializedError,
        InvalidSearchQueryError,
        InvalidTopKError,
        SearchEmbeddingError,
        SearchError,
        SearchRetrievalError,
    )

    @app.exception_handler(InvalidSearchQueryError)
    async def invalid_search_query_handler(request: Request, exc: InvalidSearchQueryError) -> JSONResponse:
        """Handle invalid or empty search query strings."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(InvalidTopKError)
    async def invalid_top_k_handler(request: Request, exc: InvalidTopKError) -> JSONResponse:
        """Handle invalid top_k parameter values."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(request: Request, exc: DocumentNotFoundError) -> JSONResponse:
        """Handle unknown document ID requests in search."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "code": status.HTTP_404_NOT_FOUND,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(IndexNotInitializedError)
    async def index_not_initialized_handler(request: Request, exc: IndexNotInitializedError) -> JSONResponse:
        """Handle uninitialized or missing search index."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "code": status.HTTP_404_NOT_FOUND,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(SearchEmbeddingError)
    async def search_embedding_error_handler(request: Request, exc: SearchEmbeddingError) -> JSONResponse:
        """Handle search query embedding failures."""
        logger.error("Search embedding error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(SearchRetrievalError)
    async def search_retrieval_error_handler(request: Request, exc: SearchRetrievalError) -> JSONResponse:
        """Handle search vector retrieval execution failures."""
        logger.error("Search retrieval error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(SearchError)
    async def general_search_error_handler(request: Request, exc: SearchError) -> JSONResponse:
        """Handle general search domain errors."""
        logger.error("General search service error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    from app.services.reranking.exceptions import (
        InvalidRerankingConfigError,
        RerankerModelLoadError,
        RerankingError,
        RerankingExecutionError,
        ScoreAlignmentError,
    )

    @app.exception_handler(InvalidRerankingConfigError)
    async def invalid_reranking_config_handler(request: Request, exc: InvalidRerankingConfigError) -> JSONResponse:
        """Handle invalid reranking parameters or query configuration."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(RerankerModelLoadError)
    async def reranker_model_load_handler(request: Request, exc: RerankerModelLoadError) -> JSONResponse:
        """Handle CrossEncoder model loading failures."""
        logger.error("Reranker model load error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(ScoreAlignmentError)
    async def score_alignment_error_handler(request: Request, exc: ScoreAlignmentError) -> JSONResponse:
        """Handle score-to-candidate alignment mismatch errors."""
        logger.error("Score alignment error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(RerankingExecutionError)
    async def reranking_execution_handler(request: Request, exc: RerankingExecutionError) -> JSONResponse:
        """Handle cross-encoder inference and forward-pass scoring failures."""
        logger.error("Reranking execution error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(RerankingError)
    async def general_reranking_handler(request: Request, exc: RerankingError) -> JSONResponse:
        """Handle general cross-encoder reranking domain failures."""
        logger.error("General reranking error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

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
        """Handle invalid or empty question errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(InvalidQAEvidenceError)
    async def invalid_qa_evidence_handler(request: Request, exc: InvalidQAEvidenceError) -> JSONResponse:
        """Handle malformed QA evidence candidates."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(InvalidAnswerSpanError)
    async def invalid_answer_span_handler(request: Request, exc: InvalidAnswerSpanError) -> JSONResponse:
        """Handle invalid answer span constraints."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(InvalidQAConfigError)
    async def invalid_qa_config_handler(request: Request, exc: InvalidQAConfigError) -> JSONResponse:
        """Handle invalid QA configuration errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(QAModelLoadError)
    async def qa_model_load_handler(request: Request, exc: QAModelLoadError) -> JSONResponse:
        """Handle QA model loading failures."""
        logger.error("QA model load error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": status.HTTP_503_SERVICE_UNAVAILABLE,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(QAInferenceError)
    async def qa_inference_handler(request: Request, exc: QAInferenceError) -> JSONResponse:
        """Handle QA inference or sliding window execution errors."""
        logger.error("QA inference error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(QAError)
    async def general_qa_error_handler(request: Request, exc: QAError) -> JSONResponse:
        """Handle general Question Answering domain errors."""
        logger.error("General QA error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

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
        """Handle invalid or empty generation query errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(InvalidGenerationEvidenceError)
    async def invalid_generation_evidence_handler(request: Request, exc: InvalidGenerationEvidenceError) -> JSONResponse:
        """Handle malformed generation evidence items."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(InvalidGenerationConfigError)
    async def invalid_generation_config_handler(request: Request, exc: InvalidGenerationConfigError) -> JSONResponse:
        """Handle invalid generation configuration parameters."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(ContextBudgetExceededError)
    async def context_budget_exceeded_handler(request: Request, exc: ContextBudgetExceededError) -> JSONResponse:
        """Handle prompt context budget exhaustion errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(GenerationModelLoadError)
    async def generation_model_load_handler(request: Request, exc: GenerationModelLoadError) -> JSONResponse:
        """Handle Seq2Seq generation model loading failures."""
        logger.error("Generation model load error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": status.HTTP_503_SERVICE_UNAVAILABLE,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(GenerationInferenceError)
    async def generation_inference_handler(request: Request, exc: GenerationInferenceError) -> JSONResponse:
        """Handle FLAN-T5 generation forward pass or beam search failures."""
        logger.error("Generation inference error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(GenerationError)
    async def general_generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
        """Handle general abstractive generation domain errors."""
        logger.error("General generation error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    # Late import of Grounding exceptions
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
        """Handle malformed or empty grounding input errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(InvalidGroundingConfigError)
    async def invalid_grounding_config_handler(request: Request, exc: InvalidGroundingConfigError) -> JSONResponse:
        """Handle out-of-bounds grounding configuration thresholds."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(NLIModelLoadError)
    async def nli_model_load_handler(request: Request, exc: NLIModelLoadError) -> JSONResponse:
        """Handle CrossEncoder NLI model loading failures."""
        logger.error("NLI model load error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": status.HTTP_503_SERVICE_UNAVAILABLE,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(NLILabelMappingError)
    async def nli_label_mapping_handler(request: Request, exc: NLILabelMappingError) -> JSONResponse:
        """Handle NLI label discovery failures."""
        logger.error("NLI label mapping error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(NLIInferenceError)
    async def nli_inference_handler(request: Request, exc: NLIInferenceError) -> JSONResponse:
        """Handle NLI forward pass or batched inference failures."""
        logger.error("NLI inference error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(GroundingError)
    async def general_grounding_error_handler(request: Request, exc: GroundingError) -> JSONResponse:
        """Handle general grounding domain errors."""
        logger.error("General grounding error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected server errors without exposing internal traces to clients."""
        logger.exception("Unhandled server error processing request to %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": "An internal server error occurred.",
                }
            },
        )
