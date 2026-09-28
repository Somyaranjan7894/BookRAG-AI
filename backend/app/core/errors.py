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
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "message": "Request validation error",
                    "details": exc.errors(),
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
