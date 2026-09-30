"""Application entry point and composition root for BookRAG AI backend service.

Configures application lifecycle, correlation middleware, global error handlers,
dependency injection routers, and OpenAPI metadata.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.correlation import CorrelationIdMiddleware
from app.core.errors import setup_exception_handlers
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)

OPENAPI_TAGS: List[Dict[str, Any]] = [
    {
        "name": "Health",
        "description": "System health verification and service discovery endpoints.",
    },
    {
        "name": "Documents",
        "description": "Asynchronous document upload, queued status tracking, and PostgreSQL persistence management.",
    },
    {
        "name": "Search",
        "description": "Semantic dense vector retrieval with optional Cross-Encoder precision reranking.",
    },
    {
        "name": "Question Answering",
        "description": "Extractive question answering with RoBERTa SQuAD2 and evidence provenance.",
    },
    {
        "name": "Answer Generation",
        "description": "Abstractive grounded answer generation using FLAN-T5 with evidence budgeting.",
    },
    {
        "name": "Grounded Answer",
        "description": "End-to-end grounded RAG with sentence-level NLI hallucination validation and book citations.",
    },
    {
        "name": "Question Generation",
        "description": "Autonomous reading comprehension question generation and validation with book provenance.",
    },
    {
        "name": "Query Planning",
        "description": "Query intent classification, constraint extraction, and multi-hop retrieval planning.",
    },
    {
        "name": "Chunks",
        "description": "Development and inspection endpoints for text cleaning and semantic chunking.",
    },
    {
        "name": "Embeddings",
        "description": "Development endpoints for dense vector embedding generation and inspection.",
    },
    {
        "name": "Retrieval",
        "description": "Development endpoints for vector index construction and search.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events."""
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "Starting %s v%s (Environment: %s, Vector Backend: %s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
        settings.VECTOR_BACKEND,
    )
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


def create_application() -> FastAPI:
    """Create and configure the FastAPI application composition root."""
    application = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=OPENAPI_TAGS,
    )

    # Register pure ASGI correlation middleware first for full request lifecycle coverage
    application.add_middleware(CorrelationIdMiddleware)

    # Register centralized exception handlers
    setup_exception_handlers(application)

    # Register API v1 versioned routers
    application.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return application


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
