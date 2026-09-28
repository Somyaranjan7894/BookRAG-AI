"""Application entry point for BookRAG AI backend service.

Initializes the FastAPI application, configures logging, registers API routers,
and sets up global exception handling.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import setup_exception_handlers
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events."""
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "Starting %s v%s (Environment: %s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


def create_application() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Register error handlers
    setup_exception_handlers(application)

    # Register API versioned routers
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
