"""Health check endpoint for BookRAG AI."""

from fastapi import APIRouter

from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check operational status and service information.",
    tags=["Health"],
)
async def check_health() -> HealthResponse:
    """Return the operational health status and service identifier."""
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
    )
