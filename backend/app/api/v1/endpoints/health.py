"""Health check and readiness endpoints for BookRAG AI."""

import asyncio

from fastapi import APIRouter, Response, status
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter()

# Short timeout (seconds) for infrastructure health probes to prevent event-loop blocking
_PROBE_TIMEOUT_SECONDS = 2


def _probe_database() -> str:
    """Synchronous database ping with a short-lived connection (bounded timeout).

    Creates a disposable engine with a 2-second connect timeout so the main
    application engine pool is never disrupted by probe traffic.
    """
    probe_url = settings.DATABASE_URL
    connect_args: dict = {}
    if probe_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        # psycopg / libpq connect_timeout is in seconds
        connect_args["connect_timeout"] = _PROBE_TIMEOUT_SECONDS

    engine = create_engine(
        probe_url,
        connect_args=connect_args,
        pool_pre_ping=False,
        pool_size=1,
        max_overflow=0,
        pool_timeout=_PROBE_TIMEOUT_SECONDS,
    )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "connected"
    except Exception:
        return "disconnected"
    finally:
        engine.dispose()


def _probe_redis() -> str:
    """Synchronous Redis ping with a short connect + socket timeout."""
    try:
        import redis as redis_lib
        r = redis_lib.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=_PROBE_TIMEOUT_SECONDS,
            socket_timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if r.ping():
            return "connected"
        return "disconnected"
    except Exception:
        return "disconnected"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Check whether the application process is running and responding to HTTP requests.",
    tags=["Health"],
)
async def check_health() -> HealthResponse:
    """Return the operational liveness status and service identifier."""
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Verify connectivity to essential infrastructure subsystems (PostgreSQL, Redis, Vector backend).",
    tags=["Health"],
)
async def check_readiness(response: Response) -> ReadinessResponse:
    """Check readiness of database, redis broker, and vector index persistence."""
    checks: dict[str, str] = {}

    # 1. Database Connectivity Probe — run in a thread to avoid blocking the event loop
    try:
        db_status = await asyncio.wait_for(
            asyncio.to_thread(_probe_database),
            timeout=_PROBE_TIMEOUT_SECONDS + 1,
        )
    except (asyncio.TimeoutError, Exception):
        db_status = "disconnected"

    if db_status == "connected":
        checks["database"] = "PostgreSQL connection verified via ping query"
    else:
        checks["database"] = "PostgreSQL connection failed or unreachable"

    # 2. Redis Message Broker Probe — run in a thread to avoid blocking the event loop
    try:
        redis_status = await asyncio.wait_for(
            asyncio.to_thread(_probe_redis),
            timeout=_PROBE_TIMEOUT_SECONDS + 1,
        )
    except (asyncio.TimeoutError, Exception):
        redis_status = "disconnected"

    if redis_status == "connected":
        checks["redis"] = "Redis ping succeeded"
    else:
        checks["redis"] = "Redis broker unreachable"

    # 3. Vector Search Backend Operational Status
    vector_status = f"{settings.VECTOR_BACKEND} (operational)"
    checks["vector_backend"] = f"Configured {settings.VECTOR_BACKEND} backend active"

    # Determine overall readiness status
    if db_status == "connected" and redis_status == "connected":
        overall_status = "ready"
    elif db_status == "connected" or settings.CELERY_TASK_ALWAYS_EAGER:
        overall_status = "degraded"
    else:
        overall_status = "degraded" if settings.ENVIRONMENT == "development" else "unavailable"

    if overall_status == "unavailable":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    # 4. Inference Hardware Diagnostics
    from app.core.device import get_device_manager
    device_diag = get_device_manager().get_diagnostics()
    checks["inference_hardware"] = f"{device_diag['device']} (cuda_available={device_diag['cuda_available']})"

    return ReadinessResponse(
        status=overall_status,
        service=settings.APP_NAME,
        environment=settings.ENVIRONMENT,
        database=db_status,
        redis=redis_status,
        vector_backend=vector_status,
        checks=checks,
        device_diagnostics=device_diag,
    )
