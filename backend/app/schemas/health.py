from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness probe response schema."""

    status: str = Field(default="ok", description="Operational status of the service")
    service: str = Field(default="BookRAG AI", description="Name of the service")


class ReadinessResponse(BaseModel):
    """Readiness probe response schema verifying core subsystem availability."""

    status: str = Field(description="Readiness status ('ready', 'degraded', 'unavailable')")
    service: str = Field(default="BookRAG AI", description="Name of the service")
    environment: str = Field(description="Operational deployment environment")
    database: str = Field(description="PostgreSQL persistence layer status")
    redis: str = Field(description="Redis asynchronous broker status")
    vector_backend: str = Field(description="Vector search backend operational status")
    checks: Dict[str, str] = Field(default_factory=dict, description="Detailed component readiness indicators")
    device_diagnostics: Optional[Dict[str, Any]] = Field(default=None, description="Inference hardware and CUDA telemetry")
