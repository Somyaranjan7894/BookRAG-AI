"""Pydantic schema for service health status."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="ok", description="Operational status of the service")
    service: str = Field(default="BookRAG AI", description="Name of the service")
