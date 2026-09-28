"""Central router for API version 1."""

from fastapi import APIRouter

from app.api.v1.endpoints import chunks, documents, embeddings, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chunks.router)
api_router.include_router(embeddings.router)



