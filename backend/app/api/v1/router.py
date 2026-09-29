"""Central router for API version 1."""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    chunks,
    documents,
    embeddings,
    generation,
    grounded_answer,
    health,
    qa,
    query_plan,
    retrieval,
    search,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chunks.router)
api_router.include_router(embeddings.router)
api_router.include_router(retrieval.router)
api_router.include_router(search.router)
api_router.include_router(qa.router)
api_router.include_router(generation.router)
api_router.include_router(grounded_answer.router)
api_router.include_router(query_plan.router)






