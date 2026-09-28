"""Development and testing endpoints for semantic vector retrieval with FAISS."""

from fastapi import APIRouter, status

from app.schemas.embedding import EmbeddingRecord
from app.schemas.retrieval import (
    BuildIndexRequest,
    BuildIndexResponse,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
)
from app.services.retrieval.service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])

# Global development service instance for holding in-memory test indexes
_dev_retrieval_service = RetrievalService()


def get_dev_retrieval_service() -> RetrievalService:
    """Retrieve the shared development RetrievalService instance."""
    return _dev_retrieval_service


@router.post(
    "/index",
    response_model=BuildIndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Build Vector Index (Dev)",
    description="Development endpoint to populate a FAISS IndexFlatIP instance from EmbeddingRecords.",
)
async def build_index(payload: BuildIndexRequest) -> BuildIndexResponse:
    """Build a FAISS vector index from serialized embedding records."""
    records = [EmbeddingRecord(**r) for r in payload.records]
    service = get_dev_retrieval_service()
    idx = service.build_index_from_records(
        records=records,
        index_id=payload.index_id or payload.document_id or "default",
        set_as_default=True,
    )

    return BuildIndexResponse(
        index_id=idx.index_id,
        total_vectors=idx.total_vectors,
        dimension=idx.dimension,
        model_name=idx.model_name,
        document_ids=idx.metadata.document_ids,
    )


@router.post(
    "/search",
    response_model=RetrievalQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Vector Index",
    description="Development endpoint to perform top-K cosine similarity search with optional document isolation.",
)
async def search_index(payload: RetrievalQueryRequest) -> RetrievalQueryResponse:
    """Search the FAISS vector index for top-K candidate chunks."""
    service = get_dev_retrieval_service()
    results = service.search(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
        index_id=payload.index_id,
    )

    return RetrievalQueryResponse(
        query=payload.query,
        top_k=payload.top_k,
        total_results=len(results),
        document_id=payload.document_id,
        results=results,
    )
