"""Application-level Semantic Search Service for BookRAG AI.

Orchestrates query validation, query embedding via EmbeddingService, FAISS vector
retrieval via RetrievalService, result provenance mapping, and document isolation.
"""

import math
from typing import List, Optional

from app.core.logging import get_logger
from app.schemas.search import DEFAULT_TOP_K, MAX_TOP_K, SearchResponse, SearchResult
from app.services.embeddings.service import EmbeddingService
from app.services.retrieval.exceptions import IndexNotFoundError
from app.services.retrieval.service import RetrievalService
from app.services.search.exceptions import (
    DocumentNotFoundError,
    IndexNotInitializedError,
    InvalidSearchQueryError,
    InvalidTopKError,
    SearchEmbeddingError,
    SearchRetrievalError,
)

logger = get_logger(__name__)


class SearchService:
    """Orchestrates natural language semantic search across indexed document chunks."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        max_top_k: int = MAX_TOP_K,
    ) -> None:
        """Initialize SearchService with injectable dependencies.

        Args:
            embedding_service: Phase 3 service generating dense vector embeddings.
            retrieval_service: Phase 4 service managing FAISS indexes and candidate retrieval.
            max_top_k: Maximum allowed candidate limit (bounds computational overhead).
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.retrieval_service = retrieval_service or RetrievalService(embedding_service=self.embedding_service)
        self.max_top_k = max_top_k

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        document_id: Optional[str] = None,
        index_id: Optional[str] = None,
    ) -> SearchResponse:
        """Execute semantic search with complete provenance and document isolation.

        Execution Pipeline:
        1. Validate query string, top_k bounds, and document_id parameters.
        2. Resolve target VectorIndex and verify document catalog existence.
        3. Handle empty indexes safely without error.
        4. Obtain dense query embedding via Phase 3 EmbeddingService.
        5. Perform vector retrieval via Phase 4 RetrievalService.
        6. Map candidates to application-level SearchResult objects preserving provenance.
        7. Return structured SearchResponse.

        Args:
            query: Natural language query string (non-empty, non-whitespace).
            top_k: Number of nearest candidates to retrieve (1 <= top_k <= max_top_k).
            document_id: Optional document ID for document-level filtering/isolation.
            index_id: Optional specific index identifier.

        Returns:
            SearchResponse containing ranked SearchResult candidates and metadata.

        Raises:
            InvalidSearchQueryError: If query is empty or whitespace-only.
            InvalidTopKError: If top_k is non-positive or exceeds max_top_k.
            DocumentNotFoundError: If requested document_id is not present in index catalog.
            IndexNotInitializedError: If no search index is available.
            SearchEmbeddingError: If query embedding generation fails.
            SearchRetrievalError: If vector retrieval fails or returns malformed results.
        """
        # 1. Parameter validation
        if query is None or not isinstance(query, str) or not query.strip():
            raise InvalidSearchQueryError("Search query cannot be empty or whitespace-only.")

        clean_query = query.strip()

        if not isinstance(top_k, int) or top_k <= 0 or top_k > self.max_top_k:
            raise InvalidTopKError(
                f"top_k must be an integer between 1 and {self.max_top_k} (got {top_k})."
            )

        clean_doc_id: Optional[str] = None
        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise InvalidSearchQueryError("document_id cannot be whitespace-only if provided.")
            clean_doc_id = document_id.strip()

        # 2. Resolve index and verify document existence
        try:
            target_index = self.retrieval_service.get_index(index_id)
        except IndexNotFoundError as exc:
            logger.warning("Search failed: no vector index found (index_id=%s).", index_id)
            raise IndexNotInitializedError(
                f"Search index is not initialized or not found: {exc.message}"
            ) from exc

        # Check empty index state
        if target_index.total_vectors == 0:
            if clean_doc_id is not None:
                raise DocumentNotFoundError(
                    f"Document '{clean_doc_id}' was not found in the search index."
                )
            logger.debug("Search executed on empty index '%s'. Returning empty results.", target_index.index_id)
            return SearchResponse(
                query=clean_query,
                results=[],
                total_results=0,
                document_id=clean_doc_id,
            )

        # Check document isolation validity
        if clean_doc_id is not None:
            known_docs = set(target_index.metadata.document_ids)
            has_doc_index = (
                hasattr(self.retrieval_service, "_indices")
                and clean_doc_id in self.retrieval_service._indices
            )
            if clean_doc_id not in known_docs and not has_doc_index:
                logger.info("Requested document '%s' does not exist in index catalog.", clean_doc_id)
                raise DocumentNotFoundError(
                    f"Document '{clean_doc_id}' was not found in the search index."
                )

        # 3. Query embedding generation via Phase 3 EmbeddingService
        try:
            query_vector = self.embedding_service.embed_query(clean_query)
        except Exception as exc:
            logger.exception("Query embedding generation failed: %s", exc)
            raise SearchEmbeddingError(
                "Failed to generate query embedding.",
                details=str(exc),
            ) from exc

        # 4. Vector retrieval via Phase 4 Retrieval component
        try:
            if hasattr(self.retrieval_service, "search_by_vector"):
                raw_results = self.retrieval_service.search_by_vector(
                    query_vector=query_vector,
                    top_k=top_k,
                    document_id=clean_doc_id,
                    index_id=index_id,
                )
            else:
                raw_results = self.retrieval_service.search(
                    query=clean_query,
                    top_k=top_k,
                    document_id=clean_doc_id,
                    index_id=index_id,
                )
        except (IndexNotFoundError, DocumentNotFoundError):
            raise
        except Exception as exc:
            logger.exception("Vector retrieval execution failed: %s", exc)
            raise SearchRetrievalError(
                "Vector retrieval execution failed.",
                details=str(exc),
            ) from exc

        # 5. Result mapping and provenance validation
        search_results: List[SearchResult] = []
        for res in raw_results:
            # Validate retrieval result integrity
            score = getattr(res, "similarity_score", None)
            chunk_id = getattr(res, "chunk_id", None)
            doc_id = getattr(res, "document_id", None)
            page_no = getattr(res, "page_number", None)
            text = getattr(res, "text", None)

            if (
                score is None
                or math.isnan(score)
                or not chunk_id
                or not doc_id
                or page_no is None
                or text is None
            ):
                raise SearchRetrievalError(
                    "Malformed retrieval result detected: missing required provenance fields or invalid score."
                )

            search_results.append(
                SearchResult(
                    rank=len(search_results) + 1,
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    page_number=page_no,
                    text=text,
                    similarity_score=float(score),
                    chunk_index=getattr(res, "chunk_index", 0),
                    metadata=getattr(res, "metadata", {}),
                )
            )

        logger.info(
            "Search for query '%s' completed: %d results (top_k=%d, doc_filter=%s).",
            clean_query[:40],
            len(search_results),
            top_k,
            clean_doc_id,
        )

        return SearchResponse(
            query=clean_query,
            results=search_results,
            total_results=len(search_results),
            document_id=clean_doc_id,
        )
