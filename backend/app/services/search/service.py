"""Application-level Semantic Search Service for BookRAG AI.

Orchestrates query validation, query embedding via EmbeddingService, FAISS vector
retrieval via RetrievalService, optional Cross-Encoder precision reranking via
RerankerService, result provenance mapping, and document isolation.
"""

import math
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.search import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    SearchResponse,
    SearchResult,
)
from app.services.embeddings.service import EmbeddingService
from app.services.reranking.exceptions import RerankingError
from app.services.reranking.service import RerankerService
from app.services.retrieval.backend import (
    FAISSVectorBackend,
    PGVectorBackend,
    VectorSearchBackend,
    create_vector_backend,
)
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
    """Orchestrates natural language semantic search and two-stage precision reranking."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        reranker_service: Optional[RerankerService] = None,
        vector_backend: Optional[VectorSearchBackend] = None,
        backend_type: Optional[str] = None,
        max_top_k: int = MAX_TOP_K,
        default_candidate_k: int = DEFAULT_CANDIDATE_K,
        enable_reranking: Optional[bool] = None,
    ) -> None:
        """Initialize SearchService with injectable dependencies.

        Args:
            embedding_service: Phase 3 service generating dense vector embeddings.
            retrieval_service: Phase 4 service managing FAISS indexes and candidate retrieval.
            reranker_service: Phase 6 service performing Cross-Encoder precision reranking.
            vector_backend: Phase 14 pluggable VectorSearchBackend (FAISS or pgvector).
            backend_type: Optional backend type override ("faiss" or "pgvector").
            max_top_k: Maximum allowed candidate limit (bounds computational overhead).
            default_candidate_k: Default first-stage candidate pool size before reranking.
            enable_reranking: Whether reranking is active by default (None = auto-detect).
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.retrieval_service = retrieval_service or RetrievalService(embedding_service=self.embedding_service)
        self.reranker_service = reranker_service
        self.max_top_k = max_top_k
        self.default_candidate_k = default_candidate_k
        self._enable_reranking_override = enable_reranking

        # Resolve vector search backend (FAISS or pgvector)
        if vector_backend is not None:
            self.backend = vector_backend
        elif retrieval_service is not None or (backend_type or settings.VECTOR_BACKEND).lower() == "faiss":
            self.backend = FAISSVectorBackend(retrieval_service=self.retrieval_service)
        else:
            self.backend = create_vector_backend(
                backend_type=backend_type or settings.VECTOR_BACKEND,
                retrieval_service=self.retrieval_service,
            )

    @property
    def is_reranking_enabled(self) -> bool:
        """Determine whether reranking is enabled by default."""
        if self._enable_reranking_override is not None:
            return self._enable_reranking_override
        return bool(settings.RERANKER_ENABLED)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        document_id: Optional[str] = None,
        index_id: Optional[str] = None,
        candidate_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None,
    ) -> SearchResponse:
        """Execute semantic search with complete provenance, document isolation, and optional reranking.

        Two-Stage Execution Pipeline:
        1. Validate query string, top_k bounds, candidate_k bounds, and document_id parameters.
        2. Resolve target VectorIndex and verify document catalog existence.
        3. Handle empty indexes safely without error.
        4. Obtain dense query embedding via Phase 3 EmbeddingService.
        5. First-Stage Retrieval: High-recall candidate pool of size candidate_k from FAISS.
        6. Second-Stage Precision Reranking: Cross-Encoder joint scoring over (query, passage) pairs.
        7. Preserves original_rank, reassigns final rank, attaches similarity_score and reranker_score.
        8. Return structured SearchResponse.

        Args:
            query: Natural language query string (non-empty, non-whitespace).
            top_k: Number of nearest candidates to return (1 <= top_k <= max_top_k).
            document_id: Optional document ID for document-level filtering/isolation.
            index_id: Optional specific index identifier.
            candidate_k: Optional first-stage candidate pool size (must be >= top_k).
            enable_reranking: Optional flag to toggle cross-encoder reranking.

        Returns:
            SearchResponse containing ranked SearchResult candidates and metadata.

        Raises:
            InvalidSearchQueryError: If query is empty or whitespace-only.
            InvalidTopKError: If top_k or candidate_k is non-positive or exceeds bounds.
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

        if candidate_k is not None:
            if not isinstance(candidate_k, int) or candidate_k < top_k:
                raise InvalidTopKError(
                    f"candidate_k ({candidate_k}) must be an integer greater than or equal to top_k ({top_k})."
                )

        clean_doc_id: Optional[str] = None
        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise InvalidSearchQueryError("document_id cannot be whitespace-only if provided.")
            clean_doc_id = document_id.strip()

        # 2. Check empty index / vector store state
        try:
            total_vecs = self.backend.total_vectors(index_id=index_id)
        except IndexNotFoundError as exc:
            logger.warning("Search failed: no vector index found (index_id=%s).", index_id)
            raise IndexNotInitializedError(
                f"Search index is not initialized or not found: {exc.message}"
            ) from exc

        # Check empty index state
        if total_vecs == 0:
            if clean_doc_id is not None:
                raise DocumentNotFoundError(
                    f"Document '{clean_doc_id}' was not found in the search index."
                )
            logger.debug("Search executed on empty vector backend. Returning empty results.")
            return SearchResponse(
                query=clean_query,
                results=[],
                total_results=0,
                document_id=clean_doc_id,
                reranking_applied=False,
            )

        # Check document isolation validity
        if clean_doc_id is not None:
            if not self.backend.has_document(clean_doc_id, index_id=index_id):
                logger.info("Requested document '%s' does not exist in search index.", clean_doc_id)
                raise DocumentNotFoundError(
                    f"Document '{clean_doc_id}' was not found in the search index."
                )

        # 3. Determine reranking activation
        apply_rerank: bool
        if enable_reranking is not None:
            apply_rerank = bool(enable_reranking)
        elif self._enable_reranking_override is not None:
            apply_rerank = bool(self._enable_reranking_override)
        else:
            apply_rerank = (self.reranker_service is not None) and bool(settings.RERANKER_ENABLED)

        # 4. Resolve candidate pool size for first-stage retrieval
        effective_candidate_k = candidate_k or (
            max(top_k, self.default_candidate_k) if apply_rerank else top_k
        )

        # 5. Query embedding generation via Phase 3 EmbeddingService
        try:
            query_vector = self.embedding_service.embed_query(clean_query)
        except Exception as exc:
            logger.exception("Query embedding generation failed: %s", exc)
            raise SearchEmbeddingError(
                "Failed to generate query embedding.",
                details=str(exc),
            ) from exc

        # 6. First-Stage Vector Retrieval via VectorSearchBackend (High-Recall Candidate Pool)
        try:
            raw_results = self.backend.search(
                query_vector=query_vector,
                top_k=effective_candidate_k,
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

        # 7. Second-Stage Precision Reranking (Cross-Encoder) or Direct Mapping
        search_results: List[SearchResult] = []
        reranking_applied: bool = False

        if apply_rerank and raw_results:
            if self.reranker_service is None:
                self.reranker_service = RerankerService()
            try:
                search_results = self.reranker_service.rerank(
                    query=clean_query,
                    candidates=raw_results,
                    top_n=top_k,
                )
                reranking_applied = True
            except RerankingError:
                raise
            except Exception as exc:
                logger.exception("Reranking execution failed: %s", exc)
                raise SearchRetrievalError(
                    "Reranking execution failed.",
                    details=str(exc),
                ) from exc
        else:
            # Direct mapping without reranking
            candidates_to_map = raw_results[:top_k] if candidate_k is not None else raw_results
            for res in candidates_to_map:
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
                        original_rank=getattr(res, "rank", len(search_results) + 1),
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        page_number=page_no,
                        text=text,
                        similarity_score=float(score),
                        reranker_score=None,
                        chunk_index=getattr(res, "chunk_index", 0),
                        metadata=getattr(res, "metadata", {}),
                    )
                )
            reranking_applied = False

        logger.info(
            "Search for query '%s' completed: %d results (top_k=%d, candidate_k=%d, reranked=%s, doc_filter=%s).",
            clean_query[:40],
            len(search_results),
            top_k,
            effective_candidate_k,
            reranking_applied,
            clean_doc_id,
        )

        return SearchResponse(
            query=clean_query,
            results=search_results,
            total_results=len(search_results),
            document_id=clean_doc_id,
            reranking_applied=reranking_applied,
        )
