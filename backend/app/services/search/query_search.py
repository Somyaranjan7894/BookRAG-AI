"""Query-Aware Search Service for Phase 11 Multi-Query Retrieval and Merging.

Implements query-plan-driven retrieval orchestration:
1. Executes individual retrieval queries from the QueryPlan.
2. Merges multi-query candidate pools with deterministic deduplication by (document_id, chunk_id).
3. Executes final cross-encoder reranking across the unified deduplicated candidate pool.
4. Preserves full evidence provenance and document isolation.
"""

from typing import Dict, List, Optional, Tuple

from app.core.logging import get_logger
from app.schemas.query_plan import QueryPlan
from app.schemas.search import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    SearchResponse,
    SearchResult,
)
from app.services.reranking.service import RerankerService
from app.services.search.service import SearchService

logger = get_logger(__name__)


class QuerySearchService:
    """Orchestrates single- and multi-query retrieval driven by QueryPlan specifications."""

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        reranker_service: Optional[RerankerService] = None,
    ) -> None:
        """Initialize QuerySearchService with injected or lazily resolved components."""
        self._search_service = search_service
        self._reranker_service = reranker_service

    @property
    def search_service(self) -> SearchService:
        """Return injected or default SearchService."""
        if self._search_service is None:
            self._search_service = SearchService()
        return self._search_service

    @property
    def reranker_service(self) -> RerankerService:
        """Return injected or default RerankerService."""
        if self._reranker_service is None:
            if hasattr(self.search_service, "reranker_service") and self.search_service.reranker_service:
                self._reranker_service = self.search_service.reranker_service
            else:
                self._reranker_service = RerankerService()
        return self._reranker_service

    def search_with_plan(
        self,
        plan: QueryPlan,
        top_k: int = DEFAULT_TOP_K,
        document_id: Optional[str] = None,
        candidate_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None,
    ) -> SearchResponse:
        """Execute retrieval governed by a QueryPlan.

        For single-query plans:
            Delegates directly to SearchService.
        For multi-query plans:
            1. Executes each sub-query independently (retrieving candidate pools).
            2. Merges candidate pools deduplicating by (document_id, chunk_id).
            3. Reranks the merged candidate pool against plan.normalized_query.
            4. Re-assigns final ranks and returns SearchResponse.
        """
        queries = plan.retrieval_queries or [plan.normalized_query]

        # Case 1: Single query plan -> Standard execution
        if len(queries) <= 1:
            target_query = queries[0] if queries else plan.normalized_query
            return self.search_service.search(
                query=target_query,
                top_k=top_k,
                document_id=document_id,
                candidate_k=candidate_k,
                enable_reranking=enable_reranking,
            )

        # Case 2: Multi-query plan -> Independent retrieval + Merge + Deduplicate + Final Rerank
        logger.info(
            "Executing multi-query retrieval for %d queries: %s",
            len(queries),
            queries,
        )

        default_k = getattr(self.search_service, "default_candidate_k", DEFAULT_CANDIDATE_K)
        if not isinstance(default_k, int):
            default_k = DEFAULT_CANDIDATE_K
        effective_candidate_k = candidate_k or max(default_k, top_k * 2)

        merged_candidates: Dict[Tuple[str, str], SearchResult] = {}

        for sub_q in queries:
            try:
                sub_resp = self.search_service.search(
                    query=sub_q,
                    top_k=effective_candidate_k,
                    document_id=document_id,
                    candidate_k=effective_candidate_k,
                    enable_reranking=False,  # Defer reranking until after merge
                )
            except Exception as exc:
                logger.warning("Sub-query '%s' retrieval failed: %s. Continuing with other queries.", sub_q, exc)
                continue

            for res in sub_resp.results:
                key = (res.document_id, res.chunk_id)
                if key not in merged_candidates:
                    metadata = dict(res.metadata or {})
                    metadata["matched_queries"] = [sub_q]
                    merged_candidates[key] = res.model_copy(update={"metadata": metadata})
                else:
                    # Duplicate chunk across multiple queries: retain one, update matched_queries
                    existing = merged_candidates[key]
                    existing_meta = dict(existing.metadata or {})
                    matched = list(existing_meta.get("matched_queries", []))
                    if sub_q not in matched:
                        matched.append(sub_q)
                    existing_meta["matched_queries"] = matched

                    # Keep higher similarity score if different
                    if res.similarity_score and (
                        existing.similarity_score is None or res.similarity_score > existing.similarity_score
                    ):
                        merged_candidates[key] = res.model_copy(
                            update={"similarity_score": res.similarity_score, "metadata": existing_meta}
                        )
                    else:
                        merged_candidates[key] = existing.model_copy(update={"metadata": existing_meta})

        candidate_pool = list(merged_candidates.values())

        if not candidate_pool:
            logger.info("Multi-query retrieval yielded zero candidates across all queries.")
            return SearchResponse(
                query=plan.normalized_query,
                results=[],
                total_results=0,
                document_id=document_id,
                reranking_applied=False,
                candidate_count=0,
            )

        # Check whether reranking should be applied
        apply_rerank = (
            enable_reranking
            if enable_reranking is not None
            else self.search_service.is_reranking_enabled
        )

        final_results: List[SearchResult] = []
        reranking_applied = False

        if apply_rerank:
            try:
                final_results = self.reranker_service.rerank(
                    query=plan.normalized_query,
                    candidates=candidate_pool,
                    top_n=top_k,
                )
                reranking_applied = True
            except Exception as exc:
                logger.warning("Final reranking failed on merged candidates: %s. Falling back to similarity ordering.", exc)
                candidate_pool.sort(key=lambda x: (x.similarity_score or 0.0), reverse=True)
                for idx, item in enumerate(candidate_pool[:top_k]):
                    final_results.append(item.model_copy(update={"rank": idx + 1}))
                reranking_applied = False
        else:
            candidate_pool.sort(key=lambda x: (x.similarity_score or 0.0), reverse=True)
            for idx, item in enumerate(candidate_pool[:top_k]):
                final_results.append(item.model_copy(update={"rank": idx + 1}))
            reranking_applied = False

        logger.info(
            "Multi-query search merged %d unique candidates down to top_%d results (reranked=%s).",
            len(candidate_pool),
            len(final_results),
            reranking_applied,
        )

        return SearchResponse(
            query=plan.normalized_query,
            results=final_results,
            total_results=len(final_results),
            document_id=document_id,
            reranking_applied=reranking_applied,
            candidate_count=len(candidate_pool),
        )
