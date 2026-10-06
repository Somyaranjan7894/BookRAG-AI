"""Comparative Evaluation for Stage 1 Vector Retrieval vs Stage 2 Cross-Encoder Reranking.

Measures Recall@K, Precision@K, and MRR before and after reranking,
computing deterministic deltas to observe ranking movement and precision shifts.
Preserves strict scientific reporting without assuming reranking always improves metrics.
"""

from typing import Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, Field

from evaluation.retrieval.metrics import (
    DEFAULT_K_VALUES,
    AggregatedRetrievalMetrics,
    QueryRetrievalResult,
    aggregate_retrieval_metrics,
    evaluate_query_retrieval,
)


class RerankingComparisonReport(BaseModel):
    """Comparative metrics container contrasting Stage 1 (Vector) vs Stage 2 (Reranked)."""

    dataset_name: str
    total_queries: int
    evaluated_queries: int
    unanswerable_queries: int

    # Stage 1 Metrics (Vector / Bi-Encoder)
    stage1_metrics: AggregatedRetrievalMetrics

    # Stage 2 Metrics (Cross-Encoder Reranked)
    stage2_metrics: AggregatedRetrievalMetrics

    # Measured Deltas (Stage 2 - Stage 1)
    delta_mrr: float
    delta_recall_at_k: Dict[int, float] = Field(default_factory=dict)
    delta_precision_at_k: Dict[int, float] = Field(default_factory=dict)
    delta_duplicate_rate: float = 0.0
    delta_page_duplicate_rate: Optional[float] = None

    # Ranking movement summary
    improved_queries_count: int
    degraded_queries_count: int
    unchanged_queries_count: int

    # Scientific integrity assertion
    scientific_statement: str = Field(
        default=(
            "Reranking quality is empirically verified against benchmark relevance annotations; "
            "reranker improvements are never claimed unless demonstrated by positive metric deltas."
        )
    )

    @property
    def empirical_effect(self) -> str:
        """Categorize observed effect of reranking objectively."""
        if self.delta_mrr > 0 or self.improved_queries_count > self.degraded_queries_count:
            return "IMPROVED"
        elif self.delta_mrr < 0 or self.degraded_queries_count > self.improved_queries_count:
            return "DEGRADED"
        return "UNCHANGED"


def compare_retrieval_stages(
    dataset_name: str,
    query_items: List[dict],
    stage1_results_by_query: Dict[str, List[str]],
    stage2_results_by_query: Dict[str, List[str]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    stage1_pages_by_query: Optional[Dict[str, List[int]]] = None,
    stage2_pages_by_query: Optional[Dict[str, List[int]]] = None,
) -> RerankingComparisonReport:
    """Compare Stage 1 (vector search) with Stage 2 (Cross-Encoder reranking).

    Args:
        dataset_name: Name of the evaluation dataset.
        query_items: List of evaluation item dicts (id, question, relevant_chunk_ids, relevant_page_numbers).
        stage1_results_by_query: Map of query_id -> ordered list of chunk_ids before reranking.
        stage2_results_by_query: Map of query_id -> ordered list of chunk_ids after reranking.
        k_values: Ranking evaluation cutoffs (default: 1, 3, 5, 10).
        stage1_pages_by_query: Optional map of query_id -> ordered list of page numbers before reranking.
        stage2_pages_by_query: Optional map of query_id -> ordered list of page numbers after reranking.

    Returns:
        RerankingComparisonReport with before, after, and delta metrics.
    """
    stage1_evals: List[QueryRetrievalResult] = []
    stage2_evals: List[QueryRetrievalResult] = []

    improved_count = 0
    degraded_count = 0
    unchanged_count = 0

    for item in query_items:
        qid = item["id"]
        qtext = item["question"]
        relevant = item.get("relevant_chunk_ids", [])
        relevant_pages = item.get("relevant_page_numbers", None)

        s1_chunks = stage1_results_by_query.get(qid, [])
        s2_chunks = stage2_results_by_query.get(qid, [])
        s1_pages = stage1_pages_by_query.get(qid, None) if stage1_pages_by_query else None
        s2_pages = stage2_pages_by_query.get(qid, None) if stage2_pages_by_query else None

        eval_s1 = evaluate_query_retrieval(
            query_id=qid,
            query=qtext,
            relevant_ids=relevant,
            retrieved_ids=s1_chunks,
            k_values=k_values,
            relevant_pages=relevant_pages,
            retrieved_pages=s1_pages,
        )
        eval_s2 = evaluate_query_retrieval(
            query_id=qid,
            query=qtext,
            relevant_ids=relevant,
            retrieved_ids=s2_chunks,
            k_values=k_values,
            relevant_pages=relevant_pages,
            retrieved_pages=s2_pages,
        )

        stage1_evals.append(eval_s1)
        stage2_evals.append(eval_s2)

        # Compare first relevant rank movement for queries with ground-truth relevance
        if len(relevant) > 0:
            r1 = eval_s1.first_relevant_rank
            r2 = eval_s2.first_relevant_rank

            if r1 is None and r2 is not None:
                improved_count += 1
            elif r1 is not None and r2 is None:
                degraded_count += 1
            elif r1 is not None and r2 is not None:
                if r2 < r1:  # lower index = higher rank position
                    improved_count += 1
                elif r2 > r1:
                    degraded_count += 1
                else:
                    unchanged_count += 1
            else:
                unchanged_count += 1

    agg_s1 = aggregate_retrieval_metrics(stage1_evals, k_values=k_values)
    agg_s2 = aggregate_retrieval_metrics(stage2_evals, k_values=k_values)

    delta_mrr = round(agg_s2.mrr - agg_s1.mrr, 4)
    delta_recall = {
        k: round(agg_s2.mean_recall_at_k.get(k, 0.0) - agg_s1.mean_recall_at_k.get(k, 0.0), 4)
        for k in k_values
    }
    delta_precision = {
        k: round(agg_s2.mean_precision_at_k.get(k, 0.0) - agg_s1.mean_precision_at_k.get(k, 0.0), 4)
        for k in k_values
    }

    delta_duplicate_rate = round(
        agg_s2.mean_duplicate_retrieval_rate - agg_s1.mean_duplicate_retrieval_rate, 4
    )
    delta_page_dup = None
    if agg_s2.mean_page_duplicate_rate is not None and agg_s1.mean_page_duplicate_rate is not None:
        delta_page_dup = round(agg_s2.mean_page_duplicate_rate - agg_s1.mean_page_duplicate_rate, 4)

    return RerankingComparisonReport(
        dataset_name=dataset_name,
        total_queries=len(query_items),
        evaluated_queries=agg_s1.evaluated_queries,
        unanswerable_queries=agg_s1.unanswerable_queries,
        stage1_metrics=agg_s1,
        stage2_metrics=agg_s2,
        delta_mrr=delta_mrr,
        delta_recall_at_k=delta_recall,
        delta_precision_at_k=delta_precision,
        delta_duplicate_rate=delta_duplicate_rate,
        delta_page_duplicate_rate=delta_page_dup,
        improved_queries_count=improved_count,
        degraded_queries_count=degraded_count,
        unchanged_queries_count=unchanged_count,
    )
