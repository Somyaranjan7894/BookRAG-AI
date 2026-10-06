"""Information Retrieval Evaluation Metrics.

Computes standard deterministic IR ranking metrics:
- Recall@1, Recall@3, Recall@5, Recall@10
- Precision@1, Precision@3, Precision@5, Precision@10
- Mean Reciprocal Rank (MRR)

Supports evaluation across:
- Chunk-level relevance (relevant_chunk_ids)
- Page-level relevance (relevant_page_numbers)

Handles edge cases robustly:
- No relevant evidence (omitted from ranking recall/MRR denominator)
- Empty retrieval results
- Duplicate chunks (deduplicated preserving best rank order)
- Out-of-scope / unanswerable queries
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from pydantic import BaseModel, Field


DEFAULT_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)
DEFAULT_PRECISION_K_VALUES: Tuple[int, ...] = (1, 3, 5)


class QueryRetrievalResult(BaseModel):
    """Retrieval evaluation metrics for a single query."""

    query_id: str
    query: str
    num_relevant: int
    num_retrieved: int
    recall_at_k: Dict[int, float] = Field(default_factory=dict)
    precision_at_k: Dict[int, float] = Field(default_factory=dict)
    reciprocal_rank: float = 0.0
    first_relevant_rank: Optional[int] = None
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    relevant_chunk_ids: List[str] = Field(default_factory=list)

    # Duplicate retrieval tracking
    duplicate_retrieval_rate: float = 0.0
    page_duplicate_rate: Optional[float] = None

    # Optional page-level retrieval evaluation
    page_recall_at_k: Optional[Dict[int, float]] = None
    page_precision_at_k: Optional[Dict[int, float]] = None
    page_reciprocal_rank: Optional[float] = None
    first_relevant_page_rank: Optional[int] = None
    retrieved_pages: Optional[List[int]] = None
    relevant_pages: Optional[List[int]] = None


class AggregatedRetrievalMetrics(BaseModel):
    """Macro-averaged retrieval metrics across a benchmark evaluation set."""

    total_queries: int
    evaluated_queries: int
    unanswerable_queries: int
    mrr: float = 0.0
    mean_recall_at_k: Dict[int, float] = Field(default_factory=dict)
    mean_precision_at_k: Dict[int, float] = Field(default_factory=dict)
    mean_duplicate_retrieval_rate: float = 0.0
    mean_page_duplicate_rate: Optional[float] = None

    # Optional macro-averaged page-level metrics
    page_mrr: Optional[float] = None
    mean_page_recall_at_k: Optional[Dict[int, float]] = None
    mean_page_precision_at_k: Optional[Dict[int, float]] = None

    per_query_results: List[QueryRetrievalResult] = Field(default_factory=list)


def compute_duplicate_rate(items: List[Any]) -> float:
    """Compute fraction of items that are duplicate entries."""
    if not items:
        return 0.0
    total = len(items)
    unique = len(set(items))
    return round(float(total - unique) / float(total), 4)


def deduplicate_preserving_order(items: List[str]) -> List[str]:
    """Deduplicate a list of identifiers while preserving first-seen rank order."""
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def compute_recall_at_k(relevant_ids: Set[Any], retrieved_ids: List[Any], k: int) -> float:
    """Compute Recall@K: fraction of relevant items retrieved in top K."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    hits = len(top_k_set.intersection(relevant_ids))
    return float(hits) / float(len(relevant_ids))


def compute_precision_at_k(relevant_ids: Set[Any], retrieved_ids: List[Any], k: int) -> float:
    """Compute Precision@K: fraction of top K retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k_slice = retrieved_ids[:k]
    if not top_k_slice:
        return 0.0
    hits = sum(1 for item in top_k_slice if item in relevant_ids)
    return float(hits) / float(k)


def compute_reciprocal_rank(relevant_ids: Set[Any], retrieved_ids: List[Any]) -> Tuple[float, Optional[int]]:
    """Compute Reciprocal Rank (1/rank) of the first relevant retrieved item."""
    if not relevant_ids or not retrieved_ids:
        return 0.0, None
    for rank_idx, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / float(rank_idx), rank_idx
    return 0.0, None


def evaluate_query_retrieval(
    query_id: str,
    query: str,
    relevant_ids: List[str],
    retrieved_ids: List[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    relevant_pages: Optional[List[int]] = None,
    retrieved_pages: Optional[List[int]] = None,
) -> QueryRetrievalResult:
    """Evaluate retrieval results for a single query against relevant ground truth IDs and pages.

    Args:
        query_id: Unique query identifier.
        query: Query text.
        relevant_ids: Ground truth relevant chunk IDs.
        retrieved_ids: List of retrieved chunk IDs ordered by rank.
        k_values: Ranking thresholds (default: 1, 3, 5, 10).
        relevant_pages: Optional list of relevant 1-based page numbers.
        retrieved_pages: Optional list of retrieved page numbers ordered by rank.

    Returns:
        QueryRetrievalResult with chunk and optional page-level metrics.
    """
    deduped_retrieved = deduplicate_preserving_order(retrieved_ids)
    relevant_set = set(relevant_ids)

    reciprocal_rank, first_rank = compute_reciprocal_rank(relevant_set, deduped_retrieved)
    recall_map: Dict[int, float] = {}
    precision_map: Dict[int, float] = {}

    for k in k_values:
        recall_map[k] = round(compute_recall_at_k(relevant_set, deduped_retrieved, k), 4)
        precision_map[k] = round(compute_precision_at_k(relevant_set, deduped_retrieved, k), 4)

    # Optional page-level evaluation
    page_recall_map: Optional[Dict[int, float]] = None
    page_prec_map: Optional[Dict[int, float]] = None
    page_rr: Optional[float] = None
    first_page_rank: Optional[int] = None

    # Compute duplicate retrieval rate on raw retrieved items
    chunk_dup_rate = compute_duplicate_rate(retrieved_ids)
    page_dup_rate = compute_duplicate_rate(retrieved_pages) if retrieved_pages is not None else None

    if relevant_pages is not None and retrieved_pages is not None:
        rel_page_set = set(relevant_pages)
        page_rr, first_page_rank = compute_reciprocal_rank(rel_page_set, retrieved_pages)
        page_recall_map = {}
        page_prec_map = {}
        for k in k_values:
            page_recall_map[k] = round(compute_recall_at_k(rel_page_set, retrieved_pages, k), 4)
            page_prec_map[k] = round(compute_precision_at_k(rel_page_set, retrieved_pages, k), 4)

    return QueryRetrievalResult(
        query_id=query_id,
        query=query,
        num_relevant=len(relevant_set),
        num_retrieved=len(deduped_retrieved),
        recall_at_k=recall_map,
        precision_at_k=precision_map,
        reciprocal_rank=round(reciprocal_rank, 4),
        first_relevant_rank=first_rank,
        retrieved_chunk_ids=deduped_retrieved,
        relevant_chunk_ids=relevant_ids,
        duplicate_retrieval_rate=chunk_dup_rate,
        page_duplicate_rate=page_dup_rate,
        page_recall_at_k=page_recall_map,
        page_precision_at_k=page_prec_map,
        page_reciprocal_rank=round(page_rr, 4) if page_rr is not None else None,
        first_relevant_page_rank=first_page_rank,
        retrieved_pages=retrieved_pages,
        relevant_pages=relevant_pages,
    )


def aggregate_retrieval_metrics(
    query_results: List[QueryRetrievalResult],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> AggregatedRetrievalMetrics:
    """Aggregate individual query retrieval results into macro-averaged benchmark metrics."""
    evaluable = [q for q in query_results if q.num_relevant > 0]
    unanswerable_count = len(query_results) - len(evaluable)

    if not evaluable:
        return AggregatedRetrievalMetrics(
            total_queries=len(query_results),
            evaluated_queries=0,
            unanswerable_queries=unanswerable_count,
            mrr=0.0,
            mean_recall_at_k={k: 0.0 for k in k_values},
            mean_precision_at_k={k: 0.0 for k in k_values},
            mean_duplicate_retrieval_rate=0.0,
            per_query_results=query_results,
        )

    n = len(evaluable)
    mrr = sum(q.reciprocal_rank for q in evaluable) / n

    mean_recall: Dict[int, float] = {}
    mean_precision: Dict[int, float] = {}

    for k in k_values:
        mean_recall[k] = sum(q.recall_at_k.get(k, 0.0) for q in evaluable) / n
        mean_precision[k] = sum(q.precision_at_k.get(k, 0.0) for q in evaluable) / n

    mean_chunk_dup = sum(q.duplicate_retrieval_rate for q in evaluable) / n

    # Aggregate page-level metrics if available
    evaluable_pages = [q for q in evaluable if q.page_recall_at_k is not None]
    page_mrr: Optional[float] = None
    mean_page_recall: Optional[Dict[int, float]] = None
    mean_page_precision: Optional[Dict[int, float]] = None
    mean_page_dup: Optional[float] = None

    if evaluable_pages:
        pn = len(evaluable_pages)
        page_mrr = round(sum(q.page_reciprocal_rank or 0.0 for q in evaluable_pages) / pn, 4)
        mean_page_recall = {}
        mean_page_precision = {}
        for k in k_values:
            mean_page_recall[k] = round(
                sum(q.page_recall_at_k.get(k, 0.0) for q in evaluable_pages if q.page_recall_at_k) / pn,
                4,
            )
            mean_page_precision[k] = round(
                sum(q.page_precision_at_k.get(k, 0.0) for q in evaluable_pages if q.page_precision_at_k) / pn,
                4,
            )
        page_dups = [q.page_duplicate_rate for q in evaluable_pages if q.page_duplicate_rate is not None]
        mean_page_dup = round(sum(page_dups) / len(page_dups), 4) if page_dups else None

    return AggregatedRetrievalMetrics(
        total_queries=len(query_results),
        evaluated_queries=n,
        unanswerable_queries=unanswerable_count,
        mrr=round(mrr, 4),
        mean_recall_at_k={k: round(v, 4) for k, v in mean_recall.items()},
        mean_precision_at_k={k: round(v, 4) for k, v in mean_precision.items()},
        mean_duplicate_retrieval_rate=round(mean_chunk_dup, 4),
        mean_page_duplicate_rate=mean_page_dup,
        page_mrr=page_mrr,
        mean_page_recall_at_k=mean_page_recall,
        mean_page_precision_at_k=mean_page_precision,
        per_query_results=query_results,
    )
