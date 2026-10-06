"""Query-Type Evaluation Subsystem for BookRAG AI Phase 23.

Evaluates performance across 9 core query categories:
- direct_fact
- definition
- explanation
- comparison
- numerical_fact
- reasoning
- multi_page_synthesis
- unanswerable
- ambiguous

Calculates category-isolated retrieval, ranking, correctness, groundedness, and refusal metrics.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from evaluation.grounding.evaluator import QueryGroundingResult
from evaluation.retrieval.metrics import QueryRetrievalResult


class CategoryEvaluationMetrics(BaseModel):
    """Evaluation metrics isolated for a specific query category."""

    category: str
    total_queries: int
    answerable_queries: int
    unanswerable_queries: int

    # Retrieval metrics
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    page_recall_at_1: Optional[float] = None

    # Answer Quality metrics
    mean_correctness_f1: Optional[float] = None
    claim_support_rate: float = 0.0
    answerable_success_rate: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0

    # Refusal metrics
    refusal_precision: float = 1.0
    false_refusal_rate: float = 0.0

    # Timing metrics
    mean_latency_ms: float = 0.0

    # Qualitative verdict: PASS, WARNING, FAIL, NOT_EVALUATED
    status: str = "NOT_EVALUATED"
    notes: Optional[str] = None


class QueryTypeReport(BaseModel):
    """Benchmark report aggregated by question categories."""

    benchmark_name: str
    total_queries: int
    categories: Dict[str, CategoryEvaluationMetrics] = Field(default_factory=dict)
    overall_status: str = "PASS"


def evaluate_by_query_type(
    benchmark_name: str,
    query_items: List[Dict[str, Any]],
    retrieval_results_map: Dict[str, QueryRetrievalResult],
    grounding_results_map: Dict[str, QueryGroundingResult],
    latency_map: Optional[Dict[str, float]] = None,
) -> QueryTypeReport:
    """Group query results by category and evaluate metrics per category."""
    # Group items by category
    categorized_items: Dict[str, List[Dict[str, Any]]] = {}
    for item in query_items:
        cat = item.get("category") or item.get("query_type") or "uncategorized"
        # Normalize legacy categories
        if cat == "multi_page":
            cat = "multi_page_synthesis"
        categorized_items.setdefault(cat, []).append(item)

    categories_report: Dict[str, CategoryEvaluationMetrics] = {}

    for cat, items in sorted(categorized_items.items()):
        total = len(items)
        ans_items = [it for it in items if it.get("answerable", it.get("answerability", True))]
        unans_items = [it for it in items if not it.get("answerable", it.get("answerability", True))]

        # Retrieval metrics across answerable items
        evaluable_retrievals = [
            retrieval_results_map[it["id"]]
            for it in ans_items
            if it["id"] in retrieval_results_map and retrieval_results_map[it["id"]].num_relevant > 0
        ]

        if evaluable_retrievals:
            n_ret = len(evaluable_retrievals)
            mrr = sum(r.reciprocal_rank for r in evaluable_retrievals) / n_ret
            rec1 = sum(r.recall_at_k.get(1, 0.0) for r in evaluable_retrievals) / n_ret
            rec3 = sum(r.recall_at_k.get(3, 0.0) for r in evaluable_retrievals) / n_ret
            rec5 = sum(r.recall_at_k.get(5, 0.0) for r in evaluable_retrievals) / n_ret
            rec10 = sum(r.recall_at_k.get(10, 0.0) for r in evaluable_retrievals) / n_ret
            prec1 = sum(r.precision_at_k.get(1, 0.0) for r in evaluable_retrievals) / n_ret
            prec3 = sum(r.precision_at_k.get(3, 0.0) for r in evaluable_retrievals) / n_ret
            prec5 = sum(r.precision_at_k.get(5, 0.0) for r in evaluable_retrievals) / n_ret
            prec10 = sum(r.precision_at_k.get(10, 0.0) for r in evaluable_retrievals) / n_ret

            page_recs = [
                r.page_recall_at_k.get(1, 0.0)
                for r in evaluable_retrievals
                if r.page_recall_at_k is not None
            ]
            page_rec1 = (sum(page_recs) / len(page_recs)) if page_recs else None
        else:
            mrr = rec1 = rec3 = rec5 = rec10 = prec1 = prec3 = prec5 = prec10 = 0.0
            page_rec1 = None

        # Grounding and Answer Quality
        groundings = [grounding_results_map[it["id"]] for it in items if it["id"] in grounding_results_map]
        ans_groundings = [g for g in groundings if g.is_answerable_gold]
        unans_groundings = [g for g in groundings if not g.is_answerable_gold]

        f1_list = [g.answer_correctness_f1 for g in ans_groundings if g.answer_correctness_f1 is not None]
        mean_f1 = (sum(f1_list) / len(f1_list)) if f1_list else None

        tot_claims = sum(g.total_claims for g in ans_groundings)
        supp_claims = sum(g.supported_claims for g in ans_groundings)
        claim_supp_rate = (supp_claims / tot_claims) if tot_claims > 0 else (1.0 if not ans_items else 0.0)

        succ_count = sum(1 for g in ans_groundings if g.is_answerable_success)
        ans_succ_rate = (succ_count / len(ans_groundings)) if ans_groundings else 0.0

        cite_precs = [g.citation_precision for g in ans_groundings]
        cite_recs = [g.citation_recall for g in ans_groundings]
        mean_cite_prec = (sum(cite_precs) / len(cite_precs)) if cite_precs else 0.0
        mean_cite_rec = (sum(cite_recs) / len(cite_recs)) if cite_recs else 0.0

        # Refusal metrics
        if unans_groundings:
            corr_refusals = sum(1 for g in unans_groundings if g.refusal_triggered)
            ref_prec = corr_refusals / len(unans_groundings)
        else:
            ref_prec = 1.0

        false_refusals = sum(1 for g in ans_groundings if g.is_false_refusal)
        false_ref_rate = (false_refusals / len(ans_groundings)) if ans_groundings else 0.0

        # Latency
        if latency_map:
            lats = [latency_map[it["id"]] for it in items if it["id"] in latency_map]
            mean_lat = (sum(lats) / len(lats)) if lats else 0.0
        else:
            mean_lat = 0.0

        # Qualitative status determination
        if unans_items:
            # For unanswerable/ambiguous: 100% refusal is PASS, <100% is FAIL/WARNING
            if ref_prec >= 1.0:
                cat_status = "PASS"
            elif ref_prec >= 0.75:
                cat_status = "WARNING"
            else:
                cat_status = "FAIL"
        else:
            # For answerable queries: based on retrieval and success
            if mrr >= 0.8 and ans_succ_rate >= 0.75 and false_ref_rate == 0.0:
                cat_status = "PASS"
            elif mrr >= 0.5 or ans_succ_rate >= 0.5:
                cat_status = "WARNING"
            else:
                cat_status = "FAIL"

        categories_report[cat] = CategoryEvaluationMetrics(
            category=cat,
            total_queries=total,
            answerable_queries=len(ans_items),
            unanswerable_queries=len(unans_items),
            mrr=round(mrr, 4),
            recall_at_1=round(rec1, 4),
            recall_at_3=round(rec3, 4),
            recall_at_5=round(rec5, 4),
            recall_at_10=round(rec10, 4),
            precision_at_1=round(prec1, 4),
            precision_at_3=round(prec3, 4),
            precision_at_5=round(prec5, 4),
            precision_at_10=round(prec10, 4),
            page_recall_at_1=round(page_rec1, 4) if page_rec1 is not None else None,
            mean_correctness_f1=round(mean_f1, 4) if mean_f1 is not None else None,
            claim_support_rate=round(claim_supp_rate, 4),
            answerable_success_rate=round(ans_succ_rate, 4),
            citation_precision=round(mean_cite_prec, 4),
            citation_recall=round(mean_cite_rec, 4),
            refusal_precision=round(ref_prec, 4),
            false_refusal_rate=round(false_ref_rate, 4),
            mean_latency_ms=round(mean_lat, 2),
            status=cat_status,
        )

    # Overall status
    has_fail = any(c.status == "FAIL" for c in categories_report.values())
    has_warn = any(c.status == "WARNING" for c in categories_report.values())
    overall = "FAIL" if has_fail else ("WARNING" if has_warn else "PASS")

    return QueryTypeReport(
        benchmark_name=benchmark_name,
        total_queries=len(query_items),
        categories=categories_report,
        overall_status=overall,
    )
