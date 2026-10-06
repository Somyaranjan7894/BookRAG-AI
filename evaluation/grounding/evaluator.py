"""Grounding, Hallucination Control, Refusal, and Answer Quality Evaluation.

Evaluates:
- Sentence-level claim decomposition via ClaimDecomposer
- NLI entailment classification via GroundingService
- Aggregated claim support rates (supported, unsupported, contradicted, conflicted)
- Answer correctness against ground-truth expected answers (token F1 overlap)
- Answerability and refusal behavior for unanswerable/out-of-scope questions
- Safe refusal verification under insufficient evidence
- Citation correctness and isolation (precision and recall over ground truth chunks/pages)
- Answerable-question success rate

CRITICAL SEMANTIC PRINCIPLES:
1. Retrieval relevance != answer correctness.
2. Reranker score != probability.
3. NLI score != truth probability.
4. Model confidence != factual correctness.
Claim support rate measures strictly premise-hypothesis NLI logical consistency under
the retrieved evidence. It does NOT represent a factual truth probability or proof.
"""

import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


SEMANTIC_DISCLAIMER: str = (
    "Claim support rate measures premise-hypothesis NLI logical consistency "
    "under the retrieved evidence set; it is strictly NOT an epistemic truth probability. "
    "Retrieval relevance != correctness; reranker score != probability; "
    "model confidence != factual correctness."
)


def normalize_answer_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, articles, and extra whitespace."""
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Normalize whitespace
    return " ".join(text.split())


def compute_token_overlap_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Compute token-level Precision, Recall, and F1 score between prediction and reference.

    Returns:
        (precision, recall, f1)
    """
    pred_tokens = normalize_answer_text(prediction).split()
    gold_tokens = normalize_answer_text(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    common: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    overlap_count = 0
    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
        if pred_counts[t] <= gold_counts.get(t, 0):
            overlap_count += 1

    if overlap_count == 0:
        return 0.0, 0.0, 0.0

    precision = float(overlap_count) / float(len(pred_tokens))
    recall = float(overlap_count) / float(len(gold_tokens))
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


class QueryGroundingResult(BaseModel):
    """Grounding, answer quality, refusal, and citation evaluation for a single query."""

    query_id: str
    query: str
    is_answerable_gold: bool
    category: Optional[str] = None
    expected_answer: Optional[str] = None
    generated_answer: Optional[str] = None
    grounding_status: str

    # Claim Decomposition & Grounding
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    conflicted_claims: int
    claim_support_rate: float

    # Answer Quality (for answerable questions)
    answer_correctness_f1: Optional[float] = None
    is_answer_correct: Optional[bool] = None

    # Refusal Evaluation
    refusal_triggered: bool
    refusal_correct: bool
    is_false_refusal: bool = False
    is_incorrect_answer_to_unanswerable: bool = False

    # Answerable Query Success
    is_answerable_success: bool = False

    # Citation Correctness
    total_citations: int = 0
    valid_citations: int = 0
    citation_precision: float = 0.0
    citation_recall: float = 0.0

    claims_breakdown: List[Dict[str, str]] = Field(default_factory=list)
    citations: List[Any] = Field(default_factory=list)


class AggregatedGroundingMetrics(BaseModel):
    """Aggregated grounding, answer quality, and refusal metrics across a benchmark."""

    dataset_name: str
    verification_status: str = "NOT_VERIFIED"
    total_questions: int
    answerable_questions: int
    unanswerable_questions: int

    # Claim Counts across all generated answers
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    conflicted_claims: int

    # Macro Rates
    overall_claim_support_rate: float
    answerable_claim_support_rate: float

    # Refusal & Safety Validation
    unanswerable_refusal_count: int
    incorrect_unanswerable_answers: int
    refusal_precision: float
    false_refusal_count: int
    over_refusal_rate: float

    # Answerable Success Rate
    answerable_success_rate: float

    # Answer Quality
    mean_answer_correctness_f1: Optional[float] = None

    # Citation Quality
    mean_citation_precision: float = 0.0
    mean_citation_recall: float = 0.0

    # Question-category breakdown
    category_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    semantic_disclaimer: str = SEMANTIC_DISCLAIMER

    # Detailed per-query evaluations
    per_query_results: List[QueryGroundingResult] = Field(default_factory=list)


def evaluate_grounded_response(
    query_id: str,
    query: str,
    is_answerable_gold: bool,
    grounding_status: str,
    generated_answer: Optional[str] = None,
    expected_answer: Optional[str] = None,
    category: Optional[str] = None,
    claims_results: Optional[List[dict]] = None,
    citations_list: Optional[List[Any]] = None,
    relevant_chunk_ids: Optional[List[str]] = None,
    relevant_page_numbers: Optional[List[int]] = None,
) -> QueryGroundingResult:
    """Evaluate grounding, answer quality, refusal, and citations for a single response.

    Args:
        query_id: Question ID.
        query: Query string.
        is_answerable_gold: Ground-truth answerability flag.
        grounding_status: Status from grounding orchestrator ('grounded', 'unsupported', 'insufficient_evidence', etc.).
        generated_answer: Answer text from generator.
        expected_answer: Benchmark ground-truth answer text (if answerable).
        category: Question category (direct_fact, definition, unanswerable, etc.).
        claims_results: List of claim dictionaries with 'claim_text' and 'classification'/'status'.
        citations_list: List of citation objects or citation dicts assigned.
        relevant_chunk_ids: Ground truth chunk IDs containing supporting evidence.
        relevant_page_numbers: Ground truth page numbers containing supporting evidence.

    Returns:
        QueryGroundingResult.
    """
    claims = claims_results or []
    total = len(claims)
    supported = sum(
        1 for c in claims if c.get("classification") == "entailed" or c.get("status") == "entailed"
    )
    unsupported = sum(
        1 for c in claims if c.get("classification") in ("unsupported", "neutral") or c.get("status") in ("unsupported", "neutral")
    )
    contradicted = sum(
        1 for c in claims if c.get("classification") == "contradicted" or c.get("status") == "contradicted"
    )
    conflicted = sum(
        1 for c in claims if c.get("classification") == "conflicted" or c.get("status") == "conflicted"
    )

    support_rate = round(float(supported) / float(total), 4) if total > 0 else (1.0 if not is_answerable_gold else 0.0)

    # Refusal detection keywords
    refusal_keywords = [
        "cannot answer",
        "insufficient evidence",
        "not provided",
        "not mentioned",
        "no evidence",
        "unanswerable",
        "cannot determine",
        "out of scope",
    ]
    answer_text = generated_answer or ""
    answer_text_lower = answer_text.lower()
    refusal_in_text = any(kw in answer_text_lower for kw in refusal_keywords)

    refusal_triggered = (
        grounding_status in ("insufficient_evidence", "ungrounded", "contradicted")
        or not generated_answer
        or refusal_in_text
        or (not is_answerable_gold and supported == 0)
    )

    # Refusal correctness
    if is_answerable_gold:
        refusal_correct = not refusal_triggered
        is_false_refusal = refusal_triggered
        is_incorrect_answer = False
    else:
        refusal_correct = refusal_triggered
        is_false_refusal = False
        is_incorrect_answer = not refusal_triggered

    # Answer correctness calculation (for answerable questions)
    f1_score: Optional[float] = None
    is_correct: Optional[bool] = None
    if is_answerable_gold:
        if expected_answer and generated_answer and not refusal_triggered:
            _, _, f1_score = compute_token_overlap_f1(generated_answer, expected_answer)
            # Answer is judged correct if substantial token overlap or factual phrases match
            is_correct = f1_score >= 0.35
        else:
            f1_score = 0.0
            is_correct = False
    else:
        # For unanswerable questions, safe refusal is correct behavior
        f1_score = 1.0 if refusal_triggered else 0.0
        is_correct = refusal_triggered

    # Answerable success: not refused, grounded or partially grounded, no contradiction
    is_answerable_success = False
    if is_answerable_gold:
        is_answerable_success = (
            not refusal_triggered
            and grounding_status in ("grounded", "partially_grounded")
            and contradicted == 0
            and supported > 0
        )

    # Citation correctness evaluation
    cites = citations_list or []
    tot_cites = len(cites)
    valid_cites = 0
    rel_chunk_set = set(relevant_chunk_ids or [])
    rel_page_set = set(relevant_page_numbers or [])

    if is_answerable_gold:
        for cite in cites:
            # Citation can be a dict, Pydantic model, or string ID
            cid = cite.get("chunk_id") if isinstance(cite, dict) else getattr(cite, "chunk_id", str(cite))
            pnum = cite.get("page_number") if isinstance(cite, dict) else getattr(cite, "page_number", None)

            is_valid = False
            if cid and rel_chunk_set and cid in rel_chunk_set:
                is_valid = True
            elif pnum and rel_page_set and pnum in rel_page_set:
                is_valid = True
            elif not rel_chunk_set and not rel_page_set:
                is_valid = True

            if is_valid:
                valid_cites += 1

        citation_prec = round(float(valid_cites) / float(tot_cites), 4) if tot_cites > 0 else 0.0
        # Citation recall: how many relevant items were cited
        cited_chunks = {
            cite.get("chunk_id") if isinstance(cite, dict) else getattr(cite, "chunk_id", str(cite))
            for cite in cites
        }
        hits = len(cited_chunks.intersection(rel_chunk_set)) if rel_chunk_set else 0
        citation_rec = round(float(hits) / float(len(rel_chunk_set)), 4) if rel_chunk_set else 1.0
    else:
        # Unanswerable queries should produce 0 citations
        if tot_cites == 0:
            citation_prec = 1.0  # Perfect isolation: no false citations
            citation_rec = 1.0
            valid_cites = 0
        else:
            citation_prec = 0.0  # Hallucinated citation on unanswerable query
            citation_rec = 0.0
            valid_cites = 0

    return QueryGroundingResult(
        query_id=query_id,
        query=query,
        is_answerable_gold=is_answerable_gold,
        category=category,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        grounding_status=grounding_status,
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        contradicted_claims=contradicted,
        conflicted_claims=conflicted,
        claim_support_rate=support_rate,
        answer_correctness_f1=f1_score,
        is_answer_correct=is_correct,
        refusal_triggered=refusal_triggered,
        refusal_correct=refusal_correct,
        is_false_refusal=is_false_refusal,
        is_incorrect_answer_to_unanswerable=is_incorrect_answer,
        is_answerable_success=is_answerable_success,
        total_citations=tot_cites,
        valid_citations=valid_cites,
        citation_precision=citation_prec,
        citation_recall=citation_rec,
        claims_breakdown=claims,
        citations=cites,
    )


def aggregate_grounding_metrics(
    dataset_name: str,
    results: List[QueryGroundingResult],
    verification_status: str = "NOT_VERIFIED",
) -> AggregatedGroundingMetrics:
    """Aggregate individual query grounding results into comprehensive benchmark metrics."""
    total_q = len(results)
    answerable_q = [r for r in results if r.is_answerable_gold]
    unanswerable_q = [r for r in results if not r.is_answerable_gold]

    total_claims = sum(r.total_claims for r in results)
    supp_claims = sum(r.supported_claims for r in results)
    unsupp_claims = sum(r.unsupported_claims for r in results)
    contra_claims = sum(r.contradicted_claims for r in results)
    conf_claims = sum(r.conflicted_claims for r in results)

    overall_support_rate = round(float(supp_claims) / float(total_claims), 4) if total_claims > 0 else 0.0

    ans_claims = sum(r.total_claims for r in answerable_q)
    ans_supp = sum(r.supported_claims for r in answerable_q)
    ans_support_rate = round(float(ans_supp) / float(ans_claims), 4) if ans_claims > 0 else 0.0

    correct_refusals = sum(1 for r in unanswerable_q if r.refusal_triggered)
    incorrect_unanswerable = sum(1 for r in unanswerable_q if not r.refusal_triggered)
    refusal_precision = (
        round(float(correct_refusals) / float(len(unanswerable_q)), 4) if unanswerable_q else 1.0
    )

    false_refusals = sum(1 for r in answerable_q if r.is_false_refusal)
    over_refusal_rate = (
        round(float(false_refusals) / float(len(answerable_q)), 4) if answerable_q else 0.0
    )

    successful_ans = sum(1 for r in answerable_q if r.is_answerable_success)
    ans_success_rate = (
        round(float(successful_ans) / float(len(answerable_q)), 4) if answerable_q else 0.0
    )

    # Mean answer correctness F1 for answerable questions
    f1_values = [r.answer_correctness_f1 for r in answerable_q if r.answer_correctness_f1 is not None]
    mean_f1 = round(sum(f1_values) / len(f1_values), 4) if f1_values else None

    # Mean citation precision and recall across answerable queries
    ans_cite_precs = [r.citation_precision for r in answerable_q]
    ans_cite_recs = [r.citation_recall for r in answerable_q]
    mean_cite_prec = round(sum(ans_cite_precs) / len(ans_cite_precs), 4) if ans_cite_precs else 0.0
    mean_cite_rec = round(sum(ans_cite_recs) / len(ans_cite_recs), 4) if ans_cite_recs else 0.0

    # Category-level breakdown
    category_map: Dict[str, List[QueryGroundingResult]] = {}
    for r in results:
        cat = r.category or "uncategorized"
        category_map.setdefault(cat, []).append(r)

    category_metrics: Dict[str, Dict[str, Any]] = {}
    for cat, cat_results in category_map.items():
        cat_ans = [r for r in cat_results if r.is_answerable_gold]
        cat_claims = sum(r.total_claims for r in cat_results)
        cat_supp = sum(r.supported_claims for r in cat_results)
        cat_supp_rate = round(float(cat_supp) / float(cat_claims), 4) if cat_claims > 0 else 0.0
        category_metrics[cat] = {
            "total_questions": len(cat_results),
            "answerable": len(cat_ans),
            "unanswerable": len(cat_results) - len(cat_ans),
            "claim_support_rate": cat_supp_rate,
            "success_rate": round(sum(1 for r in cat_ans if r.is_answerable_success) / len(cat_ans), 4) if cat_ans else (1.0 if all(r.refusal_correct for r in cat_results) else 0.0),
        }

    return AggregatedGroundingMetrics(
        dataset_name=dataset_name,
        verification_status=verification_status,
        total_questions=total_q,
        answerable_questions=len(answerable_q),
        unanswerable_questions=len(unanswerable_q),
        total_claims=total_claims,
        supported_claims=supp_claims,
        unsupported_claims=unsupp_claims,
        contradicted_claims=contra_claims,
        conflicted_claims=conf_claims,
        overall_claim_support_rate=overall_support_rate,
        answerable_claim_support_rate=ans_support_rate,
        unanswerable_refusal_count=correct_refusals,
        incorrect_unanswerable_answers=incorrect_unanswerable,
        refusal_precision=refusal_precision,
        false_refusal_count=false_refusals,
        over_refusal_rate=over_refusal_rate,
        answerable_success_rate=ans_success_rate,
        mean_answer_correctness_f1=mean_f1,
        mean_citation_precision=mean_cite_prec,
        mean_citation_recall=mean_cite_rec,
        category_metrics=category_metrics,
        semantic_disclaimer=SEMANTIC_DISCLAIMER,
        per_query_results=results,
    )
