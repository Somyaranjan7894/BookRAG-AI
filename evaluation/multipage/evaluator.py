"""Multi-Page Synthesis Evaluation Subsystem for BookRAG AI Phase 23.

Evaluates questions requiring cross-page synthesis across:
1. Whether all required pages were retrieved (all_required_pages_retrieved)
2. Whether required evidence was present in retrieved context (evidence_sufficiency)
3. Whether synthesis used the evidence correctly (synthesis_used_evidence)
4. Whether unsupported information was introduced (unsupported_information_rate)
5. Whether citations cover all necessary evidence pages (citation_page_coverage)
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from evaluation.grounding.evaluator import QueryGroundingResult
from evaluation.retrieval.metrics import QueryRetrievalResult


class MultiPageQueryEvaluation(BaseModel):
    """Detailed evaluation of cross-page synthesis for a single multi-page query."""

    query_id: str
    query: str
    required_pages: List[int]
    required_chunk_ids: List[str]

    # Retrieval across pages
    retrieved_pages: List[int] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    all_required_pages_retrieved: bool
    page_coverage_rate: float
    evidence_sufficiency: bool

    # Synthesis correctness & grounding
    generated_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    answer_correctness_f1: Optional[float] = None
    synthesis_used_evidence: bool
    unsupported_claims_count: int = 0
    contradicted_claims_count: int = 0
    has_unsupported_information: bool = False

    # Citation coverage across pages
    cited_pages: List[int] = Field(default_factory=list)
    all_required_pages_cited: bool = False
    citation_page_coverage: float = 0.0

    status: str = "PASS"
    notes: Optional[str] = None


class AggregatedMultiPageMetrics(BaseModel):
    """Macro-averaged evaluation metrics across all multi-page synthesis benchmark queries."""

    total_multipage_queries: int
    all_pages_retrieved_count: int
    all_pages_retrieved_rate: float
    mean_page_coverage_rate: float

    evidence_sufficiency_count: int
    evidence_sufficiency_rate: float

    correct_synthesis_count: int
    correct_synthesis_rate: float

    zero_unsupported_info_count: int
    zero_unsupported_info_rate: float

    full_citation_coverage_count: int
    full_citation_coverage_rate: float
    mean_citation_page_coverage: float

    mean_answer_f1: Optional[float] = None
    overall_status: str = "PASS"

    per_query_results: List[MultiPageQueryEvaluation] = Field(default_factory=list)


def evaluate_multipage_synthesis(
    multipage_items: List[Dict[str, Any]],
    retrieval_results_map: Dict[str, QueryRetrievalResult],
    grounding_results_map: Dict[str, QueryGroundingResult],
) -> AggregatedMultiPageMetrics:
    """Evaluate multi-page synthesis queries against cross-page evidence and citations."""
    evaluations: List[MultiPageQueryEvaluation] = []

    for item in multipage_items:
        qid = item["id"]
        qtext = item["question"]
        req_pages = sorted(item.get("relevant_page_numbers") or item.get("page_numbers") or [])
        req_chunks = sorted(item.get("relevant_chunk_ids") or item.get("chunk_ids") or [])
        req_page_set = set(req_pages)
        req_chunk_set = set(req_chunks)

        ret_res = retrieval_results_map.get(qid)
        ground_res = grounding_results_map.get(qid)

        ret_pages = (ret_res.retrieved_pages or []) if ret_res else []
        ret_chunks = (ret_res.retrieved_chunk_ids or []) if ret_res else []
        ret_page_set = set(ret_pages)
        ret_chunk_set = set(ret_chunks)

        # 1. Did top-K retrieve all required pages?
        all_pages_ret = req_page_set.issubset(ret_page_set) if req_page_set else True
        hits = len(req_page_set.intersection(ret_page_set)) if req_page_set else len(req_pages)
        page_cov = round(float(hits) / float(len(req_page_set)), 4) if req_page_set else 1.0

        # 2. Was required evidence present in retrieved chunks?
        if req_chunk_set:
            evidence_sufficient = req_chunk_set.issubset(ret_chunk_set)
        else:
            evidence_sufficient = all_pages_ret

        # 3. Grounding & synthesis quality
        unsupp_count = ground_res.unsupported_claims if ground_res else 0
        contra_count = ground_res.contradicted_claims if ground_res else 0
        has_unsupp = (unsupp_count > 0 or contra_count > 0)

        # Synthesis used evidence correctly: unrefused, grounded, and has reasonable F1 or high support
        synthesis_ok = False
        if ground_res and not ground_res.refusal_triggered:
            if ground_res.claim_support_rate >= 0.75 and contra_count == 0:
                synthesis_ok = True

        # 4. Citations coverage across required pages
        cited_pages_list: List[int] = []
        if ground_res and ground_res.citations:
            for cite in ground_res.citations:
                p = cite.get("page_number") if isinstance(cite, dict) else getattr(cite, "page_number", None)
                if p is not None:
                    cited_pages_list.append(p)

        cited_page_set = set(cited_pages_list)
        all_pages_cited = req_page_set.issubset(cited_page_set) if req_page_set else True
        cite_hits = len(req_page_set.intersection(cited_page_set)) if req_page_set else 0
        cite_cov = round(float(cite_hits) / float(len(req_page_set)), 4) if req_page_set else 1.0

        # Verdict
        if all_pages_ret and synthesis_ok and not has_unsupp and all_pages_cited:
            verdict = "PASS"
        elif all_pages_ret and synthesis_ok:
            verdict = "WARNING"  # E.g. citations missed one page or partial unsupported phrasing
        else:
            verdict = "FAIL"

        evaluations.append(
            MultiPageQueryEvaluation(
                query_id=qid,
                query=qtext,
                required_pages=req_pages,
                required_chunk_ids=req_chunks,
                retrieved_pages=ret_pages,
                retrieved_chunk_ids=ret_chunks,
                all_required_pages_retrieved=all_pages_ret,
                page_coverage_rate=page_cov,
                evidence_sufficiency=evidence_sufficient,
                generated_answer=ground_res.generated_answer if ground_res else None,
                expected_answer=item.get("expected_answer"),
                answer_correctness_f1=ground_res.answer_correctness_f1 if ground_res else None,
                synthesis_used_evidence=synthesis_ok,
                unsupported_claims_count=unsupp_count,
                contradicted_claims_count=contra_count,
                has_unsupported_information=has_unsupp,
                cited_pages=sorted(list(cited_page_set)),
                all_required_pages_cited=all_pages_cited,
                citation_page_coverage=cite_cov,
                status=verdict,
            )
        )

    n = len(evaluations)
    if n == 0:
        return AggregatedMultiPageMetrics(
            total_multipage_queries=0,
            all_pages_retrieved_count=0,
            all_pages_retrieved_rate=1.0,
            mean_page_coverage_rate=1.0,
            evidence_sufficiency_count=0,
            evidence_sufficiency_rate=1.0,
            correct_synthesis_count=0,
            correct_synthesis_rate=1.0,
            zero_unsupported_info_count=0,
            zero_unsupported_info_rate=1.0,
            full_citation_coverage_count=0,
            full_citation_coverage_rate=1.0,
            mean_citation_page_coverage=1.0,
            overall_status="NOT_EVALUATED",
        )

    all_pages_ret_cnt = sum(1 for e in evaluations if e.all_required_pages_retrieved)
    evid_suff_cnt = sum(1 for e in evaluations if e.evidence_sufficiency)
    synth_ok_cnt = sum(1 for e in evaluations if e.synthesis_used_evidence)
    zero_unsupp_cnt = sum(1 for e in evaluations if not e.has_unsupported_information)
    full_cite_cnt = sum(1 for e in evaluations if e.all_required_pages_cited)

    mean_cov = sum(e.page_coverage_rate for e in evaluations) / n
    mean_cite_cov = sum(e.citation_page_coverage for e in evaluations) / n

    f1_vals = [e.answer_correctness_f1 for e in evaluations if e.answer_correctness_f1 is not None]
    mean_f1 = (sum(f1_vals) / len(f1_vals)) if f1_vals else None

    # Overall status
    has_fail = any(e.status == "FAIL" for e in evaluations)
    has_warn = any(e.status == "WARNING" for e in evaluations)
    overall = "FAIL" if has_fail else ("WARNING" if has_warn else "PASS")

    return AggregatedMultiPageMetrics(
        total_multipage_queries=n,
        all_pages_retrieved_count=all_pages_ret_cnt,
        all_pages_retrieved_rate=round(float(all_pages_ret_cnt) / float(n), 4),
        mean_page_coverage_rate=round(mean_cov, 4),
        evidence_sufficiency_count=evid_suff_cnt,
        evidence_sufficiency_rate=round(float(evid_suff_cnt) / float(n), 4),
        correct_synthesis_count=synth_ok_cnt,
        correct_synthesis_rate=round(float(synth_ok_cnt) / float(n), 4),
        zero_unsupported_info_count=zero_unsupp_cnt,
        zero_unsupported_info_rate=round(float(zero_unsupp_cnt) / float(n), 4),
        full_citation_coverage_count=full_cite_cnt,
        full_citation_coverage_rate=round(float(full_cite_cnt) / float(n), 4),
        mean_citation_page_coverage=round(mean_cite_cov, 4),
        mean_answer_f1=round(mean_f1, 4) if mean_f1 is not None else None,
        overall_status=overall,
        per_query_results=evaluations,
    )
