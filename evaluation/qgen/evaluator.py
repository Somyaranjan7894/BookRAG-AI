"""Question Generation Evaluator for BookRAG AI Phase 23.

Evaluates the Phase 22 Question Generation pipeline across multiple chapters and pages:
- Raw candidates generated
- Accepted questions
- Rejected questions
- Acceptance rate & rejection rate
- Duplicate rate
- Answer mismatch rate
- Unanswerable rate
- Quality failure rate
- Question-type distribution
- Provenance completeness (document_id, page_number, chunk_id, offsets)
- Grounding validity against source evidence
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.question_generation import GeneratedQuestion, QuestionGenerationResponse


class QuestionGenerationEvaluationItem(BaseModel):
    """Detailed evaluation of question generation for a single chunk."""

    chunk_id: str
    document_id: str
    page_number: int
    raw_candidates_count: int
    accepted_count: int
    rejected_count: int

    # Rejection breakdown
    duplicates_count: int = 0
    answer_mismatches_count: int = 0
    unanswerable_count: int = 0
    quality_failures_count: int = 0

    # Provenance and grounding
    all_provenance_complete: bool = True
    all_grounded_in_source: bool = True

    # Accepted question types
    question_types: List[str] = Field(default_factory=list)
    accepted_questions: List[Dict[str, Any]] = Field(default_factory=list)


class QuestionGenerationBenchmarkReport(BaseModel):
    """Aggregated evaluation metrics for Question Generation across full-book chapters."""

    benchmark_name: str
    total_chunks_evaluated: int
    total_pages_covered: int

    # Pipeline Volume
    total_raw_candidates: int
    total_accepted_questions: int
    total_rejected_questions: int
    acceptance_rate: float
    rejection_rate: float

    # Quality & Failure Rates
    duplicate_rate: float
    answer_mismatch_rate: float
    unanswerable_rate: float
    quality_failure_rate: float

    # Provenance & Grounding
    provenance_completeness_rate: float
    grounding_validity_rate: float

    # Question Type Distribution
    question_type_distribution: Dict[str, int] = Field(default_factory=dict)

    # Status: PASS / WARNING / FAIL
    overall_status: str = "PASS"
    per_chunk_evaluations: List[QuestionGenerationEvaluationItem] = Field(default_factory=list)


def evaluate_question_generation_runs(
    benchmark_name: str,
    generation_results: List[Dict[str, Any]],
) -> QuestionGenerationBenchmarkReport:
    """Analyze QuestionGenerationResponse outputs across multiple book chunks.

    Args:
        benchmark_name: Name of benchmark run.
        generation_results: List of dicts, each with:
            - chunk_id: str
            - document_id: str
            - page_number: int
            - chunk_text: str
            - response: QuestionGenerationResponse or dict
    """
    chunk_evals: List[QuestionGenerationEvaluationItem] = []
    pages_seen = set()

    tot_raw = 0
    tot_accepted = 0
    tot_rejected = 0
    tot_dups = 0
    tot_mismatch = 0
    tot_unans = 0
    tot_qual = 0

    provenance_complete_count = 0
    grounding_valid_count = 0
    type_counts: Dict[str, int] = {}

    for item in generation_results:
        cid = item["chunk_id"]
        did = item["document_id"]
        pnum = item["page_number"]
        pages_seen.add(pnum)
        text = item.get("chunk_text", "")
        resp = item["response"]

        if isinstance(resp, QuestionGenerationResponse):
            raw_cands = getattr(resp, "generated_candidates", getattr(resp, "raw_candidates_count", len(resp.questions)))
            acc_list = resp.questions
            rej_summary = resp.rejection_summary or {}
            dups = rej_summary.get("duplicate", 0) + rej_summary.get("near_duplicate", 0)
            mismatch = rej_summary.get("answer_mismatch", 0)
            unans = rej_summary.get("unanswerable", 0)
            qual = rej_summary.get("low_quality", 0) + rej_summary.get("syntax_error", 0)
            rej_count = len(resp.rejected_candidates) if resp.rejected_candidates else (raw_cands - len(acc_list) if raw_cands >= len(acc_list) else 0)
        else:
            raw_cands = resp.get("generated_candidates", resp.get("raw_candidates_count", len(resp.get("questions", []))))
            acc_list = resp.get("questions", [])
            rej_summary = resp.get("rejection_summary", {})
            dups = rej_summary.get("duplicate", 0) + rej_summary.get("near_duplicate", 0)
            mismatch = rej_summary.get("answer_mismatch", 0)
            unans = rej_summary.get("unanswerable", 0)
            qual = rej_summary.get("low_quality", 0) + rej_summary.get("syntax_error", 0)
            rej_list = resp.get("rejected_candidates", [])
            rej_count = len(rej_list) if rej_list else (raw_cands - len(acc_list) if raw_cands >= len(acc_list) else 0)

        tot_raw += raw_cands
        tot_accepted += len(acc_list)
        tot_rejected += rej_count
        tot_dups += dups
        tot_mismatch += mismatch
        tot_unans += unans
        tot_qual += qual

        # Verify provenance and grounding for accepted questions
        prov_ok = True
        ground_ok = True
        q_types: List[str] = []
        accepted_records = []

        for q in acc_list:
            if isinstance(q, GeneratedQuestion):
                q_dict = q.model_dump()
            else:
                q_dict = q

            q_type = q_dict.get("question_type", "factoid")
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
            q_types.append(q_type)

            # Check provenance fields (support both top-level Phase 22 schema and nested dict)
            prov = q_dict.get("provenance") or {}
            chunk_id_val = q_dict.get("chunk_id") or prov.get("chunk_id")
            page_num_val = q_dict.get("page_number") if q_dict.get("page_number") is not None else prov.get("page_number")
            doc_id_val = q_dict.get("document_id") or prov.get("document_id") or did

            if not chunk_id_val or page_num_val is None or not doc_id_val:
                prov_ok = False

            # Check grounding: answer must appear in or align with text
            ans_val = q_dict.get("answer") or q_dict.get("expected_answer", "")
            if ans_val and text and ans_val.lower() not in text.lower():
                # Check token overlap
                ans_tokens = set(ans_val.lower().split())
                text_tokens = set(text.lower().split())
                if not ans_tokens.intersection(text_tokens):
                    ground_ok = False

            accepted_records.append(q_dict)

        if acc_list:
            if prov_ok:
                provenance_complete_count += len(acc_list)
            if ground_ok:
                grounding_valid_count += len(acc_list)

        chunk_evals.append(
            QuestionGenerationEvaluationItem(
                chunk_id=cid,
                document_id=did,
                page_number=pnum,
                raw_candidates_count=raw_cands,
                accepted_count=len(acc_list),
                rejected_count=rej_count,
                duplicates_count=dups,
                answer_mismatches_count=mismatch,
                unanswerable_count=unans,
                quality_failures_count=qual,
                all_provenance_complete=prov_ok,
                all_grounded_in_source=ground_ok,
                question_types=q_types,
                accepted_questions=accepted_records,
            )
        )

    # Compute macro rates
    n_chunks = len(generation_results)
    acc_rate = round(float(tot_accepted) / float(tot_raw), 4) if tot_raw > 0 else 0.0
    rej_rate = round(float(tot_rejected) / float(tot_raw), 4) if tot_raw > 0 else 0.0

    dup_rate = round(float(tot_dups) / float(tot_raw), 4) if tot_raw > 0 else 0.0
    mismatch_rate = round(float(tot_mismatch) / float(tot_raw), 4) if tot_raw > 0 else 0.0
    unans_rate = round(float(tot_unans) / float(tot_raw), 4) if tot_raw > 0 else 0.0
    qual_rate = round(float(tot_qual) / float(tot_raw), 4) if tot_raw > 0 else 0.0

    prov_rate = round(float(provenance_complete_count) / float(tot_accepted), 4) if tot_accepted > 0 else 1.0
    ground_rate = round(float(grounding_valid_count) / float(tot_accepted), 4) if tot_accepted > 0 else 1.0

    # Status verdict
    if prov_rate >= 0.99 and ground_rate >= 0.95 and tot_accepted > 0:
        overall = "PASS"
    elif tot_accepted > 0:
        overall = "WARNING"
    else:
        overall = "FAIL"

    return QuestionGenerationBenchmarkReport(
        benchmark_name=benchmark_name,
        total_chunks_evaluated=n_chunks,
        total_pages_covered=len(pages_seen),
        total_raw_candidates=tot_raw,
        total_accepted_questions=tot_accepted,
        total_rejected_questions=tot_rejected,
        acceptance_rate=acc_rate,
        rejection_rate=rej_rate,
        duplicate_rate=dup_rate,
        answer_mismatch_rate=mismatch_rate,
        unanswerable_rate=unans_rate,
        quality_failure_rate=qual_rate,
        provenance_completeness_rate=prov_rate,
        grounding_validity_rate=ground_rate,
        question_type_distribution=type_counts,
        overall_status=overall,
        per_chunk_evaluations=chunk_evals,
    )
