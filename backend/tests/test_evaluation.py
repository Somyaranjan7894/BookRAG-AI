"""Comprehensive test suite for Phase 20 Production RAG Quality & Evaluation.

Tests:
1. Golden dataset schema validation (golden_v1.json, dev_sample.json, custom datasets)
2. Verification status handling (VERIFIED, NOT_VERIFIED, BLOCKED)
3. Question category validation (direct_fact, definition, explanation, multi_page, comparison, numerical_fact, unanswerable, ambiguous)
4. Answerable vs unanswerable logical consistency
5. Retrieval metric calculations (Recall@1,3,5,10, Precision@1,3,5, MRR, deduplication, edge cases)
6. Page-level retrieval evaluation
7. Reranking comparative evaluation (deltas, movement classification, scientific neutrality)
8. Answer quality and token F1 overlap evaluation
9. Grounding and claim support rate evaluation
10. Refusal performance and safe refusal verification
11. Citation correctness (precision, recall, clean isolation on unanswerable)
12. Report generation (JSON and Markdown formats with verification status and semantic disclaimers)
"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from pydantic import ValidationError

from evaluation.datasets.schema import (
    EvaluationDataset,
    EvaluationItem,
    VerificationStatus,
)
from evaluation.retrieval.metrics import (
    AggregatedRetrievalMetrics,
    QueryRetrievalResult,
    aggregate_retrieval_metrics,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    deduplicate_preserving_order,
    evaluate_query_retrieval,
)
from evaluation.retrieval.reranking_eval import (
    RerankingComparisonReport,
    compare_retrieval_stages,
)
from evaluation.grounding.evaluator import (
    SEMANTIC_DISCLAIMER,
    AggregatedGroundingMetrics,
    QueryGroundingResult,
    aggregate_grounding_metrics,
    compute_token_overlap_f1,
    evaluate_grounded_response,
    normalize_answer_text,
)
from evaluation.runner import generate_markdown_summary, load_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Suite 1: Golden Benchmark & Schema Validation Tests
# ============================================================================

class TestGoldenBenchmarkSchema:
    """Validate golden dataset schema, items, and verification status constraints."""

    def test_load_golden_v1_json(self):
        """Ensure golden_v1.json exists, loads cleanly, and satisfies schema rules."""
        path = REPO_ROOT / "evaluation" / "datasets" / "golden_v1.json"
        assert path.exists(), "golden_v1.json must exist in evaluation/datasets/"

        dataset = load_dataset(path)
        assert dataset.name == "Machine Learning & Neural Networks Golden Benchmark"
        assert dataset.version == "1.0.0"
        assert dataset.dataset_type == "golden"
        # Golden dataset is formally verified against doc_ai_handbook_01 source text
        assert dataset.verification_status == "VERIFIED"
        assert dataset.total_count == 10
        assert dataset.answerable_count == 7
        assert dataset.unanswerable_count == 3

    def test_load_dev_sample_json_backwards_compatibility(self):
        """Ensure legacy dev_sample.json continues to load without errors."""
        path = REPO_ROOT / "evaluation" / "datasets" / "dev_sample.json"
        assert path.exists(), "dev_sample.json must be preserved"

        dataset = load_dataset(path)
        assert dataset.name == "Machine Learning & Neural Networks Dev Benchmark"
        assert dataset.total_count == 10
        assert dataset.answerable_count == 7

    def test_golden_dataset_category_coverage(self):
        """Verify that golden_v1.json covers the required question categories."""
        path = REPO_ROOT / "evaluation" / "datasets" / "golden_v1.json"
        dataset = load_dataset(path)
        categories = set(dataset.category_counts.keys())

        required_categories = {
            "direct_fact",
            "definition",
            "explanation",
            "multi_page",
            "comparison",
            "numerical_fact",
            "unanswerable",
            "ambiguous",
        }
        for req in required_categories:
            assert req in categories, f"Expected category '{req}' to be represented in golden_v1.json"

    def test_answerable_items_have_evidence_and_ground_truth(self):
        """Verify that all answerable items have expected_answer and non-empty evidence."""
        path = REPO_ROOT / "evaluation" / "datasets" / "golden_v1.json"
        dataset = load_dataset(path)

        for item in dataset.items:
            if item.answerable:
                assert item.expected_answer is not None and len(item.expected_answer) > 0
                assert len(item.relevant_page_numbers) > 0
                assert len(item.relevant_chunk_ids) > 0
                assert item.evidence is not None and len(item.evidence) > 0
            else:
                assert item.expected_answer is None
                assert len(item.relevant_page_numbers) == 0
                assert len(item.relevant_chunk_ids) == 0

    def test_unanswerable_with_chunk_ids_raises_validation_error(self):
        """Verify that marking an item unanswerable while supplying chunk IDs raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationItem(
                id="invalid_q",
                question="What is this?",
                document_id="doc_1",
                answerable=False,
                category="unanswerable",
                relevant_chunk_ids=["chunk_001"],  # Contradiction
            )

    def test_verification_status_distinctions(self):
        """Verify VERIFIED, NOT_VERIFIED, and BLOCKED statuses are accepted, but invalid strings rejected."""
        valid_statuses = ["VERIFIED", "NOT_VERIFIED", "BLOCKED"]
        for st in valid_statuses:
            ds = EvaluationDataset(
                name="Test DS",
                version="1.0.0",
                verification_status=st,  # type: ignore
                description="Test description",
                document_id="doc_1",
                document_title="Test Doc",
                items=[],
            )
            assert ds.verification_status == st

        with pytest.raises(ValidationError):
            EvaluationDataset(
                name="Invalid DS",
                version="1.0.0",
                verification_status="FABRICATED_LABEL",  # type: ignore
                description="Invalid",
                document_id="doc_1",
                document_title="Test Doc",
                items=[],
            )


# ============================================================================
# Suite 2: Retrieval Evaluation Metrics Tests
# ============================================================================

class TestRetrievalMetrics:
    """Verify deterministic ranking and retrieval metric calculations."""

    def test_compute_recall_at_k(self):
        relevant = {"c1", "c2", "c3"}
        retrieved = ["c1", "c4", "c2", "c5", "c6"]

        assert compute_recall_at_k(relevant, retrieved, 1) == pytest.approx(1.0 / 3.0, rel=1e-3)
        assert compute_recall_at_k(relevant, retrieved, 3) == pytest.approx(2.0 / 3.0, rel=1e-3)
        assert compute_recall_at_k(relevant, retrieved, 5) == pytest.approx(2.0 / 3.0, rel=1e-3)
        assert compute_recall_at_k(relevant, retrieved, 10) == pytest.approx(2.0 / 3.0, rel=1e-3)

    def test_compute_precision_at_k(self):
        relevant = {"c1", "c2"}
        retrieved = ["c1", "c4", "c2", "c5", "c6"]

        assert compute_precision_at_k(relevant, retrieved, 1) == 1.0
        assert compute_precision_at_k(relevant, retrieved, 3) == pytest.approx(2.0 / 3.0, rel=1e-3)
        assert compute_precision_at_k(relevant, retrieved, 5) == pytest.approx(2.0 / 5.0, rel=1e-3)

    def test_compute_reciprocal_rank(self):
        relevant = {"c3"}
        retrieved = ["c1", "c2", "c3", "c4"]
        rr, rank = compute_reciprocal_rank(relevant, retrieved)
        assert rr == pytest.approx(1.0 / 3.0, rel=1e-3)
        assert rank == 3

        # First rank
        rr1, rank1 = compute_reciprocal_rank({"c1"}, retrieved)
        assert rr1 == 1.0
        assert rank1 == 1

        # Not found
        rr_none, rank_none = compute_reciprocal_rank({"c999"}, retrieved)
        assert rr_none == 0.0
        assert rank_none is None

    def test_deduplicate_preserving_order(self):
        raw = ["c1", "c2", "c1", "c3", "c2", "c4"]
        deduped = deduplicate_preserving_order(raw)
        assert deduped == ["c1", "c2", "c3", "c4"]

    def test_evaluate_query_retrieval_edge_cases(self):
        # Empty retrieved
        res_empty = evaluate_query_retrieval("q1", "Empty?", ["c1"], [])
        assert res_empty.reciprocal_rank == 0.0
        assert res_empty.recall_at_k[1] == 0.0
        assert res_empty.precision_at_k[1] == 0.0
        assert res_empty.first_relevant_rank is None

        # Unanswerable query (no ground truth relevant chunks)
        res_unans = evaluate_query_retrieval("q2", "Unanswerable?", [], ["c1", "c2"])
        assert res_unans.num_relevant == 0
        assert res_unans.reciprocal_rank == 0.0
        assert res_unans.recall_at_k[1] == 0.0

    def test_page_level_retrieval_metrics(self):
        relevant_chunks = ["c1"]
        retrieved_chunks = ["c2", "c1"]
        relevant_pages = [1]
        retrieved_pages = [2, 1]

        res = evaluate_query_retrieval(
            query_id="q1",
            query="Test query",
            relevant_ids=relevant_chunks,
            retrieved_ids=retrieved_chunks,
            relevant_pages=relevant_pages,
            retrieved_pages=retrieved_pages,
        )
        assert res.reciprocal_rank == 0.5
        assert res.first_relevant_rank == 2
        assert res.page_reciprocal_rank == 0.5
        assert res.first_relevant_page_rank == 2
        assert res.page_recall_at_k is not None
        assert res.page_recall_at_k[1] == 0.0
        assert res.page_recall_at_k[3] == 1.0

    def test_aggregate_retrieval_metrics(self):
        q1 = evaluate_query_retrieval("q1", "Q1", ["c1"], ["c1", "c2"])  # RR=1.0, R@1=1.0, P@1=1.0
        q2 = evaluate_query_retrieval("q2", "Q2", ["c2"], ["c1", "c2"])  # RR=0.5, R@1=0.0, P@1=0.0
        q3 = evaluate_query_retrieval("q3", "Q3", [], ["c1", "c2"])      # Unanswerable, excluded from MRR

        agg = aggregate_retrieval_metrics([q1, q2, q3], k_values=(1, 3, 5, 10))
        assert agg.total_queries == 3
        assert agg.evaluated_queries == 2
        assert agg.unanswerable_queries == 1
        assert agg.mrr == pytest.approx(0.75, rel=1e-3)
        assert agg.mean_recall_at_k[1] == pytest.approx(0.5, rel=1e-3)
        assert agg.mean_precision_at_k[1] == pytest.approx(0.5, rel=1e-3)


# ============================================================================
# Suite 3: Reranking Comparison Tests
# ============================================================================

class TestRerankingComparison:
    """Verify Stage 1 vs Stage 2 reranking comparative evaluation."""

    def test_reranking_improvement_detection(self):
        query_items = [
            {"id": "q1", "question": "Q1", "relevant_chunk_ids": ["c1"]},
            {"id": "q2", "question": "Q2", "relevant_chunk_ids": ["c2"]},
        ]
        # In Stage 1: q1 item at rank 2, q2 item at rank 1
        s1 = {"q1": ["c9", "c1"], "q2": ["c2", "c9"]}
        # In Stage 2: q1 item promoted to rank 1, q2 unchanged
        s2 = {"q1": ["c1", "c9"], "q2": ["c2", "c9"]}

        report = compare_retrieval_stages("TestDS", query_items, s1, s2)
        assert report.improved_queries_count == 1
        assert report.unchanged_queries_count == 1
        assert report.degraded_queries_count == 0
        assert report.delta_mrr > 0
        assert report.empirical_effect == "IMPROVED"

    def test_reranking_degradation_detection(self):
        query_items = [
            {"id": "q1", "question": "Q1", "relevant_chunk_ids": ["c1"]},
        ]
        # Stage 1 has c1 at rank 1, Stage 2 pushes c1 down to rank 2
        s1 = {"q1": ["c1", "c2"]}
        s2 = {"q1": ["c2", "c1"]}

        report = compare_retrieval_stages("TestDS", query_items, s1, s2)
        assert report.improved_queries_count == 0
        assert report.degraded_queries_count == 1
        assert report.delta_mrr < 0
        assert report.empirical_effect == "DEGRADED"

    def test_reranking_scientific_neutrality_preserved(self):
        """Ensure reranking report does not claim improvement when delta is zero or negative."""
        query_items = [
            {"id": "q1", "question": "Q1", "relevant_chunk_ids": ["c1"]},
        ]
        s1 = {"q1": ["c1"]}
        s2 = {"q1": ["c1"]}

        report = compare_retrieval_stages("TestDS", query_items, s1, s2)
        assert report.delta_mrr == 0.0
        assert report.empirical_effect == "UNCHANGED"
        assert "reranker improvements are never claimed unless demonstrated" in report.scientific_statement


# ============================================================================
# Suite 4: Answer Quality, Grounding, and Refusal Tests
# ============================================================================

class TestAnswerQualityAndRefusal:
    """Verify answer correctness, claim support, refusal, and citation metrics."""

    def test_compute_token_overlap_f1(self):
        pred = "The backpropagation algorithm computes gradients using the multivariate chain rule."
        gold = "Backpropagation computes gradients by applying the multivariate chain rule across layers."

        prec, rec, f1 = compute_token_overlap_f1(pred, gold)
        assert prec > 0.5
        assert rec > 0.5
        assert f1 > 0.5

        # Disjoint text
        p_zero, r_zero, f1_zero = compute_token_overlap_f1("completely unrelated text", "alpha beta gamma")
        assert f1_zero == 0.0

    def test_evaluate_grounded_response_answerable_success(self):
        res = evaluate_grounded_response(
            query_id="q1",
            query="How does backprop work?",
            is_answerable_gold=True,
            grounding_status="grounded",
            generated_answer="Backpropagation computes gradients using the chain rule.",
            expected_answer="Backpropagation computes gradients using the multivariate chain rule.",
            category="direct_fact",
            claims_results=[
                {"claim_text": "Backprop computes gradients.", "status": "entailed"},
                {"claim_text": "It uses the chain rule.", "status": "entailed"},
            ],
            citations_list=[{"chunk_id": "c1", "page_number": 2}],
            relevant_chunk_ids=["c1"],
            relevant_page_numbers=[2],
        )
        assert res.claim_support_rate == 1.0
        assert res.refusal_triggered is False
        assert res.refusal_correct is True
        assert res.is_answerable_success is True
        assert res.answer_correctness_f1 is not None and res.answer_correctness_f1 > 0.5
        assert res.citation_precision == 1.0
        assert res.citation_recall == 1.0

    def test_evaluate_grounded_response_safe_refusal(self):
        """Unanswerable query correctly refused with insufficient_evidence status."""
        res = evaluate_grounded_response(
            query_id="q_unans",
            query="How to bake sourdough bread?",
            is_answerable_gold=False,
            grounding_status="insufficient_evidence",
            generated_answer="The provided document does not contain information on sourdough bread.",
            expected_answer=None,
            category="unanswerable",
            claims_results=[],
            citations_list=[],
            relevant_chunk_ids=[],
            relevant_page_numbers=[],
        )
        assert res.refusal_triggered is True
        assert res.refusal_correct is True
        assert res.is_incorrect_answer_to_unanswerable is False
        assert res.is_false_refusal is False
        assert res.citation_precision == 1.0  # Clean isolation: no false citations

    def test_evaluate_grounded_response_unsafe_answer_to_unanswerable(self):
        """Unanswerable query where model hallucinated an answer and cited text."""
        res = evaluate_grounded_response(
            query_id="q_unans",
            query="How to bake sourdough bread?",
            is_answerable_gold=False,
            grounding_status="grounded",
            generated_answer="Sourdough bread uses a 70% hydration ratio and flour.",
            expected_answer=None,
            category="unanswerable",
            claims_results=[{"claim_text": "Sourdough uses hydration.", "status": "entailed"}],
            citations_list=[{"chunk_id": "irrelevant_chunk", "page_number": 1}],
            relevant_chunk_ids=[],
            relevant_page_numbers=[],
        )
        assert res.refusal_triggered is False
        assert res.refusal_correct is False
        assert res.is_incorrect_answer_to_unanswerable is True
        assert res.citation_precision == 0.0  # Hallucinated citation on unanswerable query

    def test_aggregate_grounding_metrics(self):
        r_ans = evaluate_grounded_response(
            query_id="q1",
            query="Q1",
            is_answerable_gold=True,
            grounding_status="grounded",
            generated_answer="Answer text here.",
            expected_answer="Answer text here.",
            claims_results=[{"claim_text": "Claim 1", "status": "entailed"}],
            citations_list=[{"chunk_id": "c1", "page_number": 1}],
            relevant_chunk_ids=["c1"],
        )
        r_unans = evaluate_grounded_response(
            query_id="q2",
            query="Q2",
            is_answerable_gold=False,
            grounding_status="insufficient_evidence",
            generated_answer="Cannot answer.",
            claims_results=[],
            citations_list=[],
        )

        agg = aggregate_grounding_metrics(
            dataset_name="Test Golden",
            results=[r_ans, r_unans],
            verification_status="NOT_VERIFIED",
        )
        assert agg.total_questions == 2
        assert agg.answerable_questions == 1
        assert agg.unanswerable_questions == 1
        assert agg.unanswerable_refusal_count == 1
        assert agg.incorrect_unanswerable_answers == 0
        assert agg.refusal_precision == 1.0
        assert agg.over_refusal_rate == 0.0
        assert agg.answerable_success_rate == 1.0
        assert agg.verification_status == "NOT_VERIFIED"
        assert agg.mean_answer_correctness_f1 == 1.0
        assert agg.mean_citation_precision == 1.0


# ============================================================================
# Suite 5: Report Generation and Semantic Integrity Tests
# ============================================================================

class TestReportGeneration:
    """Verify markdown summary and report generation adhere to scientific integrity."""

    def test_generate_markdown_summary_includes_status_and_disclaimers(self):
        dataset_path = REPO_ROOT / "evaluation" / "datasets" / "golden_v1.json"
        dataset = load_dataset(dataset_path)

        query_items = [item.model_dump() for item in dataset.items]
        s1_results = {item.id: item.relevant_chunk_ids for item in dataset.items}
        s2_results = {item.id: item.relevant_chunk_ids for item in dataset.items}

        comparison = compare_retrieval_stages(dataset.name, query_items, s1_results, s2_results)

        grounding_results = []
        for item in dataset.items:
            q_res = evaluate_grounded_response(
                query_id=item.id,
                query=item.question,
                is_answerable_gold=item.answerable,
                grounding_status="grounded" if item.answerable else "insufficient_evidence",
                generated_answer=item.expected_answer if item.answerable else "Cannot answer.",
                expected_answer=item.expected_answer,
                category=item.category,
                claims_results=[{"claim_text": "Sample claim", "status": "entailed"}] if item.answerable else [],
                citations_list=[{"chunk_id": item.relevant_chunk_ids[0], "page_number": item.relevant_page_numbers[0]}] if item.answerable and item.relevant_chunk_ids else [],
                relevant_chunk_ids=item.relevant_chunk_ids,
                relevant_page_numbers=item.relevant_page_numbers,
            )
            grounding_results.append(q_res)

        grounding_report = aggregate_grounding_metrics(
            dataset_name=dataset.name,
            results=grounding_results,
            verification_status=dataset.verification_status,
        )

        md = generate_markdown_summary(
            dataset=dataset,
            comparison_report=comparison,
            grounding_report=grounding_report,
            avg_retrieval_ms=12.5,
            avg_rerank_ms=145.0,
            avg_gen_ms=3200.0,
        )

        # Check required sections and assertions
        assert "BookRAG AI — Phase 20 Production RAG Quality & Evaluation Report" in md
        assert "VERIFIED" in md
        assert "Question Category Distribution" in md
        assert "Retrieval & Reranking Comparative Evaluation" in md
        assert "Mean Reciprocal Rank (MRR)" in md
        assert "Recall@1" in md
        assert "Precision@1" in md
        assert "Answer Quality & Grounding Evaluation" in md
        assert "Refusal Evaluation & Hallucination Defense" in md
        assert "Measured Production Latencies" in md
        # Check critical semantic disclaimers
        assert "Retrieval relevance != factual correctness" in md
        assert "Reranker score != probability" in md
        assert "NLI score != truth probability" in md
        assert "Model confidence != factual correctness" in md
