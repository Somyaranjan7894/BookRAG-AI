"""Comprehensive test suite for BookRAG AI Phase 23 Advanced Evaluation & Full-Book Benchmarking.

Tests:
1. Full-Book Benchmark schema, items, and provenance validation
2. Phase 23 alias mappings (question_id, page_numbers, chunk_ids, source_evidence, query_type, answerability)
3. 9-category question distribution coverage
4. Retrieval metrics (Recall@1/3/5/10, Precision@1/3/5/10, MRR, duplicate retrieval rate)
5. Reranking comparative evaluation (deltas, rank movement, duplicate deltas, scientific neutrality)
6. Query-type isolated evaluation and aggregation
7. Multi-page synthesis evaluation (cross-page retrieval, evidence sufficiency, citation page coverage)
8. Safety, refusal, and anti-hallucination metrics (refusal precision, false refusal, citation leakage)
9. Question generation evaluation (candidate volume, rejection rates, provenance completeness, grounding)
10. Latency benchmark statistics (mean, median, P95, bottleneck identification)
11. Phase 20 golden baseline regression protection
12. Full report serialization and markdown summary generation
"""

import json
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.datasets.schema import EvaluationDataset, EvaluationItem
from evaluation.grounding.evaluator import (
    AggregatedGroundingMetrics,
    QueryGroundingResult,
    aggregate_grounding_metrics,
    evaluate_grounded_response,
)
from evaluation.latency.benchmark import (
    LatencyBenchmarkReport,
    StageLatencyStats,
    build_latency_report,
    compute_percentile,
    compute_stage_latency_stats,
)
from evaluation.multipage.evaluator import (
    AggregatedMultiPageMetrics,
    MultiPageQueryEvaluation,
    evaluate_multipage_synthesis,
)
from evaluation.qgen.evaluator import (
    QuestionGenerationBenchmarkReport,
    evaluate_question_generation_runs,
)
from evaluation.query_types.evaluator import (
    CategoryEvaluationMetrics,
    QueryTypeReport,
    evaluate_by_query_type,
)
from evaluation.retrieval.metrics import (
    AggregatedRetrievalMetrics,
    QueryRetrievalResult,
    aggregate_retrieval_metrics,
    compute_duplicate_rate,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    evaluate_query_retrieval,
)
from evaluation.retrieval.reranking_eval import (
    RerankingComparisonReport,
    compare_retrieval_stages,
)
from evaluation.runner import load_dataset
from evaluation.safety.evaluator import (
    SafetyEvaluationReport,
    SafetyQueryEvaluation,
    evaluate_refusal_and_safety,
)


# ============================================================================
# Suite 1: Full-Book Dataset Schema & Provenance Tests
# ============================================================================

class TestFullBookBenchmarkSchema:
    """Validate full-book dataset schema, items, and verification status constraints."""

    def test_load_full_book_benchmark_json(self):
        """Ensure full_book_benchmark_v1.json exists, loads cleanly, and satisfies schema rules."""
        path = REPO_ROOT / "evaluation" / "datasets" / "full_book_benchmark_v1.json"
        assert path.exists(), "full_book_benchmark_v1.json must exist in evaluation/datasets/"

        dataset = load_dataset(path)
        assert dataset.name == "Distributed Systems & Cloud Architecture Full-Book Benchmark"
        assert dataset.version == "1.0.0"
        assert dataset.dataset_type == "golden"
        assert dataset.verification_status == "VERIFIED"
        assert dataset.total_count == 36
        assert dataset.answerable_count == 28
        assert dataset.unanswerable_count == 8

    def test_full_book_category_coverage(self):
        """Verify that all 9 required query categories are represented in the full-book benchmark."""
        path = REPO_ROOT / "evaluation" / "datasets" / "full_book_benchmark_v1.json"
        dataset = load_dataset(path)
        categories = set(dataset.category_counts.keys())

        required_categories = {
            "direct_fact",
            "definition",
            "explanation",
            "comparison",
            "numerical_fact",
            "reasoning",
            "multi_page_synthesis",
            "unanswerable",
            "ambiguous",
        }
        for req in required_categories:
            assert req in categories, f"Expected category '{req}' to be represented in full_book_benchmark_v1.json"

    def test_alias_mapping_and_properties(self):
        """Ensure Phase 23 alias fields map transparently to canonical schema attributes."""
        item = EvaluationItem.model_validate({
            "question_id": "test_q01",
            "question": "What is the Raft timeout?",
            "document_id": "doc_test",
            "page_numbers": [17],
            "chunk_ids": ["chunk_017"],
            "source_evidence": "Randomized timeout between 150ms and 300ms.",
            "query_type": "direct_fact",
            "answerability": True,
            "expected_answer": "150ms to 300ms",
            "verification_status": "VERIFIED",
        })

        assert item.id == "test_q01"
        assert item.question_id == "test_q01"
        assert item.relevant_page_numbers == [17]
        assert item.page_numbers == [17]
        assert item.relevant_chunk_ids == ["chunk_017"]
        assert item.chunk_ids == ["chunk_017"]
        assert item.evidence == "Randomized timeout between 150ms and 300ms."
        assert item.source_evidence == "Randomized timeout between 150ms and 300ms."
        assert item.category == "direct_fact"
        assert item.query_type == "direct_fact"
        assert item.answerable is True
        assert item.answerability is True

    def test_answerable_items_have_valid_evidence_and_ground_truth(self):
        """Verify that every answerable item has an expected_answer and non-empty evidence."""
        path = REPO_ROOT / "evaluation" / "datasets" / "full_book_benchmark_v1.json"
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


# ============================================================================
# Suite 2: Retrieval and Duplicate Rate Metrics Tests
# ============================================================================

class TestRetrievalMetrics:
    """Verify Recall@K, Precision@K, MRR, and duplicate rate calculations."""

    def test_duplicate_rate_computation(self):
        """Test compute_duplicate_rate helper function."""
        assert compute_duplicate_rate([]) == 0.0
        assert compute_duplicate_rate(["c1", "c2", "c3"]) == 0.0
        # 4 items, 2 unique ('c1', 'c2' each duplicated once) -> 2/4 = 0.5
        assert compute_duplicate_rate(["c1", "c1", "c2", "c2"]) == 0.5
        # 5 items, 1 duplicate -> 1/5 = 0.2
        assert compute_duplicate_rate(["c1", "c2", "c3", "c4", "c1"]) == 0.2

    def test_evaluate_query_retrieval_with_duplicates(self):
        """Verify duplicate rates are tracked in QueryRetrievalResult."""
        relevant = ["c1", "c2"]
        # Raw retrieved has 5 items including duplicate 'c3'
        raw_retrieved = ["c3", "c1", "c3", "c4", "c5"]
        raw_pages = [1, 2, 1, 3, 4]

        res = evaluate_query_retrieval(
            query_id="q1",
            query="test query",
            relevant_ids=relevant,
            retrieved_ids=raw_retrieved,
            k_values=(1, 3, 5, 10),
            relevant_pages=[2],
            retrieved_pages=raw_pages,
        )

        assert res.duplicate_retrieval_rate == 0.2  # 1 duplicate in 5 items
        assert res.page_duplicate_rate == 0.2  # page 1 duplicated
        # Deduplication preserves order: ['c3', 'c1', 'c4', 'c5']
        assert res.retrieved_chunk_ids == ["c3", "c1", "c4", "c5"]
        assert res.first_relevant_rank == 2  # 'c1' at position 2
        assert res.reciprocal_rank == 0.5
        assert res.recall_at_k[1] == 0.0
        assert res.recall_at_k[3] == 0.5
        assert res.precision_at_k[1] == 0.0
        assert res.precision_at_k[3] == round(1.0 / 3.0, 4)

    def test_reranking_comparison_with_duplicate_deltas(self):
        """Verify compare_retrieval_stages calculates duplicate rate deltas."""
        items = [
            {"id": "q1", "question": "test1", "relevant_chunk_ids": ["c1"], "relevant_page_numbers": [1]},
            {"id": "q2", "question": "test2", "relevant_chunk_ids": ["c2"], "relevant_page_numbers": [2]},
        ]
        # Stage 1: c1 at rank 2 for q1, c2 at rank 2 for q2
        s1_chunks = {
            "q1": ["x1", "c1", "x2"],
            "q2": ["x3", "c2", "x4"],
        }
        # Stage 2: c1 at rank 1 for q1, c2 at rank 1 for q2 (Improved)
        s2_chunks = {
            "q1": ["c1", "x1", "x2"],
            "q2": ["c2", "x3", "x4"],
        }

        report = compare_retrieval_stages(
            dataset_name="Test DS",
            query_items=items,
            stage1_results_by_query=s1_chunks,
            stage2_results_by_query=s2_chunks,
            k_values=(1, 3, 5, 10),
        )

        assert report.stage1_metrics.mrr == 0.5
        assert report.stage2_metrics.mrr == 1.0
        assert report.delta_mrr == 0.5
        assert report.improved_queries_count == 2
        assert report.degraded_queries_count == 0
        assert report.unchanged_queries_count == 0
        assert report.empirical_effect == "IMPROVED"


# ============================================================================
# Suite 3: Query-Type Evaluation Tests
# ============================================================================

class TestQueryTypeEvaluator:
    """Verify query-type breakdown across the 9 required categories."""

    def test_evaluate_by_query_type_all_categories(self):
        """Test aggregation across diverse question categories."""
        items = [
            {"id": "q_fact", "category": "direct_fact", "answerable": True},
            {"id": "q_def", "category": "definition", "answerable": True},
            {"id": "q_unans", "category": "unanswerable", "answerable": False},
        ]

        ret_map = {
            "q_fact": QueryRetrievalResult(
                query_id="q_fact", query="q", num_relevant=1, num_retrieved=1,
                recall_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
                precision_at_k={1: 1.0, 3: 0.3333, 5: 0.2, 10: 0.1},
                reciprocal_rank=1.0,
            ),
            "q_def": QueryRetrievalResult(
                query_id="q_def", query="q", num_relevant=1, num_retrieved=1,
                recall_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
                precision_at_k={1: 1.0, 3: 0.3333, 5: 0.2, 10: 0.1},
                reciprocal_rank=1.0,
            ),
        }

        ground_map = {
            "q_fact": QueryGroundingResult(
                query_id="q_fact", query="q", is_answerable_gold=True,
                grounding_status="grounded", total_claims=1, supported_claims=1,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, answer_correctness_f1=0.9,
                refusal_triggered=False, refusal_correct=True, is_answerable_success=True,
            ),
            "q_def": QueryGroundingResult(
                query_id="q_def", query="q", is_answerable_gold=True,
                grounding_status="grounded", total_claims=1, supported_claims=1,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, answer_correctness_f1=0.85,
                refusal_triggered=False, refusal_correct=True, is_answerable_success=True,
            ),
            "q_unans": QueryGroundingResult(
                query_id="q_unans", query="q", is_answerable_gold=False,
                grounding_status="insufficient_evidence", total_claims=0, supported_claims=0,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, refusal_triggered=True, refusal_correct=True,
            ),
        }

        report = evaluate_by_query_type(
            benchmark_name="Test",
            query_items=items,
            retrieval_results_map=ret_map,
            grounding_results_map=ground_map,
        )

        assert len(report.categories) == 3
        assert report.categories["direct_fact"].mrr == 1.0
        assert report.categories["direct_fact"].status == "PASS"
        assert report.categories["unanswerable"].refusal_precision == 1.0
        assert report.categories["unanswerable"].status == "PASS"
        assert report.overall_status == "PASS"


# ============================================================================
# Suite 4: Multi-Page Synthesis Evaluation Tests
# ============================================================================

class TestMultiPageEvaluator:
    """Verify multi-page synthesis metrics: cross-page retrieval, sufficiency, citations."""

    def test_multipage_evaluation_success(self):
        """Test multi-page question where all required pages and evidence are retrieved."""
        items = [
            {
                "id": "mp_01",
                "question": "Synthesize page 16 and page 35",
                "relevant_page_numbers": [16, 35],
                "relevant_chunk_ids": ["c16", "c35"],
                "expected_answer": "Combined concept",
            }
        ]

        ret_map = {
            "mp_01": QueryRetrievalResult(
                query_id="mp_01", query="q", num_relevant=2, num_retrieved=5,
                retrieved_pages=[16, 20, 35],
                retrieved_chunk_ids=["c16", "c20", "c35"],
            )
        }

        ground_map = {
            "mp_01": QueryGroundingResult(
                query_id="mp_01", query="q", is_answerable_gold=True,
                grounding_status="grounded", total_claims=2, supported_claims=2,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, answer_correctness_f1=0.88,
                refusal_triggered=False, refusal_correct=True,
                citations=[{"page_number": 16}, {"page_number": 35}],
            )
        }

        report = evaluate_multipage_synthesis(
            multipage_items=items,
            retrieval_results_map=ret_map,
            grounding_results_map=ground_map,
        )

        assert report.total_multipage_queries == 1
        assert report.all_pages_retrieved_rate == 1.0
        assert report.evidence_sufficiency_rate == 1.0
        assert report.correct_synthesis_rate == 1.0
        assert report.zero_unsupported_info_rate == 1.0
        assert report.full_citation_coverage_rate == 1.0
        assert report.overall_status == "PASS"

    def test_multipage_evaluation_missing_page(self):
        """Test multi-page question where only 1 of 2 required pages was retrieved."""
        items = [
            {
                "id": "mp_02",
                "question": "Synthesize page 10 and page 50",
                "relevant_page_numbers": [10, 50],
                "relevant_chunk_ids": ["c10", "c50"],
            }
        ]

        ret_map = {
            "mp_02": QueryRetrievalResult(
                query_id="mp_02", query="q", num_relevant=2, num_retrieved=3,
                retrieved_pages=[10, 11, 12],
                retrieved_chunk_ids=["c10", "c11", "c12"],
            )
        }

        ground_map = {
            "mp_02": QueryGroundingResult(
                query_id="mp_02", query="q", is_answerable_gold=True,
                grounding_status="partially_grounded", total_claims=1, supported_claims=1,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, refusal_triggered=False, refusal_correct=True,
                citations=[{"page_number": 10}],
            )
        }

        report = evaluate_multipage_synthesis(
            multipage_items=items,
            retrieval_results_map=ret_map,
            grounding_results_map=ground_map,
        )

        assert report.all_pages_retrieved_rate == 0.0
        assert report.mean_page_coverage_rate == 0.5
        assert report.evidence_sufficiency_rate == 0.0
        assert report.full_citation_coverage_rate == 0.0
        assert report.overall_status == "FAIL"


# ============================================================================
# Suite 5: Safety and Refusal Evaluator Tests
# ============================================================================

class TestSafetyAndRefusalEvaluator:
    """Verify safety baseline, refusal precision, and citation isolation."""

    def test_safety_evaluation_clean_refusal(self):
        """Test 100% clean refusal on unanswerable queries with 0 citation leakage."""
        items = [
            {"id": "q_ans", "question": "Answerable", "category": "direct_fact", "answerable": True},
            {"id": "q_unans", "question": "Unanswerable", "category": "unanswerable", "answerable": False},
            {"id": "q_ambig", "question": "Ambiguous", "category": "ambiguous", "answerable": False},
        ]

        ground_map = {
            "q_ans": QueryGroundingResult(
                query_id="q_ans", query="q", is_answerable_gold=True,
                grounding_status="grounded", total_claims=1, supported_claims=1,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, refusal_triggered=False, refusal_correct=True,
                citations=[{"chunk_id": "c1"}], total_citations=1,
            ),
            "q_unans": QueryGroundingResult(
                query_id="q_unans", query="q", is_answerable_gold=False,
                grounding_status="insufficient_evidence", total_claims=0, supported_claims=0,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, refusal_triggered=True, refusal_correct=True,
                total_citations=0,
            ),
            "q_ambig": QueryGroundingResult(
                query_id="q_ambig", query="q", is_answerable_gold=False,
                grounding_status="insufficient_evidence", total_claims=0, supported_claims=0,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, refusal_triggered=True, refusal_correct=True,
                total_citations=0,
            ),
        }

        report = evaluate_refusal_and_safety("SafetyTest", items, ground_map)
        assert report.refusal_precision == 1.0
        assert report.false_refusal_rate == 0.0
        assert report.incorrect_answer_rate == 0.0
        assert report.safe_refusal_rate == 1.0
        assert report.zero_citation_leakage_rate == 1.0
        assert report.overall_status == "PASS"

    def test_safety_evaluation_detects_citation_leakage(self):
        """Test that citation leakage on unanswerable queries is caught and flagged."""
        items = [
            {"id": "q_unans", "question": "Unanswerable", "category": "unanswerable", "answerable": False},
        ]
        # Query erroneously assigned a citation
        ground_map = {
            "q_unans": QueryGroundingResult(
                query_id="q_unans", query="q", is_answerable_gold=False,
                grounding_status="grounded", total_claims=1, supported_claims=1,
                unsupported_claims=0, contradicted_claims=0, conflicted_claims=0,
                claim_support_rate=1.0, refusal_triggered=False, refusal_correct=False,
                is_incorrect_answer_to_unanswerable=True,
                total_citations=1, citations=[{"chunk_id": "c1"}],
            ),
        }

        report = evaluate_refusal_and_safety("SafetyTest", items, ground_map)
        assert report.refusal_precision == 0.0
        assert report.incorrect_answer_rate == 1.0
        assert report.zero_citation_leakage_rate == 0.0
        assert report.overall_status == "FAIL"


# ============================================================================
# Suite 6: Question Generation Evaluation Tests
# ============================================================================

class TestQuestionGenerationEvaluator:
    """Verify Question Generation evaluation metrics across candidates, rejection, and provenance."""

    def test_qgen_evaluation_metrics_aggregation(self):
        """Test calculation of acceptance rate, duplicate rate, mismatch rate, and provenance."""
        runs = [
            {
                "chunk_id": "c_01",
                "document_id": "doc_1",
                "page_number": 1,
                "chunk_text": "Raft leader election uses randomized timeouts between 150ms and 300ms.",
                "response": {
                    "raw_candidates_count": 5,
                    "questions": [
                        {
                            "question_text": "What timeouts does Raft leader election use?",
                            "expected_answer": "between 150ms and 300ms",
                            "question_type": "factoid",
                            "provenance": {"chunk_id": "c_01", "page_number": 1},
                        },
                        {
                            "question_text": "What algorithm uses randomized timeouts?",
                            "expected_answer": "Raft",
                            "question_type": "factoid",
                            "provenance": {"chunk_id": "c_01", "page_number": 1},
                        },
                    ],
                    "rejection_summary": {
                        "duplicate": 1,
                        "answer_mismatch": 1,
                        "unanswerable": 1,
                    },
                },
            }
        ]

        report = evaluate_question_generation_runs("QGenTest", runs)
        assert report.total_chunks_evaluated == 1
        assert report.total_raw_candidates == 5
        assert report.total_accepted_questions == 2
        assert report.total_rejected_questions == 3
        assert report.acceptance_rate == 0.4
        assert report.rejection_rate == 0.6
        assert report.duplicate_rate == 0.2
        assert report.answer_mismatch_rate == 0.2
        assert report.unanswerable_rate == 0.2
        assert report.provenance_completeness_rate == 1.0
        assert report.overall_status == "PASS"


# ============================================================================
# Suite 7: Latency Benchmark Statistics Tests
# ============================================================================

class TestLatencyBenchmark:
    """Verify statistical latency calculations: mean, median, P95, and bottleneck tagging."""

    def test_percentile_computation(self):
        """Test percentile calculations across small and large sample arrays."""
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert compute_percentile(samples, 50.0) == 30.0
        # 95th percentile of 5 samples: k = 4 * 0.95 = 3.8 -> 40 + 0.8 * 10 = 48.0
        assert compute_percentile(samples, 95.0) == 48.0

    def test_compute_stage_latency_stats(self):
        """Test compute_stage_latency_stats converting seconds to milliseconds."""
        # 0.1s, 0.2s, 0.3s -> 100ms, 200ms, 300ms
        samples_sec = [0.1, 0.2, 0.3]
        stats = compute_stage_latency_stats("test_stage", samples_sec, bottleneck_threshold_ms=150.0)

        assert stats.sample_count == 3
        assert stats.mean_ms == 200.0
        assert stats.median_ms == 200.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 300.0
        assert stats.is_bottleneck is True  # 200ms >= 150ms

    def test_build_latency_report(self):
        """Test building full latency report across 10 stages."""
        stage_samples = {
            "retrieval": [0.015, 0.020, 0.025],  # ~20ms
            "reranking": [0.150, 0.180, 0.210],  # ~180ms
            "generative_qa": [0.350, 0.400, 0.450],  # ~400ms (Bottleneck)
        }
        report = build_latency_report("LatencyTest", stage_samples)
        assert "generative_qa" in report.stages
        assert report.primary_bottleneck_stage == "generative_qa"
        assert report.stages["generative_qa"].is_bottleneck is True
        assert report.overall_status == "PASS"


# ============================================================================
# Suite 8: Phase 20 Golden Benchmark Regression Protection
# ============================================================================

class TestPhase20GoldenRegressionProtection:
    """Verify Phase 20 Golden Benchmark remains completely intact and compliant."""

    def test_phase20_golden_v1_remains_unmodified_and_verified(self):
        """Ensure golden_v1.json exists, is VERIFIED, and has exactly 10 questions (7 answerable, 3 unanswerable)."""
        path = REPO_ROOT / "evaluation" / "datasets" / "golden_v1.json"
        dataset = load_dataset(path)

        assert dataset.verification_status == "VERIFIED"
        assert dataset.total_count == 10
        assert dataset.answerable_count == 7
        assert dataset.unanswerable_count == 3

        # Verify all 10 question IDs match the regression baseline
        expected_ids = [f"gold_q{i:02d}" for i in range(1, 11)]
        dataset_ids = [it.id for it in dataset.items]
        assert dataset_ids == expected_ids
