"""Automated Evaluation Runner for BookRAG AI.

Executes the full evaluation pipeline across:
1. Golden / Sample Dataset loading (distinguishing VERIFIED, NOT_VERIFIED, BLOCKED)
2. Deterministic FAISS Vector Retrieval (Stage 1)
3. Cross-Encoder Precision Reranking (Stage 2)
4. Comparative IR Ranking Metrics (Recall@1, 3, 5, 10; Precision@1, 3, 5; MRR)
5. Abstractive Answer Generation & NLI Groundedness Validation
6. Answer Correctness Evaluation (Token F1 overlap against ground truth)
7. Answerability, Refusal Testing, and Safe Refusal Verification
8. Citation Correctness (Precision and Recall over verified chunks and pages)
9. Generation of structured JSON reports and human-readable Markdown summary.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root and backend directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.schemas.chunk import Chunk
from app.services.embeddings.model import DEFAULT_MODEL_NAME, EmbeddingModel
from app.services.embeddings.service import EmbeddingService
from app.services.generation.model import GenerationModel
from app.services.generation.service import GenerationService
from app.schemas.grounding import GroundedAnswerRequest
from app.services.grounding.claims import ClaimDecomposer
from app.services.grounding.model import NLIModel
from app.services.grounding.orchestrator import GroundedAnswerService
from app.services.grounding.service import GroundingService
from app.services.reranking.model import RerankerModel
from app.services.reranking.service import RerankerService
from app.services.retrieval.index import VectorIndex
from app.services.retrieval.service import RetrievalService
from app.services.search.service import SearchService

from evaluation.datasets.sample_document import (
    DOCUMENT_AUTHOR,
    DOCUMENT_ID,
    DOCUMENT_TITLE,
    get_sample_chunks,
)
from evaluation.datasets.schema import EvaluationDataset
from evaluation.grounding.evaluator import (
    SEMANTIC_DISCLAIMER,
    AggregatedGroundingMetrics,
    QueryGroundingResult,
    aggregate_grounding_metrics,
    evaluate_grounded_response,
)
from evaluation.retrieval.reranking_eval import (
    RerankingComparisonReport,
    compare_retrieval_stages,
)


def load_dataset(dataset_path: Path) -> EvaluationDataset:
    """Load and validate an evaluation dataset from JSON."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvaluationDataset.model_validate(data)


def build_evaluation_index(
    chunks: List[Chunk],
    embedding_service: EmbeddingService,
) -> RetrievalService:
    """Construct an in-memory FAISS VectorIndex populated with sample document chunks."""
    retrieval_service = RetrievalService(embedding_service=embedding_service)
    retrieval_service.build_index_from_chunks(chunks, index_id=DOCUMENT_ID, set_as_default=True)
    return retrieval_service


def run_full_evaluation(
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end evaluation benchmark.

    Args:
        dataset_path: Path to dataset JSON. Defaults to evaluation/datasets/golden_v1.json.
        output_dir: Path to write reports. Defaults to evaluation/reports/.

    Returns:
        Dict containing raw report objects and benchmark summaries.
    """
    if dataset_path is None:
        golden_path = REPO_ROOT / "evaluation" / "datasets" / "golden_v1.json"
        dev_path = REPO_ROOT / "evaluation" / "datasets" / "dev_sample.json"
        dataset_path = golden_path if golden_path.exists() else dev_path

    if output_dir is None:
        output_dir = REPO_ROOT / "evaluation" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("BookRAG AI — Phase 20 Production RAG Quality & Evaluation")
    print("=======================================================")
    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Dataset: '{dataset.name}' (Version: v{dataset.version})")
    print(f"Verification Status: [{dataset.verification_status}]")
    print(f"Total Questions: {dataset.total_count} (Answerable: {dataset.answerable_count}, Unanswerable: {dataset.unanswerable_count})")
    print(f"Question Categories: {dataset.category_counts}")
    print(f"Target Document: {dataset.document_title} ({dataset.document_id})")

    # 1. Initialize ML singletons and services
    print("\n[1/4] Initializing RAG components and embedding evaluation document...")
    start_setup = time.perf_counter()
    embed_model = EmbeddingModel.get_instance(model_name=settings.EMBEDDING_MODEL_NAME, device="auto")
    embedding_service = EmbeddingService(model=embed_model)

    chunks = get_sample_chunks()
    retrieval_service = build_evaluation_index(chunks, embedding_service)
    setup_duration = time.perf_counter() - start_setup
    print(f"-> Indexed {len(chunks)} chunks in {setup_duration:.2f}s")

    reranker_model = RerankerModel.get_instance(model_name=settings.RERANKER_MODEL_NAME, device="auto")
    reranker_service = RerankerService(model=reranker_model)
    search_service = SearchService(retrieval_service=retrieval_service, reranker_service=reranker_service)

    gen_model = GenerationModel.get_instance(model_name=settings.GENERATION_MODEL_NAME, device="auto")
    gen_service = GenerationService(model=gen_model)

    claim_decomposer = ClaimDecomposer(min_claim_length=settings.GROUNDING_MIN_CLAIM_LENGTH)
    nli_model = NLIModel.get_instance(model_name=settings.GROUNDING_MODEL_NAME, device="auto")
    grounding_service = GroundingService(model=nli_model, decomposer=claim_decomposer)

    grounded_answer_service = GroundedAnswerService(
        search_service=search_service,
        generation_service=gen_service,
        grounding_service=grounding_service,
    )

    # 2. Retrieval & Reranking Evaluation
    print("\n[2/4] Evaluating Stage 1 Vector Retrieval vs Stage 2 Cross-Encoder Reranking...")
    stage1_chunks_by_query: Dict[str, List[str]] = {}
    stage2_chunks_by_query: Dict[str, List[str]] = {}
    stage1_pages_by_query: Dict[str, List[int]] = {}
    stage2_pages_by_query: Dict[str, List[int]] = {}
    retrieval_latencies: List[float] = []
    rerank_latencies: List[float] = []

    for item in dataset.items:
        # Run Stage 1 (without reranking)
        t0 = time.perf_counter()
        resp_s1 = search_service.search(
            query=item.question,
            document_id=DOCUMENT_ID,
            top_k=10,
            enable_reranking=False,
        )
        retrieval_latencies.append(time.perf_counter() - t0)
        stage1_chunks_by_query[item.id] = [r.chunk_id for r in resp_s1.results]
        stage1_pages_by_query[item.id] = [r.page_number for r in resp_s1.results]

        # Run Stage 2 (with Cross-Encoder reranking over candidate_k=10)
        t1 = time.perf_counter()
        resp_s2 = search_service.search(
            query=item.question,
            document_id=DOCUMENT_ID,
            top_k=10,
            candidate_k=10,
            enable_reranking=True,
        )
        rerank_latencies.append(time.perf_counter() - t1)
        stage2_chunks_by_query[item.id] = [r.chunk_id for r in resp_s2.results]
        stage2_pages_by_query[item.id] = [r.page_number for r in resp_s2.results]

    comparison_report = compare_retrieval_stages(
        dataset_name=dataset.name,
        query_items=[item.model_dump() for item in dataset.items],
        stage1_results_by_query=stage1_chunks_by_query,
        stage2_results_by_query=stage2_chunks_by_query,
        stage1_pages_by_query=stage1_pages_by_query,
        stage2_pages_by_query=stage2_pages_by_query,
        k_values=(1, 3, 5, 10),
    )

    avg_retrieval_ms = (sum(retrieval_latencies) / len(retrieval_latencies)) * 1000.0 if retrieval_latencies else 0.0
    avg_rerank_ms = (sum(rerank_latencies) / len(rerank_latencies)) * 1000.0 if rerank_latencies else 0.0

    print(f"-> Evaluated {comparison_report.evaluated_queries} answerable queries (and {comparison_report.unanswerable_queries} unanswerable/out-of-scope queries)")
    print(f"   Stage 1 (FAISS Vector): MRR = {comparison_report.stage1_metrics.mrr:.4f} | Recall@1 = {comparison_report.stage1_metrics.mean_recall_at_k.get(1, 0.0):.4f} | Precision@1 = {comparison_report.stage1_metrics.mean_precision_at_k.get(1, 0.0):.4f}")
    print(f"   Stage 2 (CrossEncoder): MRR = {comparison_report.stage2_metrics.mrr:.4f} | Recall@1 = {comparison_report.stage2_metrics.mean_recall_at_k.get(1, 0.0):.4f} | Precision@1 = {comparison_report.stage2_metrics.mean_precision_at_k.get(1, 0.0):.4f}")
    print(f"   Measured Deltas: Delta MRR = {comparison_report.delta_mrr:+.4f} | Delta Recall@1 = {comparison_report.delta_recall_at_k.get(1, 0.0):+.4f} | Delta Precision@1 = {comparison_report.delta_precision_at_k.get(1, 0.0):+.4f}")
    print(f"   Movement: {comparison_report.improved_queries_count} improved, {comparison_report.unchanged_queries_count} unchanged, {comparison_report.degraded_queries_count} degraded")
    print(f"   Average Latencies: Vector Search = {avg_retrieval_ms:.1f}ms | Cross-Encoder Rerank = {avg_rerank_ms:.1f}ms")

    # 3. Grounding, Answer Quality, Refusal, and Citation Evaluation
    print("\n[3/4] Evaluating Answer Generation, NLI Grounding, Answer Quality, and Refusal...")
    grounding_results: List[QueryGroundingResult] = []
    generation_latencies: List[float] = []

    for item in dataset.items:
        t_gen = time.perf_counter()
        grounded_resp = grounded_answer_service.answer_with_grounding(
            GroundedAnswerRequest(
                query=item.question,
                document_id=DOCUMENT_ID,
                top_k=3,
                candidate_k=10,
                enable_reranking=True,
            )
        )
        generation_latencies.append(time.perf_counter() - t_gen)

        claims_breakdown = [
            {
                "claim_text": c.claim_text,
                "classification": c.status,
            }
            for c in grounded_resp.claims
        ]
        citation_payloads = [
            {
                "citation_id": cite.citation_id,
                "document_id": cite.document_id,
                "chunk_id": cite.chunk_id,
                "page_number": cite.page_number,
            }
            for cite in grounded_resp.citations
        ]

        q_res = evaluate_grounded_response(
            query_id=item.id,
            query=item.question,
            is_answerable_gold=item.answerable,
            grounding_status=grounded_resp.grounding_status,
            generated_answer=grounded_resp.answer,
            expected_answer=item.expected_answer,
            category=item.category,
            claims_results=claims_breakdown,
            citations_list=citation_payloads,
            relevant_chunk_ids=item.relevant_chunk_ids,
            relevant_page_numbers=item.relevant_page_numbers,
        )
        grounding_results.append(q_res)

    grounding_report = aggregate_grounding_metrics(
        dataset_name=dataset.name,
        results=grounding_results,
        verification_status=dataset.verification_status,
    )
    avg_gen_ms = (sum(generation_latencies) / len(generation_latencies)) * 1000.0 if generation_latencies else 0.0

    print(f"-> Generated and evaluated answers for {len(dataset.items)} questions:")
    print(f"   Total Claims: {grounding_report.total_claims} (Supported: {grounding_report.supported_claims}, Unsupported: {grounding_report.unsupported_claims}, Contradicted: {grounding_report.contradicted_claims}, Conflicted: {grounding_report.conflicted_claims})")
    print(f"   Answerable Claim Support Rate: {grounding_report.answerable_claim_support_rate:.2%}")
    print(f"   Answerable Success Rate: {grounding_report.answerable_success_rate:.2%}")
    if grounding_report.mean_answer_correctness_f1 is not None:
        print(f"   Mean Answer Correctness F1: {grounding_report.mean_answer_correctness_f1:.4f}")
    print(f"   Refusal Precision: {grounding_report.refusal_precision:.2%} ({grounding_report.unanswerable_refusal_count}/{grounding_report.unanswerable_questions} unanswerable correctly refused)")
    print(f"   Citation Precision: {grounding_report.mean_citation_precision:.2%} | Citation Recall: {grounding_report.mean_citation_recall:.2%}")
    print(f"   Average Generation + NLI Grounding Latency: {avg_gen_ms:.1f}ms")

    # 4. Write Reports to Disk
    print("\n[4/4] Writing structured JSON reports and Markdown summary...")

    # Write retrieval_report.json
    retrieval_report_data = {
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "dataset_type": dataset.dataset_type,
        "verification_status": dataset.verification_status,
        "target_document": dataset.document_title,
        "document_id": dataset.document_id,
        "total_queries": len(dataset.items),
        "evaluated_queries": comparison_report.evaluated_queries,
        "unanswerable_queries": comparison_report.unanswerable_queries,
        "average_latency_ms": {
            "vector_search_ms": round(avg_retrieval_ms, 2),
            "cross_encoder_rerank_ms": round(avg_rerank_ms, 2),
        },
        "stage1_vector_metrics": comparison_report.stage1_metrics.model_dump(),
        "stage2_reranked_metrics": comparison_report.stage2_metrics.model_dump(),
        "measured_deltas": {
            "delta_mrr": comparison_report.delta_mrr,
            "delta_recall_at_k": comparison_report.delta_recall_at_k,
            "delta_precision_at_k": comparison_report.delta_precision_at_k,
        },
        "ranking_movement_summary": {
            "improved_queries": comparison_report.improved_queries_count,
            "degraded_queries": comparison_report.degraded_queries_count,
            "unchanged_queries": comparison_report.unchanged_queries_count,
        },
        "scientific_statement": comparison_report.scientific_statement,
    }
    retrieval_report_path = output_dir / "retrieval_report.json"
    with open(retrieval_report_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_report_data, f, indent=2)

    # Write grounding_report.json
    grounding_report_data = {
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "dataset_type": dataset.dataset_type,
        "verification_status": dataset.verification_status,
        "target_document": dataset.document_title,
        "document_id": dataset.document_id,
        "total_questions": grounding_report.total_questions,
        "answerable_questions": grounding_report.answerable_questions,
        "unanswerable_questions": grounding_report.unanswerable_questions,
        "average_generation_and_validation_ms": round(avg_gen_ms, 2),
        "claims_summary": {
            "total_claims": grounding_report.total_claims,
            "supported_claims": grounding_report.supported_claims,
            "unsupported_claims": grounding_report.unsupported_claims,
            "contradicted_claims": grounding_report.contradicted_claims,
            "conflicted_claims": grounding_report.conflicted_claims,
            "overall_claim_support_rate": grounding_report.overall_claim_support_rate,
            "answerable_claim_support_rate": grounding_report.answerable_claim_support_rate,
        },
        "answer_quality_summary": {
            "mean_answer_correctness_f1": grounding_report.mean_answer_correctness_f1,
            "answerable_success_rate": grounding_report.answerable_success_rate,
        },
        "refusal_summary": {
            "unanswerable_questions": grounding_report.unanswerable_questions,
            "successful_refusals": grounding_report.unanswerable_refusal_count,
            "incorrect_unanswerable_answers": grounding_report.incorrect_unanswerable_answers,
            "refusal_precision": grounding_report.refusal_precision,
            "false_refusal_count": grounding_report.false_refusal_count,
            "over_refusal_rate": grounding_report.over_refusal_rate,
        },
        "citation_summary": {
            "mean_citation_precision": grounding_report.mean_citation_precision,
            "mean_citation_recall": grounding_report.mean_citation_recall,
        },
        "category_breakdown": grounding_report.category_metrics,
        "semantic_disclaimer": SEMANTIC_DISCLAIMER,
        "per_query_evaluations": [r.model_dump() for r in grounding_report.per_query_results],
    }
    grounding_report_path = output_dir / "grounding_report.json"
    with open(grounding_report_path, "w", encoding="utf-8") as f:
        json.dump(grounding_report_data, f, indent=2)

    # Write unified evaluation_report.json
    unified_report_data = {
        "benchmark": {
            "name": dataset.name,
            "version": dataset.version,
            "dataset_type": dataset.dataset_type,
            "verification_status": dataset.verification_status,
            "document_id": dataset.document_id,
            "document_title": dataset.document_title,
            "total_questions": dataset.total_count,
            "answerable_count": dataset.answerable_count,
            "unanswerable_count": dataset.unanswerable_count,
            "category_distribution": dataset.category_counts,
        },
        "retrieval": retrieval_report_data,
        "grounding_and_quality": grounding_report_data,
    }
    unified_report_path = output_dir / "evaluation_report.json"
    with open(unified_report_path, "w", encoding="utf-8") as f:
        json.dump(unified_report_data, f, indent=2)

    # Write human-readable evaluation_summary.md
    summary_md = generate_markdown_summary(
        dataset=dataset,
        comparison_report=comparison_report,
        grounding_report=grounding_report,
        avg_retrieval_ms=avg_retrieval_ms,
        avg_rerank_ms=avg_rerank_ms,
        avg_gen_ms=avg_gen_ms,
    )
    summary_path = output_dir / "evaluation_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"-> Saved: {retrieval_report_path.relative_to(REPO_ROOT)}")
    print(f"-> Saved: {grounding_report_path.relative_to(REPO_ROOT)}")
    print(f"-> Saved: {unified_report_path.relative_to(REPO_ROOT)}")
    print(f"-> Saved: {summary_path.relative_to(REPO_ROOT)}")
    print("\n=======================================================")
    print("Evaluation Complete!")
    print("=======================================================\n")

    return {
        "retrieval_report": retrieval_report_data,
        "grounding_report": grounding_report_data,
        "unified_report": unified_report_data,
        "summary_markdown_path": str(summary_path),
    }


def generate_markdown_summary(
    dataset: EvaluationDataset,
    comparison_report: RerankingComparisonReport,
    grounding_report: AggregatedGroundingMetrics,
    avg_retrieval_ms: float,
    avg_rerank_ms: float,
    avg_gen_ms: float,
) -> str:
    """Generate comprehensive human-readable Markdown evaluation report."""
    s1 = comparison_report.stage1_metrics
    s2 = comparison_report.stage2_metrics

    # Format verification badge
    v_status = dataset.verification_status
    if v_status == "VERIFIED":
        v_badge = "**VERIFIED** (Human-signed ground truth)"
    elif v_status == "BLOCKED":
        v_badge = "**BLOCKED** (Data or label integrity blocker detected)"
    else:
        v_badge = "**NOT VERIFIED** (Pending formal human verification sign-off)"

    f1_display = f"{grounding_report.mean_answer_correctness_f1:.4f}" if grounding_report.mean_answer_correctness_f1 is not None else "N/A"

    # Category breakdown table rows
    cat_rows = []
    for cat, count in sorted(dataset.category_counts.items()):
        metrics = grounding_report.category_metrics.get(cat, {})
        supp_rate = f"{metrics.get('claim_support_rate', 0.0):.1%}" if 'claim_support_rate' in metrics else "N/A"
        succ_rate = f"{metrics.get('success_rate', 0.0):.1%}" if 'success_rate' in metrics else "N/A"
        cat_rows.append(f"| `{cat}` | {count} | {supp_rate} | {succ_rate} |")
    category_table = "\n".join(cat_rows)

    md = f"""# BookRAG AI — Phase 20 Production RAG Quality & Evaluation Report

**Generated**: Complete Automated Evaluation Pipeline  
**Target Document**: {dataset.document_title} (`{dataset.document_id}`)  
**Dataset**: {dataset.name} (`v{dataset.version}`)  
**Dataset Verification Status**: {v_badge}  

---

## 1. Benchmark & Dataset Overview

| Metric | Value |
| :--- | :--- |
| **Dataset Identity** | `{dataset.name}` (v{dataset.version}) |
| **Verification Status** | `{v_status}` |
| **Total Benchmark Questions** | {dataset.total_count} |
| **Answerable Questions** | {dataset.answerable_count} |
| **Unanswerable / Refusal Questions** | {dataset.unanswerable_count} |
| **Target Document Chunks** | {len(get_sample_chunks())} chunks across 8 pages |

### Question Category Distribution & Performance

| Category | Count | Claim Support Rate | Success Rate |
| :--- | :---: | :---: | :---: |
{category_table}

---

## 2. Retrieval & Reranking Comparative Evaluation

The two-stage retrieval cascade was evaluated before and after Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) on identical candidate pools ($K=10$):

| Metric | Stage 1: Vector Search (FAISS) | Stage 2: Cross-Encoder Rerank | Measured Delta (Δ) |
| :--- | :---: | :---: | :---: |
| **Mean Reciprocal Rank (MRR)** | **{s1.mrr:.4f}** | **{s2.mrr:.4f}** | **{comparison_report.delta_mrr:+.4f}** |
| **Recall@1** | {s1.mean_recall_at_k.get(1, 0.0):.4f} | {s2.mean_recall_at_k.get(1, 0.0):.4f} | {comparison_report.delta_recall_at_k.get(1, 0.0):+.4f} |
| **Recall@3** | {s1.mean_recall_at_k.get(3, 0.0):.4f} | {s2.mean_recall_at_k.get(3, 0.0):.4f} | {comparison_report.delta_recall_at_k.get(3, 0.0):+.4f} |
| **Recall@5** | {s1.mean_recall_at_k.get(5, 0.0):.4f} | {s2.mean_recall_at_k.get(5, 0.0):.4f} | {comparison_report.delta_recall_at_k.get(5, 0.0):+.4f} |
| **Recall@10** | {s1.mean_recall_at_k.get(10, 0.0):.4f} | {s2.mean_recall_at_k.get(10, 0.0):.4f} | {comparison_report.delta_recall_at_k.get(10, 0.0):+.4f} |
| **Precision@1** | {s1.mean_precision_at_k.get(1, 0.0):.4f} | {s2.mean_precision_at_k.get(1, 0.0):.4f} | {comparison_report.delta_precision_at_k.get(1, 0.0):+.4f} |
| **Precision@3** | {s1.mean_precision_at_k.get(3, 0.0):.4f} | {s2.mean_precision_at_k.get(3, 0.0):.4f} | {comparison_report.delta_precision_at_k.get(3, 0.0):+.4f} |
| **Precision@5** | {s1.mean_precision_at_k.get(5, 0.0):.4f} | {s2.mean_precision_at_k.get(5, 0.0):.4f} | {comparison_report.delta_precision_at_k.get(5, 0.0):+.4f} |

### Ranking Movement Distribution
- **Improved Queries (First relevant item moved upward)**: {comparison_report.improved_queries_count}
- **Unchanged Queries**: {comparison_report.unchanged_queries_count}
- **Degraded Queries**: {comparison_report.degraded_queries_count}
- **Scientific Verdict**: {comparison_report.scientific_statement}

---

## 3. Answer Quality & Grounding Evaluation

Each generated answer was decomposed into sentence-level claims and verified using `cross-encoder/nli-deberta-v3-base`:

| Answer Quality / Grounding Metric | Count / Score | Interpretation |
| :--- | :---: | :--- |
| **Total Claims Analyzed** | {grounding_report.total_claims} | Total atomic claims across generated answers |
| **Supported Claims (`entailed`)** | {grounding_report.supported_claims} | Claims logically entailed by retrieved text |
| **Unsupported Claims (`neutral`)** | {grounding_report.unsupported_claims} | Claims lacking explicit source evidence |
| **Contradicted Claims** | {grounding_report.contradicted_claims} | Claims directly contradicting retrieved text |
| **Conflicted Claims** | {grounding_report.conflicted_claims} | Claims with mixed evidence |
| **Answerable Claim Support Rate** | **{grounding_report.answerable_claim_support_rate:.2%}** | NLI consistency on answerable queries |
| **Overall Claim Support Rate** | **{grounding_report.overall_claim_support_rate:.2%}** | NLI consistency across all queries |
| **Answerable Success Rate** | **{grounding_report.answerable_success_rate:.2%}** | Answers unrefused, grounded, without contradiction |
| **Mean Answer Correctness F1** | **{f1_display}** | Token-level F1 against ground-truth expected answers |
| **Citation Precision** | **{grounding_report.mean_citation_precision:.2%}** | Fraction of assigned citations matching gold evidence |
| **Citation Recall** | **{grounding_report.mean_citation_recall:.2%}** | Fraction of gold evidence covered by citations |

> **Critical Semantic Disclaimers**:
> 1. **Retrieval relevance != factual correctness**: Vector similarity identifies semantic proximity, not objective real-world truth.
> 2. **Reranker score != probability**: Cross-encoder scores reflect transformer logit rankings, not calibrated probabilities.
> 3. **NLI score != truth probability**: Entailment measures logical premise-hypothesis consistency under retrieved evidence; it does NOT prove an answer is factually correct.
> 4. **Model confidence != factual correctness**: Generative fluency and confidence do not guarantee absence of hallucination.

---

## 4. Refusal Evaluation & Hallucination Defense

Tested with {grounding_report.unanswerable_questions} unanswerable / out-of-scope / ambiguous questions:

| Refusal Metric | Result | Target / Standard |
| :--- | :---: | :--- |
| **Unanswerable Test Queries** | {grounding_report.unanswerable_questions} | Benchmark items with no book evidence |
| **Correct Safe Refusals** | {grounding_report.unanswerable_refusal_count} | Safely refused (`insufficient_evidence` / refusal phrasing) |
| **Incorrect Answers to Unanswerable** | {grounding_report.incorrect_unanswerable_answers} | Unsafe answers asserting unsupported facts |
| **Refusal Precision** | **{grounding_report.refusal_precision:.2%}** | fraction of unanswerable queries correctly refused (Target: 100%) |
| **False Refusals (Over-refusal)** | {grounding_report.false_refusal_count} | Answerable queries mistakenly refused |
| **Over-Refusal Rate** | **{grounding_report.over_refusal_rate:.2%}** | Fraction of answerable queries refused (Target: 0%) |

**Safe Refusal Verification**: When queries lack supporting evidence in the corpus, the Groundedness Orchestrator enforces safe refusal (`insufficient_evidence`), suppressing hallucinated claims and preventing citation leakage.

---

## 5. Measured Production Latencies

Observed average latencies per evaluation request:

| Pipeline Stage | Model / Technology | Average Observed Latency |
| :--- | :--- | :---: |
| **Dense Vector Retrieval** | `all-MiniLM-L6-v2` + FAISS | **{avg_retrieval_ms:.1f} ms** |
| **Precision Reranking** | `ms-marco-MiniLM-L-6-v2` | **{avg_rerank_ms:.1f} ms** |
| **Generation & NLI Grounding** | `FLAN-T5-base` + `DeBERTa-v3` | **{avg_gen_ms:.1f} ms** |

---

*Report automatically generated by `evaluation.runner` during Phase 20 validation.*
"""
    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BookRAG AI Quality & Evaluation Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset JSON (defaults to golden_v1.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save JSON reports and summary Markdown",
    )
    args = parser.parse_args()

    ds_path = Path(args.dataset) if args.dataset else None
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_full_evaluation(dataset_path=ds_path, output_dir=out_dir)
