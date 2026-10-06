# BookRAG AI — Phase 20 Production RAG Quality & Evaluation Report

**Generated**: Complete Automated Evaluation Pipeline  
**Target Document**: Foundations of Machine Learning & Neural Networks (`doc_ai_handbook_01`)  
**Dataset**: Machine Learning & Neural Networks Golden Benchmark (`v1.0.0`)  
**Dataset Verification Status**: **VERIFIED** (Human-signed ground truth)  

---

## 1. Benchmark & Dataset Overview

| Metric | Value |
| :--- | :--- |
| **Dataset Identity** | `Machine Learning & Neural Networks Golden Benchmark` (v1.0.0) |
| **Verification Status** | `VERIFIED` |
| **Total Benchmark Questions** | 10 |
| **Answerable Questions** | 7 |
| **Unanswerable / Refusal Questions** | 3 |
| **Target Document Chunks** | 8 chunks across 8 pages |

### Question Category Distribution & Performance

| Category | Count | Claim Support Rate | Success Rate |
| :--- | :---: | :---: | :---: |
| `ambiguous` | 1 | 0.0% | 100.0% |
| `comparison` | 1 | 100.0% | 100.0% |
| `definition` | 1 | 100.0% | 100.0% |
| `direct_fact` | 2 | 100.0% | 100.0% |
| `explanation` | 1 | 100.0% | 100.0% |
| `multi_page` | 1 | 100.0% | 100.0% |
| `numerical_fact` | 1 | 100.0% | 100.0% |
| `unanswerable` | 2 | 0.0% | 100.0% |

---

## 2. Retrieval & Reranking Comparative Evaluation

The two-stage retrieval cascade was evaluated before and after Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) on identical candidate pools ($K=10$):

| Metric | Stage 1: Vector Search (FAISS) | Stage 2: Cross-Encoder Rerank | Measured Delta (Δ) |
| :--- | :---: | :---: | :---: |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | **1.0000** | **+0.0000** |
| **Recall@1** | 0.9286 | 0.9286 | +0.0000 |
| **Recall@3** | 1.0000 | 1.0000 | +0.0000 |
| **Recall@5** | 1.0000 | 1.0000 | +0.0000 |
| **Recall@10** | 1.0000 | 1.0000 | +0.0000 |
| **Precision@1** | 1.0000 | 1.0000 | +0.0000 |
| **Precision@3** | 0.3809 | 0.3809 | +0.0000 |
| **Precision@5** | 0.2286 | 0.2286 | +0.0000 |

### Ranking Movement Distribution
- **Improved Queries (First relevant item moved upward)**: 0
- **Unchanged Queries**: 7
- **Degraded Queries**: 0
- **Scientific Verdict**: Reranking quality is empirically verified against benchmark relevance annotations; reranker improvements are never claimed unless demonstrated by positive metric deltas.

---

## 3. Answer Quality & Grounding Evaluation

Each generated answer was decomposed into sentence-level claims and verified using `cross-encoder/nli-deberta-v3-base`:

| Answer Quality / Grounding Metric | Count / Score | Interpretation |
| :--- | :---: | :--- |
| **Total Claims Analyzed** | 9 | Total atomic claims across generated answers |
| **Supported Claims (`entailed`)** | 8 | Claims logically entailed by retrieved text |
| **Unsupported Claims (`neutral`)** | 1 | Claims lacking explicit source evidence |
| **Contradicted Claims** | 0 | Claims directly contradicting retrieved text |
| **Conflicted Claims** | 0 | Claims with mixed evidence |
| **Answerable Claim Support Rate** | **100.00%** | NLI consistency on answerable queries |
| **Overall Claim Support Rate** | **88.89%** | NLI consistency across all queries |
| **Answerable Success Rate** | **100.00%** | Answers unrefused, grounded, without contradiction |
| **Mean Answer Correctness F1** | **0.4270** | Token-level F1 against ground-truth expected answers |
| **Citation Precision** | **100.00%** | Fraction of assigned citations matching gold evidence |
| **Citation Recall** | **92.86%** | Fraction of gold evidence covered by citations |

> **Critical Semantic Disclaimers**:
> 1. **Retrieval relevance != factual correctness**: Vector similarity identifies semantic proximity, not objective real-world truth.
> 2. **Reranker score != probability**: Cross-encoder scores reflect transformer logit rankings, not calibrated probabilities.
> 3. **NLI score != truth probability**: Entailment measures logical premise-hypothesis consistency under retrieved evidence; it does NOT prove an answer is factually correct.
> 4. **Model confidence != factual correctness**: Generative fluency and confidence do not guarantee absence of hallucination.

---

## 4. Refusal Evaluation & Hallucination Defense

Tested with 3 unanswerable / out-of-scope / ambiguous questions:

| Refusal Metric | Result | Target / Standard |
| :--- | :---: | :--- |
| **Unanswerable Test Queries** | 3 | Benchmark items with no book evidence |
| **Correct Safe Refusals** | 3 | Safely refused (`insufficient_evidence` / refusal phrasing) |
| **Incorrect Answers to Unanswerable** | 0 | Unsafe answers asserting unsupported facts |
| **Refusal Precision** | **100.00%** | fraction of unanswerable queries correctly refused (Target: 100%) |
| **False Refusals (Over-refusal)** | 0 | Answerable queries mistakenly refused |
| **Over-Refusal Rate** | **0.00%** | Fraction of answerable queries refused (Target: 0%) |

**Safe Refusal Verification**: When queries lack supporting evidence in the corpus, the Groundedness Orchestrator enforces safe refusal (`insufficient_evidence`), suppressing hallucinated claims and preventing citation leakage.

---

## 5. Measured Production Latencies

Observed average latencies per evaluation request:

| Pipeline Stage | Model / Technology | Average Observed Latency |
| :--- | :--- | :---: |
| **Dense Vector Retrieval** | `all-MiniLM-L6-v2` + FAISS | **58.1 ms** |
| **Precision Reranking** | `ms-marco-MiniLM-L-6-v2` | **266.2 ms** |
| **Generation & NLI Grounding** | `FLAN-T5-base` + `DeBERTa-v3` | **10521.7 ms** |

---

*Report automatically generated by `evaluation.runner` during Phase 20 validation.*
