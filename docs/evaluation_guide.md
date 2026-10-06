# BookRAG AI — Production RAG Quality & Evaluation Guide

This guide details the evaluation subsystem for BookRAG AI (Phase 20), designed for reproducible, human-verified golden benchmarks, multi-stage retrieval ranking assessment, answer quality quantification, refusal performance testing, and hallucination defense auditing.

---

## 1. How to Create a Human-Verified Benchmark

A human-verified golden benchmark ensures evaluation is conducted against audited, verifiable ground-truth evidence rather than synthetic model assumptions.

### Benchmark Creation Protocol:
1. **Document Selection**: Identify a canonical reference document in the corpus (e.g. `doc_ai_handbook_01`).
2. **Question Formulation Across Categories**:
   Formulate questions spanning core cognitive categories:
   - `direct_fact`: Exact factual retrieval from a single passage.
   - `definition`: Conceptual definitions, terminology, and formal classifications.
   - `explanation`: Causal reasoning and mathematical/architectural justifications.
   - `multi_page`: Questions requiring synthesis across multiple separated pages.
   - `comparison`: Contrasting two or more paradigms, algorithms, or approaches.
   - `numerical_fact`: Exact numerical values, dimensions, formulas, or constants.
   - `unanswerable`: Out-of-scope questions completely absent from the book.
   - `ambiguous`: Questions lacking necessary context or falsely assuming a unique answer.
3. **Evidence Extraction**:
   - For answerable queries, human annotators locate the exact supporting page numbers (`relevant_page_numbers`) and chunk IDs (`relevant_chunk_ids`).
   - Extract verbatim excerpts as the `evidence` field.
   - Compose a clear, factual `expected_answer`.
4. **Verification Status Sign-Off**:
   - Initial drafts MUST be marked `"verification_status": "NOT_VERIFIED"`.
   - Only after an independent human domain expert audits and signs off on the question, ground truth answer, and exact evidence quotes should the status be promoted to `"verification_status": "VERIFIED"`.
   - If data or label integrity is compromised or blocked, mark as `"verification_status": "BLOCKED"`.
   - **Never fabricate human verification**.

---

## 2. Dataset Schema

The schema is formally defined in [`evaluation/datasets/schema.py`](file:///c:/Book_Rag_AI/evaluation/datasets/schema.py) using Pydantic.

### Envelope: `EvaluationDataset`
| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | Name of the benchmark suite |
| `version` | `str` | Semantic version string (e.g. `"1.0.0"`) |
| `dataset_type` | `str` | Dataset tier: `"golden"`, `"sample"`, or `"human_verified"` |
| `verification_status` | `Literal["VERIFIED", "NOT_VERIFIED", "BLOCKED"]` | Verification state |
| `description` | `str` | Overview of dataset objectives and target domain |
| `document_id` | `str` | Target book / document identifier |
| `document_title` | `str` | Human-readable document title |
| `items` | `List[EvaluationItem]` | List of individual evaluation questions |

### Item: `EvaluationItem`
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` | Unique evaluation item identifier (`"gold_q01"`) |
| `question` | `str` | Natural language question or query |
| `document_id` | `str` | Target document identifier |
| `relevant_page_numbers` | `List[int]` | 1-based page numbers containing direct answer evidence |
| `relevant_chunk_ids` | `List[str]` | Specific chunk IDs containing supporting evidence |
| `expected_answer` | `Optional[str]` | Authoritative reference answer (null if unanswerable) |
| `answerable` | `bool` | Whether the question can be answered from book alone |
| `category` | `str` | Question category (`direct_fact`, `definition`, `explanation`, `multi_page`, `comparison`, `numerical_fact`, `unanswerable`, `ambiguous`) |
| `verification_status` | `Literal["VERIFIED", "NOT_VERIFIED", "BLOCKED"]` | Verification status of this item |
| `evidence` | `Optional[str]` | Exact verbatim excerpt or reference quote from source |
| `notes` | `Optional[str]` | Annotator rationale, test objectives, or edge cases |

---

## 3. How to Run Evaluation

### CLI Execution
Run the automated evaluation runner against the golden benchmark:
```bash
# Evaluate golden_v1.json (default)
python evaluation/runner.py

# Evaluate a specific dataset JSON
python evaluation/runner.py --dataset evaluation/datasets/golden_v1.json

# Save reports to a custom directory
python evaluation/runner.py --dataset evaluation/datasets/golden_v1.json --output-dir evaluation/reports/
```

### Programmatic Execution
```python
from pathlib import Path
from evaluation.runner import run_full_evaluation

results = run_full_evaluation(
    dataset_path=Path("evaluation/datasets/golden_v1.json"),
    output_dir=Path("evaluation/reports"),
)
```

Generated reports are written to:
- [`evaluation/reports/retrieval_report.json`](file:///c:/Book_Rag_AI/evaluation/reports/retrieval_report.json): Stage 1 vs Stage 2 ranking metrics and deltas.
- [`evaluation/reports/grounding_report.json`](file:///c:/Book_Rag_AI/evaluation/reports/grounding_report.json): Claim support rates, answer correctness F1, refusal metrics, citation precision.
- [`evaluation/reports/evaluation_report.json`](file:///c:/Book_Rag_AI/evaluation/reports/evaluation_report.json): Unified comprehensive evaluation report.
- [`evaluation/reports/evaluation_summary.md`](file:///c:/Book_Rag_AI/evaluation/reports/evaluation_summary.md): Human-readable Markdown summary with tables and disclaimers.

---

## 4. How Metrics Are Calculated

### A. Retrieval Metrics
1. **Recall@K**:
   $$\text{Recall@K} = \frac{|\text{Top-K Retrieved} \cap \text{Relevant Gold Items}|}{|\text{Relevant Gold Items}|}$$
   Evaluated at $K \in \{1, 3, 5, 10\}$.
2. **Precision@K**:
   $$\text{Precision@K} = \frac{|\text{Top-K Retrieved} \cap \text{Relevant Gold Items}|}{K}$$
   Evaluated at $K \in \{1, 3, 5\}$.
3. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q_{\text{eval}}|} \sum_{i=1}^{|Q_{\text{eval}}|} \frac{1}{\text{rank}_i}$$
   where $\text{rank}_i$ is the 1-based rank position of the first relevant item. Queries with no relevant items (unanswerable) are excluded from the ranking denominator.

### B. Reranking Comparison
Contrasts Stage 1 (FAISS Vector Retrieval) with Stage 2 (Cross-Encoder Reranking):
- **MRR Delta**: $\Delta\text{MRR} = \text{MRR}_{\text{Stage2}} - \text{MRR}_{\text{Stage1}}$
- **Recall@K Delta**: $\Delta\text{Recall@K} = \text{Recall@K}_{\text{Stage2}} - \text{Recall@K}_{\text{Stage1}}$
- **Precision@K Delta**: $\Delta\text{Precision@K} = \text{Precision@K}_{\text{Stage2}} - \text{Precision@K}_{\text{Stage1}}$
- **Query Movement**: Counts queries where first relevant item rank improved, remained unchanged, or degraded.
- **Scientific Neutrality**: Reranking improvements are never claimed unless demonstrated by positive empirical deltas.

### C. Answer Quality & Correctness
1. **Token Overlap F1**:
   Standard SQuAD/QA metric: Candidate answers and expected answers are normalized (lowercased, stripped of punctuation and articles).
   $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
2. **Answerable Success Rate**:
   Fraction of answerable queries that produce an unrefused answer with valid grounding (`grounded` or `partially_grounded`), at least one supported claim, and zero contradicted claims.

### D. Grounding & Claim Support
1. **Decomposed Claim Support Rate**:
   $$\text{Support Rate} = \frac{\text{Supported Claims (Entailed)}}{\text{Total Claims Decomposed}}$$
   Classified using `cross-encoder/nli-deberta-v3-base` across 3 NLI categories: entailment, neutral (unsupported), contradiction.
2. **Conflicted Claims**: Claims having both supporting and contradicting evidence.

### E. Refusal Performance
1. **Refusal Precision**:
   $$\text{Refusal Precision} = \frac{\text{Correct Safe Refusals}}{\text{Total Unanswerable Queries}}$$
2. **Incorrect Answers to Unanswerable**: Unsafe answers where the model asserted fabricated facts on out-of-scope or ambiguous questions.
3. **Over-Refusal Rate**: Fraction of answerable queries mistakenly refused by the safety policy.

### F. Citation Correctness
1. **Citation Precision**:
   $$\text{Citation Precision} = \frac{|\text{Assigned Citations} \cap \text{Gold Evidence}|}{|\text{Assigned Citations}|}$$
   For unanswerable queries, assigning 0 citations yields 100% precision (clean isolation).
2. **Citation Recall**:
   $$\text{Citation Recall} = \frac{|\text{Assigned Citations} \cap \text{Gold Evidence}|}{|\text{Gold Evidence}|}$$

---

## 5. How to Interpret Results

When reviewing evaluation reports, examine metrics in context:
- **High Recall@10, Low Recall@1 in Stage 1**: Indicates dense vector search successfully finds candidates in the top-10, but the best candidate is not at rank 1. This is where Cross-Encoder reranking should promote the best item.
- **$\Delta\text{MRR} = 0$**: If first-stage vector search already placed the relevant passage at rank 1, reranking cannot improve MRR further. Do not claim reranking added quality in this case.
- **High Claim Support Rate, Low Answer Correctness F1**: The answer may be strictly grounded in what was retrieved, but only partially answered the full scope of the expected ground truth answer.
- **Refusal Precision = 100%**: The system safely abstains when questions fall outside the book or lack necessary context, preventing ungrounded hallucinations.
- **Over-Refusal Rate > 0%**: The model or NLI threshold may be overly conservative, rejecting partially answerable questions.

---

## 6. Difference Between Retrieval Quality and Answer Quality

| Dimension | Retrieval Quality (IR) | Answer Quality (RAG Generation) |
| :--- | :--- | :--- |
| **Object Evaluated** | Retrieved chunk passages / pages | Synthesized natural language response |
| **Core Question** | *"Did we locate the right pages and chunks?"* | *"Did the model produce a factually correct, complete answer?"* |
| **Key Metrics** | Recall@K, Precision@K, MRR | Token F1 overlap, factual accuracy, completeness |
| **Failure Modes** | Missing relevant chunks, ranking irrelevant text first | Hallucination, reasoning flaws, truncation |
| **Architectural Dependency** | Embeddings, vector index, cross-encoder | LLM generator prompt, parameter weights, temperature |

> **Key Takeaway**: A system can have 100% Recall@1 (perfect retrieval) yet generate an incorrect answer if the generator misinterprets the text. Conversely, an answer cannot be grounded if the retrieval stage failed to find the evidence.

---

## 7. Difference Between Relevance and Groundedness

| Dimension | Retrieval Relevance | NLI Groundedness |
| :--- | :--- | :--- |
| **Pipeline Stage** | First-stage retrieval and reranking | Post-generation NLI claim validation |
| **Technology** | Dense embeddings (cosine) / Cross-encoder | `cross-encoder/nli-deberta-v3-base` |
| **Meaning** | Passages are topical and semantically related to query | Generated claims are logically entailed by retrieved passage |
| **What It Does NOT Mean** | Does NOT mean the passage contains the exact answer | Does NOT mean the claim is universally true in the real world |
| **Risk Mitigated** | Off-topic retrieval and wasted context window | Generator hallucinations and confabulations |

### Crucial Semantic Distinctions:
1. **Retrieval relevance $\neq$ factual correctness**: High vector similarity indicates vector closeness, not objective real-world truth.
2. **Reranker score $\neq$ probability**: Reranker logit scores represent comparative transformer rankings, not calibrated likelihoods.
3. **NLI score $\neq$ truth probability**: Natural Language Inference measures premise-hypothesis entailment under the given text; it is strictly not an epistemic proof of truth.
4. **Model confidence $\neq$ factual correctness**: Fluid sentence generation does not guarantee absence of error.

---

## 8. Phase 23: Full-Book Benchmarking & Advanced Evaluation

Phase 23 expands BookRAG AI evaluation from single-chapter validation to comprehensive, full-book evaluation over a realistic 105-page technical book:
*Foundations of Distributed Systems, Cloud Architecture, and Machine Learning Infrastructure* (`doc_dist_sys_handbook_01`).

### Full-Book Benchmark Methodology
1. **Corpus Realism**:
   - Spans 10 chapters, 105 pages, 105 structured chunks.
   - Covers distributed systems fundamentals, consensus (Paxos/Raft), consistency models, storage engines (LSM-trees vs B+ trees), caching & distributed memory, transactions (2PC/Sagas), microservices & security (RBAC/ABAC/mTLS), ML serving & vector databases (HNSW/IVFFlat/PQ), cloud resiliency, and observability.
   - Realistic mathematical formulas, timeout parameters, architectural trade-offs, and multi-page concept evolutions.
2. **Dataset Verification Process**:
   - `evaluation/datasets/full_book_benchmark_v1.json` contains 36 verified benchmark questions (28 answerable across 7 technical categories, 8 unanswerable/ambiguous).
   - Every benchmark item preserves:
     - `question_id` (`id`): Unique deterministic identifier
     - `question`: Natural language test prompt
     - `document_id`: Target document reference (`doc_dist_sys_handbook_01`)
     - `page_numbers` (`relevant_page_numbers`): 1-based source pages containing ground truth evidence
     - `chunk_ids` (`relevant_chunk_ids`): Canonical chunk identifiers
     - `evidence` (`source_evidence`): Verbatim excerpt quote from the book
     - `query_type` (`category`): One of 9 analytical categories
     - `answerability` (`answerable`): Boolean flag
     - `expected_answer`: Reference answer
     - `verification_status`: Strictly audited as `VERIFIED`

---

## 9. Extended Retrieval & Reranking Metrics

### Extended Metric Definitions
- **Recall@K ($K \in \{1, 3, 5, 10\}$)**: Measures the proportion of relevant chunks retrieved in top-$K$.
- **Precision@K ($K \in \{1, 3, 5, 10\}$)**: Measures the proportion of retrieved chunks in top-$K$ that are truly relevant.
- **Mean Reciprocal Rank (MRR)**: Evaluates the rank position of the first relevant chunk ($1/\text{rank}$).
- **Duplicate Retrieval Rate**:
  $$\text{Duplicate Rate} = \frac{\sum_{q} (\text{Total Retrieved} - \text{Unique Retrieved})}{\sum_{q} \text{Total Retrieved}}$$
  Tracked at chunk level and page level to identify redundancy in context windows.

### Reranking Comparison Methodology
Evaluates both:
- **Stage 1**: Dense vector retrieval (`all-MiniLM-L6-v2` via PostgreSQL + pgvector).
- **Stage 2**: Dense retrieval + Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`).

Movement Classification:
- **Improved**: First relevant passage rank improved in Stage 2 compared to Stage 1.
- **Unchanged**: First relevant passage rank remained at the same rank.
- **Degraded**: First relevant passage rank moved down in Stage 2.
- **Scientific Neutrality**: Score differences and rank movements are reported objectively without claiming artificial reranker superiority where empirical metrics show parity or regression.

---

## 10. Query-Type Isolated Breakdown

Evaluation results are decomposed across 9 distinct categories without hiding category weaknesses inside an aggregate score:
1. `direct_fact`: Exact factual retrieval from a single passage.
2. `definition`: Formal terminology, conceptual definitions, and taxonomy.
3. `explanation`: Architectural rationale, causal dynamics, and protocol mechanics.
4. `comparison`: Structural and algorithmic trade-offs between two paradigms.
5. `numerical_fact`: Exact numbers, formulas, byte counts, and timeout windows.
6. `reasoning`: Multi-hop deduction and failure analysis under partition conditions.
7. `multi_page_synthesis`: Queries requiring integration of facts across distant pages.
8. `unanswerable`: Out-of-scope technical questions absent from the book.
9. `ambiguous`: Questions lacking necessary context or falsely assuming a single universal optimum.

---

## 11. Multi-Page Synthesis Evaluation

Dedicated evaluation for queries requiring evidence dispersed across multiple pages:
- **Page Retrieval Rate**: Fraction of all required gold pages appearing anywhere in the retrieved candidate list.
- **All Pages Retrieved Rate**: Fraction of multi-page questions where 100% of required pages were retrieved.
- **Evidence Sufficiency Rate**: Whether the retrieved chunks together contain complete premises required to answer the question.
- **Unsupported Claims Introduced**: Detection of fabricated bridging claims via DeBERTa NLI.
- **Multi-Page Citation Coverage Rate**: Proportion of necessary evidence pages cited in the generated answer.

---

## 12. Safety, Refusal, and Hallucination Evaluation

Maintains the Phase 20 safety baseline on the larger 105-page book:
- **Refusal Precision**: $\frac{\text{Correct Refusals}}{\text{Total Unanswerable + Ambiguous Queries}}$
- **False Refusal Rate (Over-Refusal)**: $\frac{\text{Erroneously Refused Answerable Queries}}{\text{Total Answerable Queries}}$
- **Incorrect Answer Rate (Hallucination)**: Rate at which unanswerable queries produce affirmative incorrect statements.
- **Safe Refusal Rate**: Proportion of unanswerable queries safely refused with zero citation hallucination leakage.
- **Citation Leakage Defense**: Verifies that refused answers never attach fabricated citation references.

---

## 13. Question Generation Evaluation Subsystem

Evaluates the controlled Question Generation pipeline across multiple chapters:
- **Raw Candidates**: Total unvalidated question candidates generated by T5.
- **Accepted Questions**: Candidates passing RoBERTa SQuAD2 extractive QA and answerability checks.
- **Rejected Questions**: Candidates rejected due to validation failure, duplicate detection, or answer mismatch.
- **Acceptance / Rejection Rates**: Diagnostic yield of the candidate pipeline.
- **Duplicate Rate**: Near-duplicate candidate detection rate via token similarity and Bloom filters.
- **Answer Mismatch Rate**: Questions where RoBERTa QA predicted answer deviates from the candidate target span.
- **Quality Failure Rate**: Syntactically malformed or low-confidence questions.
- **Provenance Completeness**: Verified 100% presence of `document_id`, `chunk_id`, `page_number`, and source text spans.
- **Grounding Validity**: Verifying that accepted questions are strictly grounded in source chunk evidence.

---

## 14. 10-Stage Pipeline Latency Benchmark

Measures empirical wall-clock latency across all 10 pipeline stages:
1. `ingestion`: PDF extraction, normalization, and chunk generation.
2. `embedding`: MiniLM-L6-v2 vector encoding per chunk.
3. `retrieval`: PostgreSQL + pgvector index search.
4. `reranking`: Cross-Encoder transformer score evaluation.
5. `query_understanding`: Query classification and decomposition planning.
6. `extractive_qa`: RoBERTa SQuAD2 span extraction.
7. `generative_qa`: FLAN-T5 conditioned answer synthesis.
8. `grounding_nli`: DeBERTa-v3 NLI claim entailment validation.
9. `citation_mapping`: Page and chunk citation assignment.
10. `question_generation`: T5 candidate generation + QA validation.

For each stage with $\ge 3$ samples, the benchmark computes:
- **Mean** latency (seconds and milliseconds)
- **Median** (P50) latency
- **P95** tail latency
- **Min / Max** bounds
- **Bottleneck Identification**: Stages exceeding 1.0 second are flagged for documentation and future optimization.

---

## 15. Limitations and Scientific Disclaimers

1. **Corpus Scope**: While 105 pages across 10 chapters represents a major expansion over single-chapter benchmarks, full commercial textbooks often exceed 500 pages. Metrics must be interpreted within this scale.
2. **NLI Granularity**: DeBERTa NLI operates on individual decomposed claim sentences. Complex multi-clause sentences may occasionally be classified as neutral if premises are subtly split across chunks.
3. **Generative Model Capacity**: `google/flan-t5-base` (248M parameters) generates concise answers; on complex multi-hop comparisons, extractive QA combined with cautious NLI gating may result in conservative refusals rather than partial speculations.
4. **Hardware Variability**: Latency measurements reflect CPU inference without GPU acceleration. Production deployment with TensorRT or ONNX Runtime will yield substantially lower P95 figures.

