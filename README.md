# BookRAG AI

A research-oriented, production-quality Retrieval-Augmented Generation (RAG) system engineered from first principles for querying and analyzing complete book manuscripts and multi-chapter documents.

---

## 1. What is BookRAG AI?

BookRAG AI is a modular RAG platform specifically tailored to the unique challenges of long-form literature and technical texts. Unlike simple document Q&A prototypes that slice text into naive fixed-size windows and rely on third-party black-box wrappers, BookRAG AI is designed to explicitly implement each component of the retrieval and generation pipeline.

Key capabilities planned for the platform include:
- Preserving page numbers, chapter hierarchies, and contextual book metadata.
- Intelligent document cleaning and structural chunking.
- Dense semantic vector embeddings preserving full chunk provenance.
- Exact vector retrieval using FAISS IndexFlatIP over unit-normalized dense vectors.
- Cross-encoder reranking for precision context selection.
- Dual-mode synthesis: **Extractive QA** (exact text citations) and **Abstractive QA** (synthesized reasoning).
- Groundedness validation and hallucination mitigation.
- Fine-grained question generation for readers, researchers, and students.
- Transparent relevance visualization via an interactive Relevant Matching Board.

> **Important**: This project intentionally does **not** rely on LangChain for its core RAG architecture. Each component is designed and implemented explicitly to ensure complete transparency, auditability, and control.

---

## 2. Why It Is Being Built

Most existing RAG demonstrations suffer from severe limitations when applied to full-length books:
1. **Loss of Book Context**: Long narratives and technical treatises span hundreds of pages. Naive chunking fragments ideas and disconnects citations from original pages.
2. **Framework Opacity**: High-level abstractions frequently hide chunk boundaries, vector distance metrics, prompt assemblies, and failure points.
3. **Hallucination & Ungrounded Answers**: Without strict groundedness validation and cross-encoder reranking, LLMs confidently produce incorrect citations.
4. **Shallow Evaluation**: Few systems provide verifiable benchmarks measuring precision, recall, and citation fidelity across complete books.

BookRAG AI addresses these issues through an explicit, modular, step-by-step engineering approach.

---

## 3. High-Level Architecture

The target end-to-end architecture is structured as a modular monolith:

```
[ User Browser / Frontend Client ]
               │
               ▼ (HTTP / REST)
     [ FastAPI Backend API ]
        │             │
        ▼             ▼
 [ Query Service ]   [ Document Processing Pipeline ]
        │                     │
        ▼                     ▼
 [ Reranker & QA ]     [ Ingestion → Cleaning → Chunking → Embeddings ]
        │                     │
        ▼                     ▼
 [ FAISS Vector Index (IndexFlatIP) / PostgreSQL + pgvector ]
```

### Architectural Layering:
- **API Layer (`backend/app/api/`)**: Thin controllers handling request validation, routing, and HTTP status codes.
- **Service Layer (`backend/app/services/`)**: Core domain workflows:
  - `services/pdf/`: Safe ingestion, validation, and page-aware PDF representation.
  - `services/text/`: Conservative text normalization and paragraph/sentence-aware intelligent chunking.
  - `services/embeddings/`: Dense semantic vector representations with device negotiation and provenance retention.
  - `services/retrieval/`: Exact vector indexing (FAISS IndexFlatIP), metadata mapping synchronization, document-isolated top-K search, and disk persistence.
- **Data & Repository Layer (`backend/app/repositories/` & `backend/app/models/`)**: Abstracted persistence for book metadata, chunks, and index mappings (reserved for future database phases).
- **Core Platform (`backend/app/core/`)**: Cross-cutting concerns including centralized settings, structured logging, and unified error handling.

---

## 4. Current Phase Scope: Phase 9 Complete

This repository has completed **Phase 0 (Foundation)**, **Phase 1 (PDF Ingestion)**, **Phase 2 (Text Cleaning & Chunking)**, **Phase 3 (Semantic Embeddings)**, **Phase 4 (Vector Retrieval with FAISS)**, **Phase 5 (Semantic Search Service & API)**, **Phase 6 (Cross-Encoder Reranking)**, **Phase 7 (Extractive Question Answering)**, **Phase 8 (Abstractive QA with FLAN-T5)**, and **Phase 9 (Groundedness & Hallucination Control)**.

### What is implemented:
- **Repository & Runtime Foundation (Phase 0)**:
  - Repository layout, virtual environment, and Git configuration.
  - Centralized settings with `pydantic-settings` and `.env.example`.
  - Configurable application logging and structured HTTP error handling.
  - Operational health check endpoint: `GET /api/v1/health`.
- **PDF Ingestion & Page-Aware Representation (Phase 1)**:
  - Low-level PDF parser abstraction using PyMuPDF (`pymupdf>=1.25.0`).
  - High-level `PDFIngestionService` for safe document opening, validation, and metadata extraction.
  - 1-based page numbering preserving page provenance for future citations.
  - Exact raw text extraction with per-page and document-wide character and word counts.
  - Extraction diagnostics identifying empty and low-text pages without crashing ingestion.
  - Controlled domain exceptions (`PDFNotFoundError`, `InvalidPDFError`, `PDFExtractionError`).
  - Deterministic document ID generation derived from content SHA-256 hashes.
- **Text Cleaning & Intelligent Chunking (Phase 2)**:
  - Conservative, deterministic `TextCleaner` with line-ending normalization, safe dehyphenation (`intel-\nligence` -> `intelligence`), compound word preservation (`state-of-the-art`), and whitespace normalization.
  - Paragraph-first, sentence-aware `Chunker` respecting `target_size` (1200), `max_size` (1600), and `overlap` (200).
  - Bounded semantic overlap between adjacent chunks on the same page.
  - Strict page boundaries: pages are chunked independently to prevent cross-page provenance ambiguity.
  - Deterministic, debuggable chunk IDs (`{document_id}_p{page_number:03d}_c{chunk_index:04d}`).
  - Non-destructive `TextProcessingService` keeping raw `Page.text` immutable.
- **Semantic Embeddings (Phase 3)**:
  - Real integration with `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors.
  - Single-load model lifecycle caching via `EmbeddingModel.get_instance()` avoiding redundant reloads.
  - Configurable device support with automatic negotiation: uses CUDA if available, seamlessly falls back to CPU.
  - Batch inference via `model.encode(texts, batch_size=32)` strictly preserving input order.
  - Vector normalization to unit length ($L_2 \approx 1.0$) for cosine similarity compatibility with FAISS.
  - `EmbeddingRecord` schema guaranteeing that chunk provenance (`chunk_id`, `document_id`, `page_number`) remains attached to every vector.
  - Raw `Chunk` objects remain completely immutable after embedding.
- **Vector Retrieval with FAISS (Phase 4)**:
  - `faiss.IndexFlatIP` integration computing exact inner products over $L_2$-normalized 384-dimensional dense vectors.
  - `VectorToChunkMapping` maintaining a synchronized, deterministic 1-to-1 correspondence between FAISS internal integer vector slots and source chunk provenance.
  - `VectorIndex` container managing FAISS index lifecycle, vector addition, dimension validation, and disk persistence (`.faiss` binary + `.json` metadata mapping).
  - `RetrievalService` orchestrator embedding natural language search queries with the identical Phase 3 embedding model and executing top-K nearest vector search.
  - Document isolation and filtering allowing queries to be restricted to specific book documents without cross-book pollution.
  - Structured, ranked `RetrievalResult` objects preserving complete provenance (`chunk_id`, `document_id`, `page_number`, `text`, `similarity_score`, `rank`).
  - Strict validation: rejects dimension mismatches, empty/whitespace queries, non-positive top_k, and corrupted metadata files.
- **Semantic Search Service & API (Phase 5)**:
  - Application-level `SearchService` orchestrating the retrieval pipeline.
  - Thin FastAPI endpoint: `POST /api/v1/search` with dependency injection.
  - Complete provenance preservation: `rank`, `chunk_id`, `document_id`, `page_number`, `text`, `similarity_score`, `chunk_index`, and `metadata`.
  - Document isolation and filtering with structured HTTP 404 for unknown documents.
  - Safe empty index handling returning empty results without crashing.
  - Public API contract abstraction hiding internal FAISS vector positions (`vector_index`).
- **Cross-Encoder Precision Reranking (Phase 6)**:
  - Two-stage retrieval pipeline: First-stage FAISS dense vector search retrieves a high-recall candidate pool (`candidate_k`, default 20); second-stage `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) performs joint full-attention cross-scoring over `(query, passage)` pairs to select final high-precision results (`top_k`, default 5).
  - Dedicated `RerankerService` and `RerankerModel` with single-load model lifecycle caching.
  - Full provenance retention (`original_rank` alongside final `rank`).
- **Extractive Question Answering (Phase 7)**:
  - Extractive span-selection using `deepset/roberta-base-squad2`.
  - Sliding-window passage scoring with stride tokenization and SQuAD 2.0 unanswerability handling.
  - Public endpoint: `POST /api/v1/qa`.
- **Abstractive QA / Generation with FLAN-T5 (Phase 8)**:
  - Controlled abstractive answer synthesis using `google/flan-t5-base`.
  - Token-budgeted context assembly via `EvidenceBuilder` with strict evidence-only instructions.
  - Deterministic beam search generation without random sampling.
  - Public endpoint: `POST /api/v1/answer`.
- **Groundedness & Hallucination Control (Phase 9)**:
  - Cross-Encoder NLI model wrapper using `cross-encoder/nli-deberta-v3-base` with dynamic `id2label` discovery.
  - Deterministic `ClaimDecomposer` extracting sentence-level propositions with whitespace normalization.
  - Batch NLI inference evaluating premise (evidence passage) vs hypothesis (generated claim).
  - Four-state claim classification: `entailed`, `contradicted`, `unsupported`, and `conflicted`.
  - Transparent provenance tracking of supporting and contradicting evidence chunks.
  - Safe decision policy suppressing ungrounded or contradicted answers while preserving diagnostic claim breakdown and evidence provenance.
  - Dedicated public endpoint: `POST /api/v1/grounded-answer`.
  - Complete test suite: **239 unit, integration, and API tests** passing with 100% success rate.


---

## 5. Two-Stage Retrieval Architecture: Bi-Encoder vs. Cross-Encoder

BookRAG AI implements a classical information retrieval two-stage cascade:

```
User Query: "Why does backpropagation suffer from vanishing gradients?"
      │
      ▼
[Stage 1: Bi-Encoder (SentenceTransformers all-MiniLM-L6-v2)]
      │  Query embedding generated in isolation: vector ∈ R^384
      ▼
[FAISS IndexFlatIP Dense Retrieval]
      │  High-recall candidate pool search across entire book corpus
      ▼
Candidate Pool: candidate_k = 20 candidates (ranked by similarity_score)
      │
      ▼
[Stage 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2)]
      │  Batched inference over (query, candidate_text) pairs
      │  Full cross-attention across all query and passage token pairs
      ▼
Candidate Scoring & Re-ranking:
      │  Candidates sorted by descending reranker_score
      │  original_rank preserved, new rank 1..N assigned
      ▼
Final Results: top_k = 5 high-precision evidence chunks
```

### Why Dense Retrieval is Used First
A complete book or technical manual can easily contain thousands of chunks. Passing thousands of candidate pairs into a heavy transformer model with full cross-attention for every incoming query would require thousands of forward passes and hundreds of milliseconds to seconds of latency per search request. Bi-encoder dense retrieval solves this by pre-computing chunk embeddings once and performing exact inner-product vector search in sub-millisecond time.

### Why Cross-Encoder Reranking is Needed
Bi-encoders embed queries and documents independently:
$$\text{sim}(q, d) = \cos(\mathbf{u}_q, \mathbf{v}_d)$$
Because the query and document cannot attend to each other during vector encoding, bi-encoders cannot capture complex cross-term interactions, query conditionality, or negative modifier dependencies.

Cross-encoders concatenate the query and passage into a single transformer input sequence:
$$\text{Input} = \text{[CLS]} \, q_1 \dots q_m \, \text{[SEP]} \, d_1 \dots d_n \, \text{[SEP]}$$
This allows every query token to attend to every document token across all transformer layers via bidirectional self-attention, yielding substantially higher relevance discrimination and context precision.

### Bi-Encoder vs. Cross-Encoder Comparison

| Dimension | Bi-Encoder (EmbeddingService) | Cross-Encoder (RerankerService) |
| :--- | :--- | :--- |
| **Model** | `all-MiniLM-L6-v2` | `ms-marco-MiniLM-L-6-v2` |
| **Input Structure** | Query alone $\rightarrow \mathbf{u}$; Document alone $\rightarrow \mathbf{v}$ | Joint pair: `(query, passage)` |
| **Attention Mechanism** | Independent intra-sequence self-attention | Full cross-attention across query and passage tokens |
| **Computation Speed** | Extremely fast ($O(1)$ vector distance lookup) | Slower forward pass per candidate pair |
| **Primary Role** | First-stage high-recall candidate retrieval | Second-stage high-precision candidate reranking |
| **Input Scope** | Entire corpus of book chunks | Candidate pool of size `candidate_k` |

### Candidate_k vs. Final Top_k
- `candidate_k` (default: 20): Determines the size of the high-recall pool retrieved from FAISS. A larger candidate pool gives the cross-encoder a richer set of semantically diverse candidates to evaluate.
- `top_k` (default: 5): Determines the final number of top-ranked evidence chunks returned to the caller after cross-encoder reordering.
- Constraint: `candidate_k` must always be greater than or equal to `top_k`.

### Strict Score Semantics
- `similarity_score`: Inner-product cosine similarity from first-stage dense retrieval ($[-1.0, 1.0]$ for unit-normalized vectors). Measures vector-space proximity between isolated embeddings.
- `reranker_score`: Continuous logit score output by the cross-encoder transformer for joint `(query, passage)`.
- **CRITICAL NOTE**: Neither score represents answer confidence, probability of truth, factual correctness, or hallucination metrics. They measure semantic relevance in their respective retrieval stages.

---

## 6. Extractive Question Answering (Phase 7)

Phase 7 introduces an extractive Question Answering layer on top of the two-stage retrieval pipeline. Rather than generating or hallucinating synthetic text, the system extracts the exact token span from the retrieved book evidence that answers the user's question.

### Architectural Pipeline

```
User Query
    │
    ▼
EmbeddingService (Bi-Encoder: all-MiniLM-L6-v2)
    │  Generates 384-d normalized query vector
    ▼
FAISS Dense Retrieval (Stage 1: High Recall)
    │  Retrieves candidate_k candidate chunks (default: 20)
    ▼
Cross-Encoder Reranker (Stage 2: High Precision)
    │  Joint query-passage transformer scoring (ms-marco-MiniLM-L-6-v2)
    ▼
Top Evidence Selection (Top-N Chunks)
    │  Passes top_k reranked evidence chunks to QA service
    ▼
Extractive QA Service (Stage 3: Span Extraction)
    │  Hugging Face AutoModelForQuestionAnswering (deepset/roberta-base-squad2)
    │  Sliding-window tokenization with overflow stride (512 max_length, 128 stride)
    │  SQuAD 2.0 unanswerable thresholding (score_diff > threshold)
    ▼
Best Answer Span with Complete Provenance
    ├── answer: "1998"
    ├── answer_start / answer_end character offsets in source chunk text
    ├── chunk_id, document_id, page_number, chunk_index
    └── similarity_score, reranker_score, qa_score
```

### Why Extractive QA with RoBERTa SQuAD2?
- **Factual Integrity**: Extractive QA copies text spans directly from the source book; it cannot hallucinate non-existent facts, make up dates, or invent citations.
- **The Evidence is the Truth**: The book is the authoritative ground truth. If the book does not contain the answer, the system explicitly reports the question as unanswerable.
- **SQuAD 2.0 Unanswerability**: `deepset/roberta-base-squad2` is fine-tuned on SQuAD 2.0, which includes negative (unanswerable) questions. Token 0 (`<s>`) represents the null/no-answer token with score `start_logits[0] + end_logits[0]`.

### Sliding-Window Overflow Handling
Books contain long context passages that frequently exceed a transformer's maximum token limit (512 tokens). Rather than truncating away critical evidence:
1. Long passages are split into overlapping context windows using `stride=128`.
2. Offset mappings and `sequence_ids` ensure answer spans are selected exclusively from context tokens (`sequence_id == 1`), never special tokens or question tokens.
3. Candidate spans are evaluated across all sliding windows of all evidence chunks.
4. Token start and end indices are converted back to exact character offsets (`answer_start`, `answer_end`) in the original uncompressed source text.

### Unified Score Semantics

| Metric | Origin | Meaning | What It Does NOT Mean |
| :--- | :--- | :--- | :--- |
| `similarity_score` | Bi-Encoder (Stage 1) | Cosine similarity in dense vector space | NOT probability, NOT factual truth |
| `reranker_score` | Cross-Encoder (Stage 2) | Joint transformer relevance logit | NOT confidence, NOT factual correctness |
| `qa_score` | RoBERTa SQuAD2 (Stage 3) | Extractive span logit ($s_{\text{start}} + s_{\text{end}}$) | NOT answer accuracy, NOT confidence % |
| `no_answer_score` | RoBERTa SQuAD2 (Stage 3) | Logit for the null/unanswerable token ($s_0$) | NOT probability of non-existence |

> **IMPORTANT PRINCIPLE**: None of these scores represents factual correctness, answer confidence, or hallucination metrics. The retrieved passage is the sole source of truth; the QA model simply identifies which span within the retrieved evidence best matches the query.

---

## 7. Abstractive Question Answering (Phase 8)

Phase 8 introduces controlled abstractive text generation using `google/flan-t5-base`. While Phase 7 extracts verbatim substrings from the retrieved text, Phase 8 synthesizes fluent, comprehensive natural language answers that integrate evidence across multiple passages or reformulate complex explanations.

### Architectural Pipeline

```
User Query
    │
    ▼
EmbeddingService (Bi-Encoder: all-MiniLM-L6-v2)
    │  Generates 384-d normalized query vector
    ▼
FAISS Dense Retrieval (Stage 1: High Recall)
    │  Retrieves candidate_k candidate chunks (default: 20)
    ▼
Cross-Encoder Reranker (Stage 2: High Precision)
    │  Joint query-passage transformer scoring (ms-marco-MiniLM-L-6-v2)
    ▼
Top Evidence Selection (Top-N Chunks)
    │  Passes top_k reranked evidence chunks to Generation pipeline
    ▼
EvidenceBuilder (Context Budgeting & Grounding)
    │  Enforces GENERATION_MAX_INPUT_TOKENS (default: 2048)
    │  Formats passages with explicit boundaries: [Page X] <text>
    │  Prioritizes higher-ranked chunks, carefully truncating oversized candidates
    ▼
FLAN-T5 Generation (Stage 3: Abstractive Synthesis)
    │  Hugging Face AutoModelForSeq2SeqLM (google/flan-t5-base)
    │  Controlled deterministic beam search (do_sample=False, num_beams=4)
    ▼
Synthesized Answer with Full Evidence Provenance
    ├── answer: "Backpropagation computes gradient vectors through recursive application of the chain rule..."
    ├── answerable: true
    ├── model_name: "google/flan-t5-base"
    ├── evidence: [ { rank, chunk_id, document_id, page_number, similarity_score, reranker_score, ... } ]
    └── evidence_count: 2
```

### Extractive vs. Abstractive QA

| Dimension | Extractive QA (Phase 7) | Abstractive QA (Phase 8) |
| :--- | :--- | :--- |
| **Model** | `deepset/roberta-base-squad2` | `google/flan-t5-base` |
| **Model Type** | Encoder-only (`AutoModelForQuestionAnswering`) | Encoder-Decoder Seq2Seq (`AutoModelForSeq2SeqLM`) |
| **Output Type** | Exact substring slice from source text | Synthesized natural language sentence/paragraph |
| **Answer Boundary** | Fixed character offsets (`answer_start`, `answer_end`) | Newly generated token sequence |
| **Multi-chunk Synthesis** | Selects best single span from highest-scoring chunk | Integrates and summarizes information across multiple chunks |
| **Hallucination Risk** | Zero (impossible to output words not in evidence) | Non-zero (controlled via grounded prompt and evidence restriction) |
| **Source of Truth** | Book evidence chunk | Book evidence chunk |

### Prompt Grounding and Controlled Template
To prevent the model from answering out of its pre-trained parametric memory, FLAN-T5 is conditioned with a deterministic prompt:

```text
Answer the question using only the provided context.

Context:
[Page 12] Rumelhart, Hinton, and Williams popularized backpropagation in 1986...

[Page 15] Backpropagation enables training deep networks by computing gradients...

Question:
How did backpropagation impact neural network training?

Answer:
```

### Context Budgeting & Token Management
- `GENERATION_MAX_INPUT_TOKENS=2048`: Upper token limit on the combined prompt (template instructions + question + evidence context).
- **Greedy Priority Ordering**: Evidence chunks are processed in descending rank order. As many complete chunks are accommodated as fit within the budget.
- **Graceful Truncation**: If a single top-priority chunk exceeds the available budget, it is carefully truncated with an ellipsis while retaining its full provenance metadata (`chunk_id`, `page_number`, `similarity_score`, `reranker_score`).
- **Empty Evidence Guard**: If no retrieved evidence is available, the system immediately returns a structured no-answer (`answerable=false`, `answer=null`) without invoking FLAN-T5.

### Explicit Semantic Principles
1. **The Book is the Source of Truth**: FLAN-T5 is an answer synthesis tool conditioned on retrieved text, NOT an independent factual authority.
2. **Unsupported Statements**: While prompt constraints heavily reduce hallucination, generative models can still synthesize unsupported statements.
3. **No False Confidence**: Generation output must never be interpreted as factual confidence or probability of truth.
4. **Phase 9 Boundary**: Formal NLI-based groundedness verification and automated hallucination scoring are explicitly reserved for Phase 9.

---

## 8. Technology Stack


### Backend (Current Phase 6):
- **Language**: Python 3.11+ (Tested on Python 3.13.7)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/) (>= 0.115.0)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (>= 0.32.0)
- **PDF Extraction**: [PyMuPDF](https://pymupdf.readthedocs.io/) (>= 1.25.0)
- **Bi-Encoder Embeddings**: [Sentence Transformers](https://www.sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`), PyTorch (>= 2.2.0)
- **Cross-Encoder Reranker**: [Sentence Transformers CrossEncoder](https://www.sbert.net/docs/pretrained_cross-encoders.html) (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Vector Indexing & Retrieval**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu>=1.9.0`)
- **Configuration & Validation**: [Pydantic v2](https://docs.pydantic.dev/) & [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **Testing**: [pytest](https://docs.pytest.org/) & [HTTPX](https://www.python-httpx.org/)

### Future Planned Stack:
- **Frontend**: React, TypeScript, Vite, TailwindCSS
- **Generative QA**: Extractive QA and Abstractive LLM synthesis
- **Vector Storage**: PostgreSQL + pgvector (for persistent production deployment)
- **Task Queues**: Redis & Celery
- **Infrastructure**: Docker & Docker Compose

---

## 6. Project Structure

```
BookRAG-AI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── endpoints/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── chunks.py          # POST /api/v1/chunks/clean & chunk-page
│   │   │       │   ├── documents.py       # POST /api/v1/documents/ingest
│   │   │       │   ├── embeddings.py      # POST /api/v1/embeddings/embed-chunk & embed-chunks
│   │   │       │   ├── health.py          # GET /api/v1/health implementation
│   │   │       │   ├── retrieval.py       # POST /api/v1/retrieval/index & search
│   │   │       │   └── search.py          # POST /api/v1/search (Phase 5 search endpoint)
│   │   │       ├── __init__.py
│   │   │       └── router.py              # Assembles version 1 routes
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py                  # Pydantic BaseSettings management
│   │   │   ├── errors.py                  # Safe global & domain exception handlers
│   │   │   └── logging.py                 # Standardized logging setup
│   │   ├── models/                        # Domain entities (Reserved for future DB models)
│   │   ├── repositories/                  # Persistence repositories (Future phase)
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── chunk.py                   # Chunk, ChunkingConfig models
│   │   │   ├── document.py                # Document, Page, Metadata Pydantic models
│   │   │   ├── embedding.py               # EmbeddingRecord, EmbeddingConfig models
│   │   │   ├── health.py                  # Health check Pydantic schemas
│   │   │   ├── retrieval.py               # RetrievalResult, IndexMetadata, VectorMappingItem
│   │   │   └── search.py                  # SearchRequest, SearchResponse, SearchResult (Phase 5)
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── pdf/                       # Phase 1 Ingestion Service
│   │   │   │   ├── __init__.py
│   │   │   │   ├── exceptions.py          # Domain-specific PDF ingestion exceptions
│   │   │   │   ├── ingestion.py           # Orchestration, validation, diagnostics
│   │   │   │   └── parser.py              # PyMuPDF-specific extraction mechanics
│   │   │   ├── text/                      # Phase 2 Text Processing Service
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chunker.py             # Paragraph & sentence-aware intelligent chunker
│   │   │   │   ├── cleaner.py             # Conservative deterministic text cleaner
│   │   │   │   ├── exceptions.py          # Text processing domain exceptions
│   │   │   │   └── processor.py           # Document-level multi-page text processing
│   │   │   ├── embeddings/                # Phase 3 Semantic Embeddings Service
│   │   │   │   ├── __init__.py
│   │   │   │   ├── exceptions.py          # Embedding domain exceptions
│   │   │   │   ├── model.py               # EmbeddingModel wrapper & instance registry
│   │   │   │   └── service.py             # EmbeddingService & batch inference
│   │   │   ├── retrieval/                 # Phase 4 FAISS Vector Retrieval Service
│   │   │   │   ├── __init__.py
│   │   │   │   ├── exceptions.py          # Retrieval & FAISS domain exceptions
│   │   │   │   ├── index.py               # VectorIndex wrapper & persistence
│   │   │   │   ├── mapping.py             # VectorToChunkMapping synchronization
│   │   │   │   └── service.py             # RetrievalService query search orchestrator
│   │   │   └── search/                    # Phase 5 Application Search Service
│   │   │       ├── __init__.py
│   │   │       ├── exceptions.py          # Search domain exceptions
│   │   │       └── service.py             # SearchService query orchestrator
│   │   ├── __init__.py
│   │   └── main.py                        # FastAPI application entry point
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                    # Pytest client and deterministic PDF fixtures
│   │   ├── fixtures/
│   │   │   ├── __init__.py
│   │   │   └── pdf_factory.py             # Synthetic, reproducible PDF generators
│   │   ├── test_embeddings.py             # Phase 3 embedding inference, normalization, batch tests
│   │   ├── test_health.py                 # Startup and health check tests (Phase 0)
│   │   ├── test_pdf_ingestion.py          # Phase 1 ingestion, validation, and API tests
│   │   ├── test_retrieval.py              # Phase 4 FAISS index, top-k search, mapping, persistence tests
│   │   ├── test_search.py                 # Phase 5 SearchService & API tests
│   │   └── test_text_processing.py        # Phase 2 cleaning, chunking, and overlap tests
│   ├── requirements.txt                   # Project dependencies
│   └── .env.example                       # Non-sensitive configuration template
│
├── frontend/                              # Reserved for future React application
│
├── data/                                  # Runtime data directories (version controlled via .gitkeep)
│   ├── uploads/                           # Destination for uploaded book PDFs
│   ├── processed/                         # Destination for extracted/chunked artifacts
│   └── indexes/                           # Destination for saved FAISS indexes & metadata JSON
│
├── docs/                                  # Architectural specifications and design records
├── evaluation/                            # Benchmarking datasets and evaluation scripts
│
├── .gitignore                             # Ignore rules for caches, env, data, models, *.pdf
├── README.md                              # Project documentation
└── docker-compose.yml                     # Minimal deployment skeleton for future phases
```

---

## 7. Vector Retrieval Pipeline (Phase 4)

```
User Query ("How do neural networks learn features?")
       │
       ▼
[ EmbeddingService.embed_query(query) ]
       │
       ▼
384-dimensional L2-normalized Query Vector (||q|| ≈ 1.0)
       │
       ▼
[ FAISS IndexFlatIP Search (top_k = 5) ]
   ├── Exact inner product computation: S(q, d) = q · d
   ├── Document isolation / filter check (document_id = "doc_ai")
   └── Returns top-K nearest slots & inner product distances
       │
       ▼
[ VectorToChunkMapping ]
   ├── Slot 0 -> doc_ai_p001_c0001 (Page 1)
   ├── Slot 1 -> doc_ai_p002_c0002 (Page 2)
   └── Synchronizes integer slot with complete chunk provenance
       │
       ▼
[ Structured RetrievalResult[] ]
   ├── Rank 1: Chunk doc_ai_p002_c0002 (Score: 0.812, Page: 2)
   ├── Rank 2: Chunk doc_ai_p001_c0001 (Score: 0.745, Page: 1)
   └── Complete text and provenance attached!
```

### Why FAISS IndexFlatIP?
`faiss.IndexFlatIP` computes the exact inner product between vectors with brute-force precision. Because all document chunk vectors and query vectors are $L_2$-normalized to unit length ($\|v\|_2 = 1.0$), the inner product mathematically equals the **cosine similarity**:

$$\text{Cosine Similarity}(q, d) = \frac{q \cdot d}{\|q\|_2 \|d\|_2} = q \cdot d = \text{Inner Product}(q, d)$$

This provides an exact, uncompressed baseline search without quantization distortions (such as IVF or PQ).

### Understanding Similarity Scores
- **Ranking Signal**: Similarity scores indicate relative semantic closeness between the query and candidate passages.
- **Not a Probability**: A score of `0.85` does not mean 85% probability or confidence.
- **Not Factual Correctness**: Semantic proximity does not guarantee that a text chunk contains a factually accurate answer to the question. Reranking and groundedness evaluation are applied in subsequent phases.

### Persistence Format
Persisted vector indexes are stored as two co-located files:
1. `<base_name>.faiss`: Binary serialized FAISS index structure.
2. `<base_name>.json`: Structured JSON containing index metadata (`dimension`, `model_name`, `total_vectors`, `document_ids`, `normalized`) and contiguous `VectorMappingItem` records.

---

---

## 8. Groundedness & Hallucination Control (Phase 9)

### The Purpose of Groundedness in BookRAG AI
> **"The NLI validator checks whether generated claims are supported by retrieved book evidence. It is not a general-purpose factual truth detector."**

A fundamental principle in technical RAG systems is distinguishing **groundedness** from **objective factual correctness**:
- **Groundedness**: Does the claim logically follow from (is it entailed by) the specific text passages retrieved from the indexed book?
- **Factual Correctness**: Is the statement an objective truth about the physical universe?

BookRAG AI operates as a retrieval and question-answering assistant for **books**. The book's retrieved evidence is the source of truth for the system. The NLI model evaluates whether the generative model synthesized an answer strictly faithful to the book passages, or whether it extrapolated unsupported propositions (hallucination).

### The Grounded Question Answering Pipeline
```
User Query
    ↓
SearchService (FAISS Dense Retrieval + Cross-Encoder Precision Reranking)
    ↓
Retrieved Evidence Passages
    ↓
GenerationService (FLAN-T5 Abstractive Synthesis)
    ↓
Raw Generated Answer
    ↓
ClaimDecomposer (Deterministic Sentence-Level Proposition Extraction)
    ↓
NLI Pair Construction (Premise = Evidence Passage, Hypothesis = Extracted Claim)
    ↓
NLIModel (cross-encoder/nli-deberta-v3-base Batched Forward Pass)
    ↓
Claim-Level Classification (Entailed / Contradicted / Conflicted / Unsupported)
    ↓
Safe Decision Policy (All claims supported? Contradictions present?)
    ↓
Safe Final Response (Grounded Answer OR Safe Refusal with Diagnostic Metadata)
```

### Natural Language Inference (NLI) Model
We employ `cross-encoder/nli-deberta-v3-base` through `sentence_transformers.CrossEncoder`:
- **Dynamic Label Discovery**: Rather than hardcoding class indices, the wrapper inspects `model.config.id2label` at load time to dynamically discover class mappings for:
  - `entailment`: The evidence passage logically guarantees or strongly supports the claim.
  - `contradiction`: The evidence passage directly refutes or is incompatible with the claim.
  - `neutral`: The evidence passage provides insufficient information to confirm or deny the claim.
- **Inference Mode & Caching**: Models are loaded once per process using thread-safe singleton caching under `torch.inference_mode()`.

### Deterministic Claim Decomposition
Before validation, generated answers are split into verifiable claim propositions using `ClaimDecomposer`:
- Splits text along sentence boundaries (`[.!?]\s+`).
- Normalizes whitespace (collapsing tabs, internal newlines, and multi-spaces).
- Filters out empty fragments and fragments shorter than `GROUNDING_MIN_CLAIM_LENGTH` (default: 3 characters).
- Assigns deterministic 0-based claim indices (`claim_index`).
- **Known Limitations**: Sentence splitting is a deterministic heuristic. Compound sentences containing multiple independent clauses are evaluated as a single claim unit.

### Claim-Level Classification Logic
For each extracted claim $c$ and top retrieved evidence chunks $E = \{e_1, \dots, e_K\}$:
1. Every `(e_i.source_text, c.claim_text)` pair is evaluated in batched NLI inference.
2. The strongest entailment score $e_{\max}$ and strongest contradiction score $c_{\max}$ across all evidence passages are identified along with source provenance.
3. Decision criteria using thresholds $\tau_e$ (`GROUNDING_ENTAILMENT_THRESHOLD=0.80`) and $\tau_c$ (`GROUNDING_CONTRADICTION_THRESHOLD=0.80`):
   - **`entailed`**: $e_{\max} \ge \tau_e$ AND $c_{\max} < \tau_c$.
   - **`contradicted`**: $c_{\max} \ge \tau_c$ AND $e_{\max} < \tau_e$.
   - **`conflicted`**: $e_{\max} \ge \tau_e$ AND $c_{\max} \ge \tau_c$ (different book passages present contradictory statements).
   - **`unsupported`**: $e_{\max} < \tau_e$ AND $c_{\max} < \tau_c$ (evidence is neutral or weakly related).

### Groundedness Score & Safe Decision Policy
The overall groundedness score is computed as:
$$\text{groundedness\_score} = \frac{\text{supported\_claims}}{\text{total\_claims}}$$
*(If no substantive claims exist, total claims is 0 and groundedness score is explicitly 0.0 with status `empty`).*

When `GROUNDING_REQUIRE_ALL_CLAIMS_SUPPORTED=true` (default):
- An answer is accepted **only** when all substantive claims are sufficiently supported (`supported_claims == total_claims`) and there are zero contradicted or conflicted claims.
- **Safe Refusal Behavior**: If any claim is unsupported, contradicted, or conflicted, the answer is safely refused:
  - `answer = null`
  - `answerable = false`
  - `grounded = false`
  - `groundedness_score`: accurately reflects partial support (e.g. 0.5)
  - `grounding_status`: `"unsupported"`, `"contradicted"`, or `"conflicted"`
  - `claims`: detailed claim breakdown with provenance
  - `evidence`: complete source evidence passages with provenance
  - `reason`: clear explanation (e.g., *"The generated answer contains claims that are not sufficiently supported by the retrieved book evidence."*)

---

## 9. Configuration Settings

| Parameter | Default | Constraint | Purpose |
| :--- | :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Valid HF model ID | Hugging Face model repository identifier. |
| `EMBEDDING_BATCH_SIZE` | `32` | `gt=0` | Number of text chunks encoded in parallel per forward pass. |
| `EMBEDDING_NORMALIZE` | `True` | boolean | Normalizes vectors to unit length ($L_2 = 1.0$). |
| `EMBEDDING_DEVICE` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` | Hardware target; `"auto"` selects CUDA if available, else CPU. |
| `RETRIEVAL_DEFAULT_TOP_K` | `5` | `gt=0` | Default number of candidate chunks returned per query. |
| `RETRIEVAL_MAX_TOP_K` | `100` | `gt=0` | Maximum allowable top_k limit for search queries. |
| `INDEX_STORAGE_DIR` | `"data/indexes"` | Directory path | Local filesystem directory for saving/loading FAISS indexes. |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Valid HF model ID | Cross-Encoder reranker repository identifier. |
| `RERANKER_MAX_LENGTH` | `512` | `gt=0` | Maximum sequence length for reranker joint transformer. |
| `RERANKER_ENABLED` | `True` | boolean | Enables or disables Phase 6 Cross-Encoder reranking. |
| `QA_MODEL_NAME` | `deepset/roberta-base-squad2` | Valid HF model ID | Extractive QA span selection model. |
| `QA_TOP_K_EVIDENCE` | `5` | `gt=0` | Number of top reranked chunks evaluated by extractive QA. |
| `GENERATION_MODEL_NAME` | `google/flan-t5-base` | Valid HF model ID | Abstractive Seq2Seq generation model. |
| `GENERATION_MAX_INPUT_TOKENS` | `2048` | `gt=0` | Prompt token budget limit for context assembly. |
| `GENERATION_MAX_NEW_TOKENS` | `128` | `gt=0` | Maximum generated tokens for abstractive answer. |
| `GENERATION_NUM_BEAMS` | `4` | `gt=0` | Beam search width for deterministic generation. |
| `GROUNDING_MODEL_NAME` | `cross-encoder/nli-deberta-v3-base` | Valid HF model ID | NLI CrossEncoder model for premise-hypothesis scoring. |
| `GROUNDING_ENABLED` | `True` | boolean | Globally enables or disables NLI groundedness validation. |
| `GROUNDING_DEVICE` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` | Execution device for NLI inference. |
| `GROUNDING_ENTAILMENT_THRESHOLD` | `0.80` | `0.0 <= x <= 1.0` | Minimum entailment probability to consider a claim supported. |
| `GROUNDING_CONTRADICTION_THRESHOLD` | `0.80` | `0.0 <= x <= 1.0` | Minimum contradiction probability to flag a contradiction. |
| `GROUNDING_TOP_K_EVIDENCE` | `5` | `gt=0` | Number of top evidence passages evaluated per claim. |
| `GROUNDING_REQUIRE_ALL_CLAIMS_SUPPORTED` | `True` | boolean | Enforces strict safe refusal if any claim is ungrounded. |
| `GROUNDING_MIN_CLAIM_LENGTH` | `3` | `gt=0` | Minimum character length for extracted claim sentences. |

---

## 10. How to Run the Backend

With the virtual environment activated:

```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 11. How to Run Tests

Run pytest from the `backend` directory:

```bash
cd backend
.\.venv\Scripts\pytest.exe -v
```

All **239 tests** will run, covering:
- **Phase 0 (5 tests)**: FastAPI initialization, settings, logging, health check probe.
- **Phase 1 (16 tests)**: PDF opening, page counts, 1-based page numbers, text extraction, empty/low-text diagnostics, error handling.
- **Phase 2 (27 tests)**: Conservative cleaning, safe dehyphenation, paragraph preservation, sentence-aware chunking, overlap control, chunk immutability.
- **Phase 3 (20 tests)**: Model loading, 384-dimensional output verification, batch inference order preservation, numerical $L_2$ unit normalization ($\|v\| \approx 1.0$), provenance survival, chunk immutability, determinism.
- **Phase 4 (23 tests)**: FAISS IndexFlatIP initialization, slot count synchronization, top-K boundaries, dimension validation, disk persistence and reload.
- **Phase 5 (18 tests)**: SearchService orchestration, query validation, top_k overrides, document filtering, HTTP 404 handling.
- **Phase 6 (49 tests)**: CrossEncoder model loading, device resolution, candidate pool scoring, provenance retention, rank synchronization.
- **Phase 7 (24 tests)**: RoBERTa SQuAD2 extractive QA, sliding window, answer span extraction, unanswerability thresholds.
- **Phase 8 (23 tests)**: FLAN-T5 abstractive generation, EvidenceBuilder budgeting, prompt formatting, empty-evidence handling, beam search.
- **Phase 9 (34 tests)**: NLI model wrapper, dynamic id2label mapping, sentence-level claim decomposition, pairwise NLI validation, threshold boundaries, safe refusal decision policy, end-to-end orchestration, and API endpoints.

---

## 12. API Endpoints

### 1. Health Probe
- **Method**: `GET`
- **Path**: `/api/v1/health`

### 2. Semantic Search (Two-Stage Retrieval)
- **Method**: `POST`
- **Path**: `/api/v1/search`
- **Request Body**: `{"query": "What is backpropagation?", "top_k": 5, "enable_reranking": true}`

### 3. Extractive Question Answering
- **Method**: `POST`
- **Path**: `/api/v1/qa`
- **Request Body**: `{"query": "Who popularized backpropagation?", "top_k": 5}`

### 4. Abstractive Question Answering (FLAN-T5)
- **Method**: `POST`
- **Path**: `/api/v1/answer`
- **Request Body**: `{"query": "Explain how backpropagation computes gradients.", "top_k": 5}`

### 5. Grounded Abstractive QA with Hallucination Control (Phase 9)
- **Method**: `POST`
- **Path**: `/api/v1/grounded-answer`
- **Request Body**:
```json
{
  "query": "How does backpropagation compute gradients in deep networks?",
  "top_k": 5,
  "require_all_claims_supported": true,
  "entailment_threshold": 0.80,
  "contradiction_threshold": 0.80
}
```

---

## 13. Future Roadmap

| Phase | Milestone | Status | Focus Areas |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Foundation | **Complete** | Repository structure, configuration, logging, health API, test suite. |
| **Phase 1** | Document Ingestion | **Complete** | PyMuPDF parser, page-aware data models, diagnostics, deterministic test fixtures. |
| **Phase 2** | Text Cleaning & Chunking | **Complete** | Conservative text cleaning, dehyphenation, paragraph/sentence-aware chunking, provenance. |
| **Phase 3** | Semantic Embeddings | **Complete** | Sentence Transformers, all-MiniLM-L6-v2, 384d vectors, L2 normalization, batch inference. |
| **Phase 4** | Vector Retrieval (FAISS) | **Complete** | IndexFlatIP, VectorToChunkMapping, top-K search, document isolation, disk persistence. |
| **Phase 5** | Semantic Search API | **Complete** | SearchService, Pydantic contracts, thin FastAPI endpoint `POST /api/v1/search`. |
| **Phase 6** | Cross-Encoder Reranking | **Complete** | Precision reranker (`ms-marco-MiniLM-L-6-v2`), two-stage retrieval, candidate_k pool. |
| **Phase 7** | Extractive Question Answering | **Complete** | RoBERTa SQuAD2 span extraction, sliding window, SQuAD 2.0 unanswerability, `POST /api/v1/qa`. |
| **Phase 8** | Abstractive QA (Generation) | **Complete** | FLAN-T5 abstractive synthesis, EvidenceBuilder context budgeting, prompt grounding, `POST /api/v1/answer`. |
| **Phase 9** | Groundedness & Evaluation | **Complete** | DeBERTa-v3 NLI model wrapper, dynamic id2label discovery, claim decomposition, safe decision policy, `POST /api/v1/grounded-answer`. |
| **Phase 10** | Production Hardening & UI | Planned | React frontend, PostgreSQL + pgvector, Redis task queues, Docker deployment. |



