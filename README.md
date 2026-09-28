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

## 4. Current Phase Scope: Phase 4 Complete

This repository has completed **Phase 0 (Foundation)**, **Phase 1 (PDF Ingestion)**, **Phase 2 (Text Cleaning & Chunking)**, **Phase 3 (Semantic Embeddings)**, and **Phase 4 (Vector Retrieval with FAISS)**.

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
  - Complete automated test suite: **91 unit, validation, and API integration tests** passing with 100% success rate.

### Explicit Architectural Boundaries:
- **Retrieval $\neq$ Question Answering**: Phase 4 retrieves candidate chunks based on semantic similarity. It does not synthesize answers, evaluate truthfulness, or generate citations.
- **No Rerankers or Cross-Encoders**: Cross-encoder precision reranking is reserved for future phases.
- **No LLM Generation or Prompt Assembly**: No FLAN-T5, OpenAI, or question answering models.
- **No Complex Databases or Distributed Queues**: No PostgreSQL, pgvector, Redis, or Celery.

---

## 5. Technology Stack

### Backend (Current Phase 4):
- **Language**: Python 3.11+ (Tested on Python 3.13.7)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/) (>= 0.115.0)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (>= 0.32.0)
- **PDF Extraction**: [PyMuPDF](https://pymupdf.readthedocs.io/) (>= 1.25.0)
- **Embeddings & NLP**: [Sentence Transformers](https://www.sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`), PyTorch (>= 2.2.0)
- **Vector Indexing & Retrieval**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu>=1.9.0`)
- **Configuration & Validation**: [Pydantic v2](https://docs.pydantic.dev/) & [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **Testing**: [pytest](https://docs.pytest.org/) & [HTTPX](https://www.python-httpx.org/)

### Future Planned Stack:
- **Frontend**: React, TypeScript, Vite, TailwindCSS
- **Reranker**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2` or similar)
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
│   │   │       │   └── retrieval.py       # POST /api/v1/retrieval/index & search
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
│   │   │   └── retrieval.py               # RetrievalResult, IndexMetadata, VectorMappingItem
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
│   │   │   └── retrieval/                 # Phase 4 FAISS Vector Retrieval Service
│   │   │       ├── __init__.py
│   │   │       ├── exceptions.py          # Retrieval & FAISS domain exceptions
│   │   │       ├── index.py               # VectorIndex wrapper & persistence
│   │   │       ├── mapping.py             # VectorToChunkMapping synchronization
│   │   │       └── service.py             # RetrievalService query search orchestrator
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

## 8. Configuration Settings

| Parameter | Default | Constraint | Purpose |
| :--- | :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Valid HF model ID | Hugging Face model repository identifier. |
| `EMBEDDING_BATCH_SIZE` | `32` | `gt=0` | Number of text chunks encoded in parallel per forward pass. |
| `EMBEDDING_NORMALIZE` | `True` | boolean | Normalizes vectors to unit length ($L_2 = 1.0$). |
| `EMBEDDING_DEVICE` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` | Hardware target; `"auto"` selects CUDA if available, else CPU. |
| `RETRIEVAL_DEFAULT_TOP_K` | `5` | `gt=0` | Default number of candidate chunks returned per query. |
| `RETRIEVAL_MAX_TOP_K` | `100` | `gt=0` | Maximum allowable top_k limit for search queries. |
| `INDEX_STORAGE_DIR` | `"data/indexes"` | Directory path | Local filesystem directory for saving/loading FAISS indexes. |

---

## 9. How to Run the Backend

With the virtual environment activated:

```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 10. How to Run Tests

Run pytest from the `backend` directory:

```bash
cd backend
.\.venv\Scripts\pytest.exe -v
```

All **91 tests** will run, covering:
- **Phase 0 (5 tests)**: FastAPI initialization, settings, logging, health check probe.
- **Phase 1 (16 tests)**: PDF opening, page counts, 1-based page numbers, text extraction, empty/low-text diagnostics, error handling.
- **Phase 2 (27 tests)**: Conservative cleaning, safe dehyphenation, paragraph preservation, sentence-aware chunking, overlap control, chunk immutability.
- **Phase 3 (20 tests)**: Model loading, 384-dimensional output verification, batch inference order preservation, numerical $L_2$ unit normalization ($\|v\| \approx 1.0$), provenance survival, chunk immutability, determinism.
- **Phase 4 (23 tests)**:
  - FAISS `IndexFlatIP` initialization with 384 dimensions.
  - Index creation, vector insertion, and slot count synchronization.
  - Incremental batch vector addition with contiguous slot mapping.
  - Semantic query search and rank-1 relevance verification.
  - Top-K boundaries (`top_k=1`, `top_k=3`, `top_k > ntotal`).
  - Strict provenance retention (`chunk_id`, `document_id`, `page_number`, `text`).
  - Dimension mismatch rejection on vector addition and query search.
  - Controlled empty index behavior (returns `[]` without error).
  - Unregistered index handling (`IndexNotFoundError`).
  - Document isolation and filtering across multi-document corpuses.
  - Disk persistence (`.faiss` + `.json`) and reload verification.
  - Corrupted metadata and vector count mismatch detection.
  - Determinism across repeated queries.
  - Query input validation (rejects empty / whitespace-only queries).
  - API development endpoints (`POST /api/v1/retrieval/index` and `POST /api/v1/retrieval/search`).

---

## 11. API Endpoints

### 1. Health Probe
- **Method**: `GET`
- **Path**: `/api/v1/health`

### 2. Document Ingestion (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/documents/ingest`
- **Request Body**: `{"file_path": "path/to/book.pdf"}`

### 3. Text Cleaning (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/chunks/clean`
- **Request Body**: `{"text": "Raw text with intel-\nligence."}`

### 4. Page Chunking (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/chunks/chunk-page`
- **Request Body**: `{"document_id": "b1", "page_number": 1, "text": "Text..."}`

### 5. Single Chunk Embedding (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/embeddings/embed-chunk`
- **Request Body**: `{"chunk_id": "doc_001_p001_c0001", "document_id": "doc_001", "page_number": 1, "text": "Text..."}`

### 6. Batch Chunk Embedding (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/embeddings/embed-chunks`

### 7. Vector Indexing (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/retrieval/index`
- **Request Body**:
```json
{
  "index_id": "book_intro_index",
  "document_id": "doc_ai",
  "records": [...]
}
```

### 8. Semantic Vector Search (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/retrieval/search`
- **Request Body**:
```json
{
  "query": "How do deep neural networks learn hierarchical representations?",
  "top_k": 5,
  "document_id": "doc_ai"
}
```

---

## 12. Future Roadmap

| Phase | Milestone | Status | Focus Areas |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Foundation | **Complete** | Repository structure, configuration, logging, health API, test suite. |
| **Phase 1** | Document Ingestion | **Complete** | PyMuPDF parser, page-aware data models, diagnostics, deterministic test fixtures. |
| **Phase 2** | Text Cleaning & Chunking | **Complete** | Conservative text cleaning, dehyphenation, paragraph/sentence-aware chunking, provenance. |
| **Phase 3** | Semantic Embeddings | **Complete** | Sentence Transformers, all-MiniLM-L6-v2, 384d vectors, L2 normalization, batch inference. |
| **Phase 4** | Vector Retrieval (FAISS) | **Complete** | IndexFlatIP, VectorToChunkMapping, top-K search, document isolation, disk persistence. |
| **Phase 5** | Extractive QA | Planned | Span extraction, page-level citation mapping. |
| **Phase 6** | Abstractive QA | Planned | Synthesis, groundedness verification, hallucination checks. |
| **Phase 7** | Question Generation | Planned | User-controlled question synthesis across chapters and difficulty levels. |
| **Phase 8** | Web UI & Evaluation | Planned | React + Vite UI, Relevant Matching Board, RAG benchmark metrics. |
| **Phase 9** | Production Hardening | Planned | PostgreSQL + pgvector, Redis task queues, Docker Compose deployment. |
