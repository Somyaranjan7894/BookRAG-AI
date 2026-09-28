# BookRAG AI

A research-oriented, production-quality Retrieval-Augmented Generation (RAG) system engineered from first principles for querying and analyzing complete book manuscripts and multi-chapter documents.

---

## 1. What is BookRAG AI?

BookRAG AI is a modular RAG platform specifically tailored to the unique challenges of long-form literature and technical texts. Unlike simple document Q&A prototypes that slice text into naive fixed-size windows and rely on third-party black-box wrappers, BookRAG AI is designed to explicitly implement each component of the retrieval and generation pipeline.

Key capabilities planned for the platform include:
- Preserving page numbers, chapter hierarchies, and contextual book metadata.
- Intelligent document cleaning and structural chunking.
- Dense semantic vector embeddings preserving full chunk provenance.
- Vector indexing and hybrid lexical/dense search.
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
 [ FAISS Vector Index / PostgreSQL + pgvector ]
```

### Architectural Layering:
- **API Layer (`backend/app/api/`)**: Thin controllers handling request validation, routing, and HTTP status codes.
- **Service Layer (`backend/app/services/`)**: Core domain workflows:
  - `services/pdf/`: Safe ingestion, validation, and page-aware PDF representation.
  - `services/text/`: Conservative text normalization and paragraph/sentence-aware intelligent chunking.
  - `services/embeddings/`: Dense semantic vector representations with device negotiation and provenance retention.
- **Data & Repository Layer (`backend/app/repositories/` & `backend/app/models/`)**: Abstracted persistence for book metadata, chunks, and index mappings (reserved for future database phases).
- **Core Platform (`backend/app/core/`)**: Cross-cutting concerns including centralized settings, structured logging, and unified error handling.

---

## 4. Current Phase Scope: Phase 3 Complete

This repository has completed **Phase 0 (Foundation)**, **Phase 1 (PDF Ingestion)**, **Phase 2 (Text Cleaning & Chunking)**, and **Phase 3 (Semantic Embeddings)**.

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
  - Complete automated test suite: **68 unit, validation, and API integration tests** passing with 100% success rate.

### Explicit Architectural Boundaries:
- **Embedding Generation $\neq$ Vector Search**: Phase 3 generates dense vectors. FAISS indexing and vector retrieval are intentionally reserved for Phase 4.
- **No Vector Databases or Indexes**: No FAISS, pgvector, or Chroma.
- **No Rerankers or QA Inference**: No cross-encoders, LLM calls, or question generation.
- **No Database Persistence or Background Tasks**: No PostgreSQL, Redis, or Celery.

---

## 5. Technology Stack

### Backend (Current Phase 3):
- **Language**: Python 3.11+ (Tested on Python 3.13.7)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/) (>= 0.115.0)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (>= 0.32.0)
- **PDF Extraction**: [PyMuPDF](https://pymupdf.readthedocs.io/) (>= 1.25.0)
- **Embeddings & NLP**: [Sentence Transformers](https://www.sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`), PyTorch (>= 2.2.0)
- **Configuration & Validation**: [Pydantic v2](https://docs.pydantic.dev/) & [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **Testing**: [pytest](https://docs.pytest.org/) & [HTTPX](https://www.python-httpx.org/)

### Future Planned Stack:
- **Frontend**: React, TypeScript, Vite, TailwindCSS
- **Vector Storage**: FAISS (Phase 4) transitioning to PostgreSQL + pgvector
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
│   │   │       │   └── health.py          # GET /api/v1/health implementation
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
│   │   │   └── health.py                  # Health check Pydantic schemas
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
│   │   │   └── embeddings/                # Phase 3 Semantic Embeddings Service
│   │   │       ├── __init__.py
│   │   │       ├── exceptions.py          # Embedding domain exceptions
│   │   │       ├── model.py               # EmbeddingModel wrapper & instance registry
│   │   │       └── service.py             # EmbeddingService & batch inference
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
│   │   └── test_text_processing.py        # Phase 2 cleaning, chunking, and overlap tests
│   ├── requirements.txt                   # Phase 0, 1, 2, 3 dependencies
│   └── .env.example                       # Non-sensitive configuration template
│
├── frontend/                              # Reserved for future React application
│
├── data/                                  # Runtime data directories (version controlled via .gitkeep)
│   ├── uploads/                           # Destination for uploaded book PDFs
│   ├── processed/                         # Destination for extracted/chunked artifacts
│   └── indexes/                           # Destination for vector indexes
│
├── docs/                                  # Architectural specifications and design records
├── evaluation/                            # Benchmarking datasets and evaluation scripts
│
├── .gitignore                             # Ignore rules for caches, env, data, models, *.pdf
├── README.md                              # Project documentation
└── docker-compose.yml                     # Minimal deployment skeleton for future phases
```

---

## 7. Semantic Embedding Architecture

```
Chunk (from Phase 2)
  ├── chunk_id: "doc_3b364081b1aeaafc_p001_c0000"
  ├── document_id: "doc_3b364081b1aeaafc"
  ├── page_number: 1
  └── text: "Traditional information retrieval systems have long relied on keyword matching..."
       │
       ▼
[ EmbeddingService.embed_chunks(chunks) ]
  ├── Input validation (rejects None, malformed, empty/whitespace text)
  ├── SentenceTransformer model resolution (cached singleton)
  ├── Hardware negotiation (CUDA if present, otherwise CPU)
  ├── Batch encoding (batch_size = 32)
  └── L2 normalization (||v|| ≈ 1.0)
       │
       ▼
[ EmbeddingRecord ]
  ├── chunk_id: "doc_3b364081b1aeaafc_p001_c0000"     <-- Provenance Preserved!
  ├── document_id: "doc_3b364081b1aeaafc"             <-- Provenance Preserved!
  ├── page_number: 1                                   <-- Provenance Preserved!
  ├── text: "Traditional information retrieval..."     <-- Source Text Preserved!
  ├── embedding: [-0.0418, 0.0812, ..., 0.0125]       <-- 384-dimensional dense vector
  ├── dimension: 384
  ├── model_name: "sentence-transformers/all-MiniLM-L6-v2"
  ├── device: "cpu"
  └── normalized: true
```

### Why all-MiniLM-L6-v2?
`all-MiniLM-L6-v2` maps sentences and paragraphs into a 384-dimensional dense vector space. It is specifically optimized for semantic search, offering an outstanding balance between inference speed (~5x faster than BERT-base), compact vector storage footprint (384 floats = 1,536 bytes per chunk), and strong retrieval quality.

### Normalization
When `normalize_embeddings=True`, vectors are normalized such that their Euclidean norm ($L_2$) equals 1.0:
$$\|v\|_2 = \sqrt{\sum_{i=1}^{384} v_i^2} \approx 1.0$$
This guarantees that the dot product of two normalized vectors equals their cosine similarity, enabling maximum search efficiency during Phase 4 vector retrieval:
$$\text{Cosine Similarity}(u, v) = u \cdot v$$

---

## 8. Embedding Configuration

| Parameter | Default | Constraint | Purpose |
| :--- | :--- | :--- | :--- |
| `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Valid HF model ID | Hugging Face model repository identifier. |
| `batch_size` | `32` | `gt=0` | Number of text chunks encoded in parallel per forward pass. |
| `normalize_embeddings` | `True` | boolean | Normalizes vectors to unit length ($L_2 = 1.0$). |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` | Hardware target; `"auto"` selects CUDA if available, else CPU. |

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

All **68 tests** will run, covering:
- Phase 0: FastAPI initialization, settings, logging, health check probe.
- Phase 1: PDF opening, page counts, 1-based page numbers, text extraction, empty/low-text diagnostics, error handling.
- Phase 2: Conservative cleaning, safe dehyphenation, paragraph preservation, sentence-aware chunking, overlap control, chunk immutability.
- Phase 3:
  - Model loading and 384-dimensional output verification.
  - Single chunk and batch chunk embedding generation.
  - Output order preservation across batches.
  - $L_2$ unit normalization verification ($\|v\| \approx 1.0$).
  - Strict provenance survival (`chunk_id`, `document_id`, `page_number`).
  - Raw `Chunk` object immutability.
  - Input validation (rejection of None, empty text, whitespace-only chunks).
  - Determinism across repeated inference runs.
  - Hardware device resolution (`auto` $\rightarrow$ `cpu`/`cuda`).
  - Development API endpoints (`POST /api/v1/embeddings/embed-chunk` and `embed-chunks`).

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
- **Request Body**: `{"text": "Raw  text with intel-\nligence."}`

### 4. Page Chunking (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/chunks/chunk-page`
- **Request Body**: `{"document_id": "b1", "page_number": 1, "text": "Text..."}`

### 5. Single Chunk Embedding (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/embeddings/embed-chunk`
- **Request Body**:
```json
{
  "chunk_id": "doc_001_p001_c0001",
  "document_id": "doc_001",
  "page_number": 1,
  "text": "Dense representations enable semantic search."
}
```

### 6. Batch Chunk Embedding (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/embeddings/embed-chunks`
- **Request Body**:
```json
{
  "chunks": [
    {
      "chunk_id": "doc_001_p001_c0001",
      "document_id": "doc_001",
      "page_number": 1,
      "text": "First passage on page one."
    },
    {
      "chunk_id": "doc_001_p002_c0002",
      "document_id": "doc_001",
      "page_number": 2,
      "text": "Second passage on page two."
    }
  ]
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
| **Phase 4** | Retrieval & Indexing | Planned | FAISS vector indexing, similarity search, top-k retrieval, evaluation. |
| **Phase 5** | Extractive QA | Planned | Span extraction, page-level citation mapping. |
| **Phase 6** | Abstractive QA | Planned | Synthesis, groundedness verification, hallucination checks. |
| **Phase 7** | Question Generation | Planned | User-controlled question synthesis across chapters and difficulty levels. |
| **Phase 8** | Web UI & Evaluation | Planned | React + Vite UI, Relevant Matching Board, RAG benchmark metrics. |
| **Phase 9** | Production Hardening | Planned | PostgreSQL + pgvector, Redis task queues, Docker Compose deployment. |
