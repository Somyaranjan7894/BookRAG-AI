# BookRAG AI

A research-oriented, production-quality Retrieval-Augmented Generation (RAG) system engineered from first principles for querying and analyzing complete book manuscripts and multi-chapter documents.

---

## 1. What is BookRAG AI?

BookRAG AI is a modular RAG platform specifically tailored to the unique challenges of long-form literature and technical texts. Unlike simple document Q&A prototypes that slice text into naive fixed-size windows and rely on third-party black-box wrappers, BookRAG AI is designed to explicitly implement each component of the retrieval and generation pipeline.

Key capabilities planned for the platform include:
- Preserving page numbers, chapter hierarchies, and contextual book metadata.
- Intelligent document cleaning and structural chunking.
- Semantic vector retrieval and hybrid lexical/dense search.
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
 [ Query Service ]   [ Ingestion & Text Processing ]
        │                     │
        ▼                     ▼
 [ Reranker & QA ]     [ Intelligent Chunker & Embedder ]
        │                     │
        ▼                     ▼
 [ PostgreSQL + pgvector / FAISS Index ]
```

### Architectural Layering:
- **API Layer (`backend/app/api/`)**: Thin controllers handling request validation, routing, and HTTP status codes.
- **Service Layer (`backend/app/services/`)**: Core domain workflows:
  - `services/pdf/`: Safe ingestion, validation, and page-aware PDF representation.
  - `services/text/`: Conservative text normalization and paragraph/sentence-aware intelligent chunking.
- **Data & Repository Layer (`backend/app/repositories/` & `backend/app/models/`)**: Abstracted persistence for book metadata, chunks, and index mappings (reserved for future database phases).
- **Core Platform (`backend/app/core/`)**: Cross-cutting concerns including centralized settings, structured logging, and unified error handling.

---

## 4. Current Phase Scope: Phase 2 Complete

This repository has completed **Phase 0 (Foundation)**, **Phase 1 (PDF Ingestion)**, and **Phase 2 (Text Cleaning & Intelligent Chunking)**.

### What is implemented:
- **Repository & Runtime Foundation (Phase 0)**:
  - Repository layout, virtual environment, and Git configuration.
  - Centralized settings with `pydantic-settings` and `.env.example`.
  - Configurable application logging.
  - Structured HTTP error handling and unhandled exception safety.
  - FastAPI application initialization with `/api/v1` versioning.
  - Operational health check endpoint: `GET /api/v1/health`.
- **PDF Ingestion & Page-Aware Representation (Phase 1)**:
  - Low-level PDF parser abstraction using PyMuPDF (`pymupdf>=1.25.0`).
  - High-level `PDFIngestionService` for safe document opening, validation, and metadata extraction.
  - 1-based page numbering preserving page provenance for future citations.
  - Exact raw text extraction with per-page and document-wide character and word counts.
  - Extraction diagnostics identifying empty and low-text pages without crashing the ingestion process.
  - Controlled domain exceptions (`PDFNotFoundError`, `InvalidPDFError`, `PDFExtractionError`).
  - Deterministic document ID generation derived from content SHA-256 hashes.
- **Text Cleaning & Intelligent Chunking (Phase 2)**:
  - Conservative, deterministic `TextCleaner`:
    - Normalizes line endings (CRLF/CR -> LF).
    - Repairs line-break hyphenation (e.g. `intel-\nligence` -> `intelligence`) while strictly preserving genuine compound words (e.g. `state-of-the-art`, `Smith-Jones`).
    - Collapses excessive horizontal whitespace and normalizes repeated blank lines (3+ newlines -> 2 newlines).
    - Preserves numbers, equations, code-like structures, and bullet/numbered outlines.
    - Strips outer whitespace without altering authorial meaning.
  - Paragraph-first, sentence-aware `Chunker`:
    - Treats paragraphs as primary semantic units.
    - Splitting hierarchy: Paragraph -> Sentences (protecting abbreviations like `Dr.`, `Prof.`, `e.g.`) -> Word/Hard character boundary fallback.
    - Configurable `target_size` (1200), `max_size` (1600), and `overlap` (200).
    - Bounded semantic overlap between adjacent chunks on the same page.
    - Strict page boundaries: pages are chunked independently to prevent cross-page provenance ambiguity.
    - Deterministic, debuggable chunk IDs (`{document_id}_p{page_number:03d}_c{chunk_index:04d}`).
  - Non-destructive `TextProcessingService` that processes full documents while keeping raw `page.text` immutable.
  - Complete automated test suite: **48 unit, validation, and API integration tests** passing with 100% success rate.

### Explicit Architectural Boundaries:
- **No Embeddings or Vector DBs**: No Sentence Transformers, PyTorch, FAISS, pgvector, or Chroma.
- **No Rerankers or QA Inference**: No cross-encoders, LLM calls, or answer generation.
- **No Database Persistence or Background Tasks**: No PostgreSQL, Redis, or Celery.
- **Engineering Defaults**: Default chunk sizes (`target_size = 1200`, `max_size = 1600`, `overlap = 200`) are initial engineering values that will later be quantitatively benchmarked against retrieval recall in evaluation phases.

---

## 5. Technology Stack

### Backend (Current Phase 2):
- **Language**: Python 3.11+ (Tested on Python 3.13.7)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/) (>= 0.115.0)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (>= 0.32.0)
- **PDF Extraction**: [PyMuPDF](https://pymupdf.readthedocs.io/) (>= 1.25.0)
- **Configuration & Validation**: [Pydantic v2](https://docs.pydantic.dev/) & [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **Testing**: [pytest](https://docs.pytest.org/) & [HTTPX](https://www.python-httpx.org/)

### Future Planned Stack:
- **Frontend**: React, TypeScript, Vite, TailwindCSS
- **NLP & Embedding Models**: Hugging Face Transformers, Sentence Transformers, PyTorch
- **Vector Storage**: FAISS (early phases) transitioning to PostgreSQL + pgvector
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
│   │   │   ├── chunk.py                   # Chunk, ChunkingConfig, and request/response models
│   │   │   ├── document.py                # Document, Page, Metadata Pydantic models
│   │   │   └── health.py                  # Health check Pydantic schemas
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── pdf/                       # Phase 1 Ingestion Service
│   │   │   │   ├── __init__.py
│   │   │   │   ├── exceptions.py          # Domain-specific PDF ingestion exceptions
│   │   │   │   ├── ingestion.py           # Orchestration, validation, diagnostics
│   │   │   │   └── parser.py              # PyMuPDF-specific extraction mechanics
│   │   │   └── text/                      # Phase 2 Text Processing Service
│   │   │       ├── __init__.py
│   │   │       ├── chunker.py             # Paragraph & sentence-aware intelligent chunker
│   │   │       ├── cleaner.py             # Conservative deterministic text cleaner
│   │   │       ├── exceptions.py          # Text processing domain exceptions
│   │   │       └── processor.py           # Document-level multi-page text processing
│   │   ├── __init__.py
│   │   └── main.py                        # FastAPI application entry point
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                    # Pytest client and deterministic PDF fixtures
│   │   ├── fixtures/
│   │   │   ├── __init__.py
│   │   │   └── pdf_factory.py             # Synthetic, reproducible PDF generators
│   │   ├── test_health.py                 # Startup and health check tests (Phase 0)
│   │   ├── test_pdf_ingestion.py          # Phase 1 ingestion, validation, and API tests
│   │   └── test_text_processing.py        # Phase 2 cleaning, chunking, and overlap tests
│   ├── requirements.txt                   # Phase 0, 1, 2 dependencies
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

## 7. Text Cleaning vs. Intelligent Chunking

```
Raw Page Text (from Phase 1)
           │
           ▼
[ Conservative Text Cleaner ]
  ├── 1. Line ending normalization (CRLF/CR -> LF)
  ├── 2. Safe line-break dehyphenation ("intel-\nligence" -> "intelligence")
  ├── 3. Horizontal whitespace collapse per line
  ├── 4. Paragraph separation preservation (\n\n)
  └── 5. Code, math, and list structure preservation
           │
           ▼
[ Intelligent Chunker ]
  ├── 1. Identify natural paragraph boundaries
  ├── 2. Paragraph fits in max_size? Keep intact as atomic unit
  ├── 3. Paragraph > max_size? Split into sentences (protecting abbreviations)
  ├── 4. Sentence > max_size? Fall back to word / character slices
  ├── 5. Accumulate units up to target_size (default 1200 chars)
  ├── 6. Emit chunk (max_size <= 1600 chars)
  ├── 7. Apply bounded tail overlap (default 200 chars) to next chunk
  └── 8. Guarantee independent page boundaries (no cross-page leakage)
           │
           ▼
[ Structured Chunk Collection ]
  ├── chunk_id: "doc_3b364081b1aeaafc_p001_c0000"
  ├── document_id: "doc_3b364081b1aeaafc"
  ├── page_number: 1
  ├── text: "..."
  ├── char_count: 1184
  └── word_count: 172
```

---

## 8. Chunk Configuration & Provenance Design

### Configuration Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `target_size` | `1200` characters | Soft maximum target. The chunker accumulates units up to this limit before considering boundary emission. |
| `max_size` | `1600` characters | Hard maximum ceiling. No chunk will exceed this size under any circumstance. |
| `overlap` | `200` characters | Semantic context overlap prepended to the subsequent chunk on the same page. |

> **Note on Chunk Sizes**: These defaults represent practical, conservative engineering baselines for English prose. In Phase 8, chunk sizing will be subjected to quantitative retrieval benchmarking (hit rate, MRR, citation fidelity) to determine optimal domain configurations.

### Provenance Guarantee

Each chunk preserves unambiguous provenance:
- **`document_id`**: Invariant identifier from ingestion.
- **`page_number`**: 1-based source page number. Chunks never span across page boundaries, ensuring exact page citations.
- **`chunk_id`**: Deterministic format: `{document_id}_p{page_number:03d}_c{chunk_index:04d}` (e.g. `doc_3b364081b1aeaafc_p001_c0000`).

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

All **48 tests** will run, covering:
- Phase 0: FastAPI initialization, settings, logging, health check probe.
- Phase 1: PDF opening, page counts, 1-based page numbers, text extraction, empty/low-text diagnostics, error handling.
- Phase 2:
  - Line ending and whitespace normalization.
  - Safe dehyphenation (line-break repair vs. compound word preservation).
  - Paragraph preservation and sentence-aware splitting.
  - Hard boundary fallback for oversized continuous sentences.
  - Bounded overlap and non-leakage across page boundaries.
  - Chunk ID determinism, uniqueness, and strict provenance.
  - Immutability of raw `Page.text`.
  - Development API endpoints for cleaning and chunking.

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
- **Response**: `{"original_length": 34, "cleaned_length": 29, "cleaned_text": "Raw text with intelligence."}`

### 4. Page Chunking (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/chunks/chunk-page`
- **Request Body**:
```json
{
  "document_id": "book_001",
  "page_number": 1,
  "text": "Paragraph 1 text...\n\nParagraph 2 text...",
  "config": {
    "target_size": 1200,
    "max_size": 1600,
    "overlap": 200
  }
}
```

---

## 12. Future Roadmap

| Phase | Milestone | Status | Focus Areas |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Foundation | **Complete** | Repository structure, configuration, logging, health API, test suite. |
| **Phase 1** | Document Ingestion | **Complete** | PyMuPDF parser, page-aware data models, diagnostics, deterministic test fixtures. |
| **Phase 2** | Text Cleaning & Chunking | **Complete** | Conservative text cleaning, dehyphenation, paragraph/sentence-aware chunking, provenance. |
| **Phase 3** | Embeddings & Indexing | Planned | Sentence Transformers, dense embeddings, FAISS indexing. |
| **Phase 4** | Retrieval & Reranking | Planned | Hybrid lexical + semantic retrieval, cross-encoder reranking. |
| **Phase 5** | Extractive QA | Planned | Span extraction, page-level citation mapping. |
| **Phase 6** | Abstractive QA | Planned | Synthesis, groundedness verification, hallucination checks. |
| **Phase 7** | Question Generation | Planned | User-controlled question synthesis across chapters and difficulty levels. |
| **Phase 8** | Web UI & Evaluation | Planned | React + Vite UI, Relevant Matching Board, RAG benchmark metrics. |
| **Phase 9** | Production Hardening | Planned | PostgreSQL + pgvector, Redis task queues, Docker Compose deployment. |
