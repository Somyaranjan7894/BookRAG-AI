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
 [ Query Service ]   [ Ingestion Service ]
        │                     │
        ▼                     ▼
 [ Reranker & QA ]     [ Intelligent Chunker & Embedder ]
        │                     │
        ▼                     ▼
 [ PostgreSQL + pgvector / FAISS Index ]
```

### Architectural Layering:
- **API Layer (`backend/app/api/`)**: Thin controllers handling request validation, routing, and HTTP status codes.
- **Service Layer (`backend/app/services/`)**: Core domain workflows, orchestration of retrieval pipelines, and QA synthesis.
- **Data & Repository Layer (`backend/app/repositories/` & `backend/app/models/`)**: Abstracted persistence for book metadata, chunks, and index mappings.
- **Core Platform (`backend/app/core/`)**: Cross-cutting concerns including centralized settings, structured logging, and unified error handling.

---

## 4. Current Phase Scope: Phase 1 Complete

This repository has completed **Phase 0: Foundation and Architecture** and **Phase 1: PDF Ingestion and Page-Aware Document Representation**.

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
  - Minimal development API endpoint (`POST /api/v1/documents/ingest`).
  - Automated test suite with 21 unit and integration tests using deterministic test fixtures.

### Explicit Architectural Boundaries:
- **Text extraction is not OCR**: PyMuPDF extracts embedded digital text streams. Scanned image-only PDFs will produce empty-text diagnostics rather than trigger OCR.
- **No semantic text cleaning or chunking**: Raw text is preserved as extracted. Phase 2 will introduce structural cleaning and intelligent chunking.
- **Preserves page-level provenance**: Page numbers and boundaries are maintained throughout the ingestion data structures to enable page-level citations in later QA phases.
- **No mock implementations of future phases**: No vector stores, embeddings, database migrations, background task queues, or LLMs are present.

---

## 5. Technology Stack

### Backend (Current Phase 1):
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
│   │   │   ├── document.py                # Document, Page, Metadata Pydantic models
│   │   │   └── health.py                  # Health check Pydantic schemas
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── pdf/
│   │   │       ├── __init__.py
│   │   │       ├── exceptions.py          # Domain-specific PDF ingestion exceptions
│   │   │       ├── ingestion.py           # Orchestration, validation, diagnostics
│   │   │       └── parser.py              # PyMuPDF-specific extraction mechanics
│   │   ├── __init__.py
│   │   └── main.py                        # FastAPI application entry point
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                    # Pytest client and deterministic PDF fixtures
│   │   ├── fixtures/
│   │   │   ├── __init__.py
│   │   │   └── pdf_factory.py             # Synthetic, reproducible PDF generators
│   │   ├── test_health.py                 # Startup and health check tests
│   │   └── test_pdf_ingestion.py          # Phase 1 ingestion, validation, and API tests
│   ├── requirements.txt                   # Phase 0 & 1 dependencies
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

## 7. Data Representation & Extraction Model

During Phase 1, documents are parsed and represented in structured Pydantic models:

```json
{
  "document_id": "doc_3b364081b1aeaafc",
  "filename": "sample_book.pdf",
  "source_path": "C:\\Book_Rag_AI\\data\\uploads\\sample_book.pdf",
  "page_count": 3,
  "total_characters": 72,
  "total_words": 13,
  "metadata": {
    "title": "BookRAG AI Test Document",
    "author": "Antigravity Engineering",
    "subject": "Phase 1 Ingestion Verification",
    "creator": null,
    "producer": null,
    "creation_date": null,
    "mod_date": null,
    "custom": {}
  },
  "pages": [
    {
      "page_number": 1,
      "text": "BookRAG AI Phase 1\n",
      "char_count": 19,
      "word_count": 4,
      "has_text": true,
      "extraction_warning": null
    },
    {
      "page_number": 2,
      "text": "This is a PDF ingestion test.\n",
      "char_count": 30,
      "word_count": 6,
      "has_text": true,
      "extraction_warning": null
    }
  ],
  "warnings": []
}
```

---

## 8. Backend Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13 installed.
- Git installed.

### Virtual Environment Configuration

1. Open a terminal and navigate to the project directory:
   ```bash
   cd c:/Book_Rag_AI
   ```

2. Create and activate a virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv backend/.venv
     .\backend\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python -m venv backend/.venv
     source backend/.venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

4. (Optional) Configure environment variables:
   ```bash
   cp backend/.env.example backend/.env
   ```

---

## 9. How to Run the Backend

With the virtual environment activated and working directory at `backend`:

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

All 21 tests will run, covering:
- Application startup and metadata.
- Health endpoint status and schema.
- PDF opening, page counts, and 1-based page numbering.
- Text, character, and word count accuracy.
- Extraction diagnostics on empty and low-text pages without process crashes.
- Controlled error handling for missing files, corrupt files, and 0-page PDFs.
- Deterministic document ID generation and custom ID retention.
- Dev API endpoint functionality and HTTP status mappings.

---

## 11. API Endpoints

### 1. Health Probe
- **Method**: `GET`
- **Path**: `/api/v1/health`
- **Response**:
```json
{
  "status": "ok",
  "service": "BookRAG AI"
}
```

### 2. Document Ingestion (Development / Testing)
- **Method**: `POST`
- **Path**: `/api/v1/documents/ingest`
- **Request Body**:
```json
{
  "file_path": "data/uploads/sample.pdf",
  "document_id": "optional_custom_id"
}
```
- **Response (200 OK)**:
```json
{
  "status": "success",
  "document": { ... }
}
```

---

## 12. Future Roadmap

| Phase | Milestone | Status | Focus Areas |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Foundation | **Complete** | Repository structure, configuration, logging, health API, test suite. |
| **Phase 1** | Document Ingestion | **Complete** | PyMuPDF parser, page-aware data models, diagnostics, deterministic test fixtures. |
| **Phase 2** | Structural Chunking | Planned | Document cleaning, chapter/section identification, semantic window chunking. |
| **Phase 3** | Embeddings & Indexing | Planned | Sentence Transformers, dense embeddings, FAISS indexing. |
| **Phase 4** | Retrieval & Reranking | Planned | Hybrid lexical + semantic retrieval, cross-encoder reranking. |
| **Phase 5** | Extractive QA | Planned | Span extraction, page-level citation mapping. |
| **Phase 6** | Abstractive QA | Planned | Synthesis, groundedness verification, hallucination checks. |
| **Phase 7** | Question Generation | Planned | User-controlled question synthesis across chapters and difficulty levels. |
| **Phase 8** | Web UI & Evaluation | Planned | React + Vite UI, Relevant Matching Board, RAG benchmark metrics. |
| **Phase 9** | Production Hardening | Planned | PostgreSQL + pgvector, Redis task queues, Docker Compose deployment. |
