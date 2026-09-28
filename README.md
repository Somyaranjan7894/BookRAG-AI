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

## 4. Current Phase 0 Scope

This repository is currently at **Phase 0: Foundation and Architecture**.

### What is implemented in Phase 0:
- Repository layout and directory conventions.
- Git configuration and comprehensive `.gitignore` rules.
- Centralized configuration management using `pydantic-settings` with `.env.example`.
- Configurable application logging.
- Structured HTTP error handling and unhandled exception safety.
- FastAPI application initialization with `/api/v1` versioning.
- Operational health check endpoint: `GET /api/v1/health`.
- Automated test suite using `pytest` and `httpx`.
- Minimal skeleton configuration for future Docker setups.

### What is intentionally NOT implemented in Phase 0:
To guarantee deliberate, incremental development, future components remain **intentionally unbuilt**:
- No PDF parsing, text extraction, or OCR.
- No text cleaning or chunking logic.
- No embeddings or Sentence Transformers.
- No vector stores (FAISS, pgvector, Chromadb, etc.).
- No relational database schemas or migrations.
- No background task workers (Redis, Celery).
- No reranking models or Cross-Encoders.
- No Extractive or Abstractive QA inference.
- No LLM answer synthesis or question generation.
- No frontend React application.
- No user authentication or authorization.

---

## 5. Technology Stack

### Backend (Current Phase 0):
- **Language**: Python 3.11+ (Tested on Python 3.13)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/) (>= 0.115.0)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (>= 0.32.0)
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
│   │   │       │   └── health.py          # GET /api/v1/health implementation
│   │   │       ├── __init__.py
│   │   │       └── router.py              # Assembles version 1 routes
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py                  # Pydantic BaseSettings management
│   │   │   ├── errors.py                  # Safe global exception handlers
│   │   │   └── logging.py                 # Standardized logging setup
│   │   ├── models/                        # Domain models (Phase 0 placeholder)
│   │   ├── repositories/                  # Persistence repositories (Phase 0 placeholder)
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── health.py                  # Health check Pydantic schemas
│   │   ├── services/                      # Business logic services (Phase 0 placeholder)
│   │   ├── __init__.py
│   │   └── main.py                        # FastAPI application entry point
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                    # Pytest client fixtures
│   │   └── test_health.py                 # Startup and health check tests
│   ├── requirements.txt                   # Phase 0 dependencies only
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
├── .gitignore                             # Ignore rules for caches, env, data, models
├── README.md                              # Project documentation
└── docker-compose.yml                     # Minimal deployment skeleton for future phases
```

---

## 7. Backend Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13 installed.
- Git installed.

### Virtual Environment Configuration

1. Open a terminal and navigate to the project directory:
   ```bash
   cd c:/Book_Rag_AI
   ```

2. Create a virtual environment:
   ```bash
   python -m venv backend/.venv
   ```

3. Activate the virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     .\backend\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     backend\.venv\Scripts\activate.bat
     ```
   - **Linux / macOS**:
     ```bash
     source backend/.venv/bin/activate
     ```

4. Install Phase 0 dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

5. (Optional) Configure environment variables:
   ```bash
   cp backend/.env.example backend/.env
   ```

---

## 8. How to Run the Backend

With the virtual environment activated and working directory at `backend`:

```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Or from the project root:
```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Access interactive documentation in your browser:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 9. How to Run Tests

Ensure the virtual environment is activated, then run pytest from the `backend` directory:

```bash
cd backend
pytest -v
```

All tests should pass, confirming application initialization, router resolution, health response structure, and structured error responses.

---

## 10. Health Endpoint

### Endpoint Details
- **Method**: `GET`
- **Path**: `/api/v1/health`
- **Authentication**: None (public monitoring probe)

### Example Request
```bash
curl -X GET http://127.0.0.1:8000/api/v1/health
```

### Example Response
```json
{
  "status": "ok",
  "service": "BookRAG AI"
}
```

---

## 11. Future Roadmap

BookRAG AI will be developed in rigorous, verifiable phases:

| Phase | Milestone | Focus Areas |
| :--- | :--- | :--- |
| **Phase 0** | **Foundation (Current)** | Repository structure, configuration, logging, health API, test suite. |
| **Phase 1** | Document Ingestion | PDF page extraction, layout preservation, text cleaning. |
| **Phase 2** | Structural Chunking | Chapter-aware and semantic window chunking with page metadata. |
| **Phase 3** | Embeddings & Indexing | Sentence Transformers, dense embeddings, FAISS indexing. |
| **Phase 4** | Retrieval & Reranking | Hybrid lexical + semantic retrieval, cross-encoder reranking. |
| **Phase 5** | Extractive QA | Span extraction, page-level citation mapping. |
| **Phase 6** | Abstractive QA | Synthesis, groundedness verification, hallucination checks. |
| **Phase 7** | Question Generation | User-controlled question synthesis across chapters and difficulty levels. |
| **Phase 8** | Web UI & Evaluation | React + Vite UI, Relevant Matching Board, RAG benchmark metrics. |
| **Phase 9** | Production Hardening | PostgreSQL + pgvector, Redis task queues, Docker Compose deployment. |
