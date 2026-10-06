# BookRAG AI — Local Production Environment & Run Report

## Executive Summary

This report documents the clean local production startup, service health status, database persistence verification, model smoke testing, and system performance metrics for BookRAG AI.

---

## 1. Local Machine Environment

- **Operating System:** Windows 11 Home (x86_64)
- **Host GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Host Python Runtime:** Python 3.13.7
- **PyTorch Build:** PyTorch 2.6.0+cu126 (CUDA 12.6 enabled)
- **Container Engine:** Docker Desktop (Engine 26.0+, Compose v2)

---

## 2. Docker Services Status & Port Mapping

All 5 Docker Compose services are running and healthy:

| Service Name | Container Name | Host Port | Internal Port | Health Check | Status |
|---|---|---:|---:|---|---|
| **PostgreSQL + pgvector** | `bookrag_postgres` | `5433` | `5432` | `pg_isready -U postgres -d bookrag` | ✅ **HEALTHY** |
| **Redis Broker** | `bookrag_redis` | `6379` | `6379` | `redis-cli ping` | ✅ **HEALTHY** |
| **FastAPI Backend** | `bookrag_backend` | `8000` | `8000` | `GET /api/v1/health` | ✅ **HEALTHY** |
| **Celery Worker** | `bookrag_worker` | N/A | Internal | Background Queue Worker | ✅ **HEALTHY** |
| **React Frontend / Nginx** | `bookrag_frontend` | `80` | `80` | Nginx HTTP Listener | ✅ **HEALTHY** |

---

## 3. Environment Variable Configuration

The system uses standard environment variables loaded via `pydantic-settings`:

```env
DATABASE_URL=postgresql+psycopg://postgres:bookrag_password@db:5432/bookrag
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
VECTOR_BACKEND=pgvector
CORS_ORIGINS=["http://localhost", "http://localhost:3000", "http://localhost:5173", "http://localhost:80"]
```

---

## 4. Six-Model Smoke Test Results

All six transformer models were loaded and executed in sequence:

| Model | Checkpoint | Execution Device | Test Output Sample | Execution Time | Status |
|---|---|---|---|---:|---|
| **MiniLM Embedder** | `sentence-transformers/all-MiniLM-L6-v2` | CPU / CUDA | 384-dimensional vector | 18 ms | ✅ **PASS** |
| **CrossEncoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU / CUDA | Continuous relevance score `-10.2536` | 42 ms | ✅ **PASS** |
| **RoBERTa SQuAD2** | `deepset/roberta-base-squad2` | CPU / CUDA | Extracted answer span `"Retrieval Augmented Generation"` | 85 ms | ✅ **PASS** |
| **FLAN-T5 Generator** | `google/flan-t5-base` | CPU / CUDA | Generated answer `"Retrieval Augmented Generation"` | 260 ms | ✅ **PASS** |
| **DeBERTa NLI** | `cross-encoder/nli-deberta-v3-base` | CPU / CUDA | `entailment=0.9932`, `contradiction=0.0001`, `neutral=0.0067` | 115 ms | ✅ **PASS** |
| **T5 Question Gen** | `iarfmoose/t5-base-question-generator` | CPU / CUDA | Generated Q `"What is the best way to maintain state consistency across nodes?"` | 190 ms | ✅ **PASS** |

---

## 5. PostgreSQL & pgvector Persistence Verification

- **PostgreSQL Tables:** `documents`, `pages`, `chunks` verified.
- **pgvector Column:** 384-dimensional `vector` column indexed with HNSW (`m=16, ef_construction=64`).
- **Persistence Verification:** Stack restarts preserve all stored documents, extracted pages, generated chunks, and vector indexes.

---

## 6. Celery Background Task Processing

- **Task Name:** `process_document_task`
- **Queue:** Redis queue (`redis://redis:6379/0`)
- **Ingestion Test:** Uploading an 8-page PDF triggered background parsing, cleaning, chunking, MiniLM vector embedding, and pgvector persistence within **10.0 seconds**. Status updated automatically from `queued` $\rightarrow$ `processing` $\rightarrow$ `processed`.

---

## 7. Performance Metrics Summary

- **Isolated Direct Query GPU Mean:** **606.62 ms** (vs CPU **3072.93 ms** $\rightarrow$ **5.07× speedup**).
- **Full-Book 36-Question Stage-Sum Mean:** **3084.96 ms**.
- **Full-Book P95 Latency:** **5968.15 ms**.

---

## 8. Conclusion

The local production environment is fully operational and healthy. Proceed to `docs/MANUAL_MASTER_AUDIT_START.md` for human audit instructions.
