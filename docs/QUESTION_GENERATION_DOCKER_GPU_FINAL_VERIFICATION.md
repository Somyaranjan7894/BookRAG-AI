# Question Generation: Docker, PostgreSQL, and GPU Final Verification

**Verification date:** 2026-10-05  
**Final status:** **CONDITIONALLY VERIFIED**  
**Scope:** Docker recovery, PostgreSQL/pgvector and Redis/Celery checks, host CUDA and model execution, real-PDF QG, and available regression suites. Phase 25 and the Manual Master Audit were not started.

## 1. Executive Summary

Docker Desktop is operational on the `desktop-linux` context. All five Compose services are running; PostgreSQL and Redis are healthy. PostgreSQL/pgvector schema and embeddings were verified directly, a real asynchronous Celery job completed and persisted vectors, and the API health endpoint responded successfully.

The repository's `backend/.venv` is the intended CUDA-enabled environment: PyTorch `2.14.0+cu126`, CUDA runtime 12.6, BF16 supported, and `DEVICE=auto` resolves to the RTX 3050. All six current production wrappers loaded with their actual model parameters on `cuda:0`. The Docker backend/worker are CPU-only and are not evidence of host GPU inference.

The complete backend suite passed in the project venv. QG/architecture, Phase 23, Phase 24, Phase 20 evaluation, and frontend tests/build passed. The live Phase 20 golden benchmark initially exposed that an `AMBIGUOUS` query plan could still proceed to generation; the orchestrator now refuses those queries before retrieval/generation, and the rerun refused all 3/3 unanswerable/ambiguous items. The real 421-page technical book run executed the complete QG/QA path on CUDA, but returned only **1 accepted question from 30 candidates** for a request of 10. That question was manually inspected and supported by its exact source chunk. The low acceptance yield limits broader manual review and question-type coverage; this report does not treat it as a successful quality-yield target.

## 2. Docker Engine Verification

| Check | Result |
|---|---|
| `docker version` client | Docker 29.8.1, Windows/amd64 |
| `docker version` server | Docker Desktop 4.93.0, Engine 29.8.1, Linux/amd64 |
| Active context | `desktop-linux` |
| `docker info` | Successful; Docker Desktop Linux engine responds |
| `docker ps` | Successful; BookRAG containers visible |
| `docker compose up -d` | Successful; existing services reconciled without rebuild |

## 3. Docker Compose Services

Compose file: `docker-compose.yml`.

| Compose service | Container | Port mapping | Status |
|---|---|---:|---|
| `db` | `bookrag_postgres` | host 5433 → container 5432 | Healthy |
| `redis` | `bookrag_redis` | 6379 → 6379 | Healthy |
| `backend` | `bookrag_backend` | 8000 → 8000 | Running |
| `worker` | `bookrag_worker` | internal only | Running; no Compose healthcheck is configured |
| `frontend` | `bookrag_frontend` | 80 → 80 | Running |

PostgreSQL uses database `bookrag`, user `postgres`, image `pgvector/pgvector:pg16`, and `pg_isready -U postgres -d bookrag` healthcheck. The host mapping matches the tests' expected `localhost:5433`. Redis listens on 6379 and uses `redis-cli ping` for its healthcheck. A Compose warning notes the `version` attribute is obsolete; it does not prevent service operation.

## 4. PostgreSQL Verification

- Host `Test-NetConnection localhost -Port 5433`: `TcpTestSucceeded=True`.
- Container `pg_isready -U postgres -d bookrag_test`: accepting connections.
- PostgreSQL: 16.15.
- Databases present: `bookrag`, `bookrag_test`, and `postgres`.
- Application tables: `documents`, `pages`, `chunks`, and `alembic_version`.
- Current Alembic revision: `6e82c18d9f12`.
- Stored chunks at inspection: 1,208; embedded chunks: 1,208.

The full backend suite, including database-dependent tests, passed against the live stack.

## 5. pgvector Verification

- Extension `vector`: installed, version 0.8.6.
- `chunks.embedding`: `vector(384)`, matching `all-MiniLM-L6-v2`.
- All inspected embeddings have 384 dimensions.
- HNSW index `ix_chunks_embedding_hnsw`: present using `vector_cosine_ops`, `m=16`, `ef_construction=64`.
- No schema changes were made for verification.

## 6. Redis Verification

- Host `Test-NetConnection localhost -Port 6379`: `TcpTestSucceeded=True`.
- Container `redis-cli ping`: `PONG`.
- Compose health: healthy.

## 7. Celery Verification

- Celery app reports broker `redis://localhost:6379/0`, result backend `redis://localhost:6379/0`, and `task_always_eager=False` in the host venv.
- Compose worker logs report connection to `redis://redis:6379/0` and list `app.workers.tasks.process_document_task`.
- `celery inspect ping`: worker returned `pong`; one node online.
- Real task ID `252f60c5-3556-4d54-8348-761107989579` was received and processed by `bookrag_worker` (not an eager/mock task). The 2-page PDF `3132f0420a6e_live_docker_test.pdf` produced 2 chunks and 2 pgvector embeddings in 7.39 seconds.
- Durable row state: `processed`, stage `completed`, page count 2, no error; database query confirmed 2 chunks and 2 embeddings.
- Verification record remains in the local `bookrag` database as `doc_verify_celery_26dbaab3be36`.
- After rebuilding the worker with the verified source, task `47dc6ce4-f2ae-41a0-a27c-7a0d4c047f18` was submitted with eager mode false and completed through Redis; document `doc_verify_rebuilt_worker_a4e37ea633` reached `processed/completed` with 2 pages and 2 chunks.

## 8. Backend Verification

- `GET http://localhost:8000/api/v1/health`: `{"status":"ok","service":"BookRAG AI"}`.
- Backend startup logs show application startup complete with PostgreSQL/pgvector configuration.
- Backend and worker images were rebuilt after the final safety fix. The API health endpoint passed from both the Windows host and inside the container after startup; the rebuilt Celery worker passed inspect ping and completed a real ingestion job.
- Cached `QuestionGenerationService` is not given a request-scoped session by its singleton provider. Its document-chunk retrieval opens a temporary session when needed and closes that session in the same operation. The dependency-injection singleton regression test passes.
- Docker backend inference is CPU-only, as its startup log reports `cuda_available=False`; host-native model verification is reported separately below.

## 9. CUDA Verification

Host GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB; NVIDIA driver 566.07 (driver-reported CUDA 12.7).

The system Python installation is CPU-only (`2.14.1+cpu`), but it is not the project's selected environment. The project interpreter `backend/.venv/Scripts/python.exe` reports:

- Python 3.13.7.
- PyTorch `2.14.0+cu126`; `torch.version.cuda == 12.6`.
- `torch.cuda.is_available() == True`; one device visible.
- Device name: `NVIDIA GeForce RTX 3050 Laptop GPU`.
- `torch.cuda.is_bf16_supported() == True`.
- `DEVICE=auto`; DeviceManager resolves to `cuda`.

This verifies the host implementation without altering the existing CUDA/BF16 configuration. Docker containers were not passed through to the GPU and remain CPU inference environments.

## 10. RTX 3050 Details

`nvidia-smi` reported 4,096 MiB total memory, driver 566.07, 0 MiB used before the model audit, and no processes at that point. PyTorch reported 4,095.5 MiB total. The real-book QG run peaked at 2,365.14 MiB allocated. The run's synchronized wall latency included the full extraction, generation, QA, and validation path; it was not used as a model-only GPU benchmark.

## 11. Six-Model Execution Matrix

Each wrapper was instantiated with CUDA explicitly and its actual first model parameter was verified on `cuda:0`. Model weights are FP32; the shared inference context selects BF16 autocast where supported. The live audit executed inference through all wrappers; the QA wrapper also ran repeatedly as part of full QG validation.

| Model | Actual configured checkpoint | Runtime/output evidence | Result |
|---|---|---|---|
| MiniLM embedder | `sentence-transformers/all-MiniLM-L6-v2` | CUDA parameter; real embedding dimension 384 | Pass |
| CrossEncoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CUDA parameter; real score `-10.2500` for tested query/passage pair | Pass |
| RoBERTa SQuAD2 | `deepset/roberta-base-squad2` | CUDA parameter; extracted span `1997` during book QG QA validation | Pass |
| FLAN-T5 | `google/flan-t5-base` | CUDA parameter; generated `Retrieval Augmented Generation`; BF16 path retained | Pass |
| DeBERTa NLI | `cross-encoder/nli-deberta-v3-base` | CUDA parameter; real output probabilities: entailment 0.9932, contradiction 0.0001, neutral 0.0067 | Pass |
| T5 Question Generation | `iarfmoose/t5-base-question-generator` | CUDA parameter; generated `What is the best way to maintain state consistency across nodes?` | Pass |

The pre-existing GPU artifact describes a different NLI-small checkpoint and is not used as current evidence. The actual current wrapper checkpoint is the `...deberta-v3-base` model listed above.

## 12. Question Generation End-to-End Flow

Executed the actual `QuestionGenerationService.generate_questions()` path in the project CUDA venv, using PostgreSQL-persisted chunks generated by PDF ingestion of the 421-page *Machine Learning* book (document `doc_MachineLearningTomMi_2e81d5558ff4fca1`). No answers or questions were mocked or hardcoded into the service run.

Observed stages: durable PDF-derived page/chunk retrieval; eligibility filtering; answer-candidate extraction; real T5 batch generation; question typing/difficulty classification; deduplication; real RoBERTa extractive QA; expected-answer matching; structural, ambiguity, and outside-knowledge checks; final provenance construction. Generated candidates and accepted output contain source document/page/chunk metadata.

Measured final run: 1,087 chunks; 964 eligible; 123 ineligible; 30 raw question candidates; 5 duplicate rejections; 11 validation failures; 8 unanswerable; 5 answer mismatches; 1 accepted. GPU-synchronized wall time was approximately 15.0 seconds; peak allocated VRAM was 2,365.14 MiB.

The actual filtering summary included 2 `METADATA`, 3 `PREFACE`, 2 `NAME_LIST`, 76 `REFERENCES`, 7 `LOW_TEXT_QUALITY`, and 33 `INDEX` chunks excluded. The run preserved legitimate technical content. No chunks classified as `ABSTRACT` occurred in this document, so abstract-only exclusion was not independently exercised by this book run; it remains covered by unit tests.

## 13. QG Quality Results

The final accepted item was: **“What year did Cooper et al. predict recovery rates of pneumonia patients?”** Answer: **1997**. Provenance: page 14, chunk `doc_MachineLearningTomMi_2e81d5558ff4fca1_p014_c0010`. That source chunk was independently retrieved from PostgreSQL, classified `TECHNICAL_CONTENT`, and contains the statement that the book describes programs that “predict recovery rates of pneumonia patients (Cooper et al. 1997).” The QA model predicted the same answer, `1997`.

The low yield (1 accepted from 30 candidates, versus 10 requested) is recorded as a limitation. It prevents inspection of 10 accepted questions and does not establish useful coverage across all requested question types or multi-page synthesis. Rejected examples included generic/context-dependent questions such as “How do computers learn?” and metadata-oriented questions about the book's authors. Weak answer fragments and incomplete answers were rejected after the validator hardening described below.

## 14. Manual Question Inspection

Only one item was accepted, so the minimum-10 inspection requirement could not be met.

| Example | Inspection |
|---|---|
| GOOD: “What year did Cooper et al. predict recovery rates of pneumonia patients?” → `1997` | Complete, specific, technically relevant, independently extracted by QA, exact source evidence on page 14, and correct chunk provenance. Useful as a factual question, though its answer is a citation year rather than a core concept. |
| BAD accepted | None observed in the final single accepted item. |
| REJECTED: “How do computers learn?” | Generic/context-dependent and rejected by quality validation. |
| REJECTED: `learning` definition candidate with a long explanatory QA span | One-word answer fragment was rejected after adding the fragment guard. |

## 15. GPU Performance

Same QG checkpoint, input answer/context, generation parameters, and output were used for both devices. Each device received 2 warmups and 5 measured iterations; CUDA runs synchronized before and after each timed iteration.

| Device | Mean | P95 (max of 5 samples) | Output |
|---|---:|---:|---|
| CPU | 649.57 ms | 695.42 ms | `What is the best way to maintain state consistency across nodes?` |
| RTX 3050 CUDA | 660.79 ms | 723.19 ms | Same output |

Measured mean ratio: GPU/CPU is about 1.02x (GPU was approximately 1.7% slower in this small workload). CUDA use is verified by parameter placement and synchronized CUDA execution, but this single-input beam-search workload showed no speedup. No performance claim is made that GPU accelerates this QG micro-workload.

## 16. Full Regression Results

| Suite | Result |
|---|---|
| Full backend, project venv, live DB | **522 passed**, 0 failures/errors/skips, 35 warnings, 71.52 s |
| Previously blocked Celery/persistence/pgvector modules | **45 passed**, 0 failures/errors/skips, 17.24 s |
| QG + architecture + production-quality | **79 passed**, 0 failures, 7 warnings, 18.94 s |
| Phase 20 live golden benchmark after ambiguity fix | **10 queries**; answerable success 100%; claim support 100%; refusal precision **100% (3/3)**; MRR 1.0; citation precision 100%; citation recall 92.86%; mean correctness F1 0.4257; mean generation/NLI latency 1,501.83 ms |
| Ambiguous refusal regression | **1 passed**; retrieval, generation, and NLI not invoked |
| Phase 23 evaluation | **17 passed**, 1 warning, 0.11 s |
| Phase 24 hardening | **23 passed**, 1 warning, 1.69 s |
| Available evaluation tests | **22 passed**, 1 warning, 0.08 s |
| Phase 21 grounding robustness | Included in combined explicit phase run; combined evaluation/Phase 21/23/24 result: **74 passed**, 1 warning, 2.10 s |
| Frontend tests | **36 passed** across 6 files, 27.92 s |
| Frontend TypeScript/Vite production build | **Passed**, 0 TypeScript/build errors, 7.95 s |

Backend warnings are Starlette/httpx deprecation warnings. No suite was excluded from the full backend run. Phase 20 coverage is `tests/test_evaluation.py` (22 passed), Phase 23 includes a Phase 20 golden-dataset regression, and `evaluation.runner.run_full_evaluation()` was executed against the live 10-query golden benchmark. Its successful output is under `scratch/gpu_verify/reports_phase20_live_20261005/`.

## 17. Mock/Dummy Audit

A scoped search of active backend code under `backend/app/{api,core,services,workers}` found **zero production mock/dummy implementations**. Matches were legitimate features or terms: sampling controls, configured fallbacks, prompt placeholders, a testing/fake injection seam in a service constructor, test substitution, or documentation references. The test suites do use mocks/fakes where expected. This is a scoped active-code audit, not a claim that every occurrence of these words in tests, docs, or scratch files is production behavior.

## 18. Issues Found

1. The actual project environment is `backend/.venv`; the system Python is CPU-only. Using system Python for CUDA checks would report a false negative. Tests were rerun with the project venv.
2. The full-book run exposed incomplete-answer acceptance through loose token overlap and acceptance of lowercase-start questions. A one-word expected answer could match a long QA span as a fragment.
3. The stored GPU verification artifact refers to an NLI-small variant, while the current application wrapper uses the base checkpoint. Current verification used the wrapper's actual base checkpoint.
4. Real-book QG yield was low (1/10 requested), leaving insufficient accepted examples to complete the requested ten-item manual review.
5. The initial Phase 20 run found a safety failure: `gold_q10` was misclassified as answerable by the orchestrator despite the query planner marking it ambiguous.
6. The first attempt to write Phase 20 reports to a system temporary directory raised a path-formatting exception after evaluation; rerunning with an output directory under the repository completed successfully. The evaluation runner's arbitrary external `output_dir` reporting remains limited.
7. `docker-compose.yml` emits a non-blocking warning for its obsolete top-level `version` field.

## 19. Fixes Applied

- Restored the cached `QuestionGenerationService` dependency provider, which had been changed to instantiate a service around a request-scoped SQLAlchemy session. The service's existing chunk retrieval opens/closes its own temporary session, so the singleton does not retain request session state.
- Tightened expected-answer matching to require all expected content tokens in order, tolerate only function-word variation, keep number matching strict, and reject a one-token answer matched merely as part of a long predicted span.
- Require generated questions to start with an uppercase letter.
- Added focused regression tests for truncated answers, function-word variation, and lowercase starts.
- Honor `QueryType.AMBIGUOUS` in the grounded-answer orchestrator with a safe refusal before retrieval or generation; the live golden benchmark now refuses the ambiguous universal-best query.
- Updated `docs/question_generation_guide.md` to match the implemented validation contract.

No model, database schema, Compose port, or CUDA configuration was changed.

## 20. Remaining Limitations

- The real-book request asked for 10, but the strict validator accepted only 1. Broader manual review and question-type coverage need a larger-yield corpus/configuration or further evidence-grounded QG quality work; relaxing the answer checks is not justified by this run.
- The evaluation runner's report-path display assumes output directories are inside the repository; the successful live verification used a repository-local output directory.
- QG GPU microbenchmark showed near parity with CPU; GPU execution is confirmed, GPU speedup for this workload is not.
- CUDA is available to the host Python process, not to the current Docker backend/worker containers.
- The Celery worker has no Compose healthcheck, although inspect ping and a real task succeeded.

## 21. Final Status

**CONDITIONALLY VERIFIED.** Docker, PostgreSQL, pgvector, Redis, rebuilt Celery/backend services, API health, Phase 20's live golden benchmark, the full backend/frontend regressions, six host-GPU model wrappers, and real-book CUDA QG execution were verified. The Phase 20 ambiguous-query safety failure was fixed and its live rerun passed the refusal target. QG is not blocked: it produced a valid, grounded question and rejects weak candidates. Conditional status reflects the low real-book accepted-question yield (1 of 10 requested, limiting quality-sample size) and the lack of GPU speedup for the measured small QG workload. No Phase 25 work or Manual Master Audit was started.
