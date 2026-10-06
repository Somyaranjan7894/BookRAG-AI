# BookRAG AI — Manual Master Audit Human Start Guide

Welcome to the **Manual Master Audit** for BookRAG AI (Phases 0–24.1).

This guide provides everything a human auditor needs to independently test and verify the entire BookRAG AI system running locally.

---

## 1. Application Access URLs

- **React Frontend Application:** [`http://localhost`](http://localhost) (or [`http://localhost:5173`](http://localhost:5173))
- **FastAPI Backend REST API:** [`http://localhost:8000`](http://localhost:8000)
- **FastAPI OpenAPI Interactive Swagger Documentation:** [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **System Health & Readiness Endpoint:** [`http://localhost:8000/api/v1/health/ready`](http://localhost:8000/api/v1/health/ready)

---

## 2. Prepared Test Datasets

The repository contains pre-configured test PDF documents in `data/`:

1. **`data/distributed_systems_handbook.pdf`**
   - *Topic:* Foundations of Distributed Systems, Cloud Architecture, and Machine Learning Infrastructure.
   - *Length:* 105 pages (412 chunks indexed).
   - *Pre-indexed ID:* `doc_dist_sys_handbook_01`
2. **`data/sample_ai_handbook.pdf`**
   - *Topic:* Foundations of Machine Learning & Neural Networks.
   - *Length:* 8 pages.

---

## 3. 20 Representative Test Questions for Audit

Use these representative questions to test system capabilities across query types:

| # | Question Type | Question Text | Expected Behavior |
|---|---|---|---|
| 1 | **Direct Fact** | *"What is the exact derivative of the Rectified Linear Unit (ReLU) activation function for positive inputs?"* | Returns answer span `1` with citation to Page 3 |
| 2 | **Definition** | *"What are the three fundamental semantic categories defined in Natural Language Inference (NLI)?"* | Returns `entailment`, `contradiction`, `neutral` |
| 3 | **Explanation** | *"Why are dot products divided by the square root of key dimension in Transformer attention?"* | Returns explanation regarding softmax gradient stability |
| 4 | **Comparison** | *"How do bi-encoders and cross-encoders compare in terms of latency and cross-attention?"* | Compares independent encoding vs joint attention |
| 5 | **Numerical Fact** | *"What is the vector embedding dimension used by all-MiniLM-L6-v2?"* | Returns `384` |
| 6 | **Reasoning** | *"How does Paxos guarantee consensus despite node failures?"* | Explains majority quorums and two-phase commits |
| 7 | **Multi-page Synthesis** | *"What techniques are described across pages 3 and 5 to address vanishing gradients?"* | Combines ReLU and scaled dot-product attention |
| 8 | **Unanswerable (Safe Refusal)** | *"What is the traditional culinary recipe, hydration ratio, and baking temperature for French sourdough bread?"* | **Safe Refusal** triggered; 0 citations returned |
| 9 | **Out-of-Domain (`fb_q30`)** | *"What are the quantum key distribution protocols and single-photon detector specifications for BB84 satellite quantum communications?"* | **Safe Refusal** or unanswerable status |
| 10 | **Ambiguous** | *"How should one configure the parameters?"* | Refuses or requests clarification |

---

## 4. Recommended Step-by-Step Human Auditor Workflow

Follow this 20-step workflow to verify the system from browser to database:

1. **Open Frontend:** Navigate to [`http://localhost`](http://localhost) in your browser.
2. **Inspect Hero Banner:** Confirm the visual design, title, and "Production RAG Architecture" badge.
3. **Upload PDF:** Click **"Upload New Book"**, select `data/sample_ai_handbook.pdf`, and submit.
4. **Observe Progress:** Watch the progress bar transition from `QUEUED` $\rightarrow$ `PROCESSING` $\rightarrow$ `PROCESSED`.
5. **Select Book:** Click on the newly indexed book to open its detail page.
6. **Execute Search:** Type `"activation function"` into the search box.
7. **Inspect Matching Board:** Click **"Relevant Matching Board"** to review Stage 1 vector candidates and Stage 2 CrossEncoder reranker scores (`↑ Promoted`).
8. **Ask Factual Question:** Submit: *"What is the derivative of ReLU for positive inputs?"*
9. **Inspect Answer & Grounding:** Confirm the answer string and DeBERTa NLI badge (`FULLY SUPPORTED`).
10. **Inspect Citation:** Click the citation pill (`[Page 3]`) to expand the source chunk text.
11. **Ask Multi-Page Question:** Submit: *"What techniques address vanishing gradients?"*
12. **Verify Multi-Page Citations:** Confirm multi-page citations (`Pages 3, 5`).
13. **Ask Unanswerable Question:** Submit: *"What is the recipe for artisanal sourdough bread?"*
14. **Verify Safe Refusal:** Confirm the safe refusal response and zero citation leakage.
15. **Generate Questions:** Navigate to the **"Question Generator"** tab, set count to `3`, and click **"Generate Questions"**.
16. **Inspect Question Cards:** Review generated questions, extractive QA validated answers, and page provenance.
17. **Query Generated Question:** Click **"Query Book"** on a generated question to execute it through the RAG pipeline.
18. **Test Document Isolation:** Upload a second PDF (`data/smoke_test.pdf`), select it, and ask a question. Verify 0 citations leak from the first PDF.
19. **Inspect Health API:** Open [`http://localhost:8000/api/v1/health/ready`](http://localhost:8000/api/v1/health/ready) in a browser tab to confirm PostgreSQL and Redis status.
20. **Record Result:** Record `PASS` in `docs/MANUAL_MASTER_AUDIT_CHECKLIST.md`.

---

## 5. Helpful Local CLI Commands for the Auditor

- **View All Running Services:**
  ```bash
  docker compose ps
  ```
- **Stream Backend Logs:**
  ```bash
  docker compose logs -f backend
  ```
- **Stream Celery Worker Logs:**
  ```bash
  docker compose logs -f worker
  ```
- **Run Full Automated Regression Suite:**
  ```bash
  pytest backend/tests
  ```
