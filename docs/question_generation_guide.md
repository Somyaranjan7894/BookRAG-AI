# Phase 22: Controlled Question Generation & QA Expansion Guide

BookRAG AI provides a production-grade, evidence-grounded question generation and reading comprehension subsystem. It automatically extracts factual concept spans from indexed book passages, conditions question generation on local sequence-to-sequence models, and enforces strict extractive QA answerability checks so that no hallucinated or unanswerable questions reach the user.

---

## 1. Primary Architecture

The Question Generation pipeline follows an evidence-first architecture:

```
Book Evidence Passage (Chunk / Page Provenance)
       ↓
Answer Candidate Extraction (Typed regex & grammatical heuristics)
       ↓
Question Generation (iarfmoose/t5-base-question-generator)
       ↓
Quality Validation (Linguistic, length, circularity, vague checks)
       ↓
Answerability Validation (deepset/roberta-base-squad2 Extractive QA)
       ↓
Duplicate Detection (Exact, Normalized, Token Jaccard near-duplicates)
       ↓
Complete Provenance Preservation (Document, Pages, Chunks, Offsets)
       ↓
Validated Question Pair
```

---

## 2. Core Concepts: Generated Candidate vs. Verified Question

| Stage | Definition | Guarantee |
| :--- | :--- | :--- |
| **Generated Candidate** | Raw natural-language question synthesized by the T5 model conditioned on an extracted evidence span. | Grammatically fluent question text; **not yet verified**. May contain slight answer mismatches, external knowledge assumptions, or duplicate phrasing. |
| **Verified Question** | A generated question that has passed deterministic quality checks, contains zero outside entities, and whose target answer is independently extracted by RoBERTa with high confidence. | **Grounded in book evidence**. Verifiably answerable from the indicated book page and chunk. |

---

## 3. Supported Question Types

The taxonomy supports both high-level analytical categories and granular grammatical types:

1. **Direct Fact (`direct_fact`)**: Targets specific named entities, components, dates, or mechanisms directly stated in text.
2. **Definition (`definition`)**: Targets technical definitions ("What is defined as X?", "What is meant by Y?").
3. **Explanation (`explanation`)**: Targets causal mechanisms and rationale ("Why does ReLU prevent vanishing gradients?").
4. **Comparison (`comparison`)**: Targets distinctions and comparative analyses between two or more concepts across chunks.
5. **Numerical Fact (`numerical_fact`)**: Targets mathematical constants, statistics, rates, and percentages ("What is the derivative of ReLU for positive inputs?").
6. **Reasoning (`reasoning`)**: Targets causal consequences and impacts ("What is the consequence of high learning rates?").
7. **Multi-Page Synthesis (`multi_page_synthesis`)**: Targets relationships, contrasts, or shared concepts spanning consecutive pages.

---

## 4. Multi-Stage Validation Pipeline

Every candidate undergoes three validation filters before acceptance:

### A. Deterministic Quality Checks
- **Length & Bounds**: Rejects questions $< 8$ or $> 250$ characters.
- **Sentence Form**: Must begin with an uppercase letter, end with `?`, and cannot start with invalid punctuation (`?`, `!`, `,`, `.`).
- **Incomplete / Dangling Structure**: Detects questions ending with dangling prepositions or conjunctions (e.g. `role of?`, `because?`, `with?`).
- **Vague & Generic Questions**: Suppresses non-informative queries (e.g. `What is this?`, `What happened?`, `What did they do?`).
- **Circularity & Tautology**: Rejects questions identical to the target answer or source passage, as well as trivial tautological restatements (`Who is {answer}?`).

### B. Outside Knowledge & Grounding Checks
- **Entity Containment**: Capitalized proper nouns introduced in the question must physically appear in the source passage. Questions cannot introduce outside entities (e.g., asking about Einstein when the passage only mentions support vector machines).
- **Target Answer Grounding**: The candidate answer must occur as an exact substring in the source passage.

### C. Answerability Validation (Extractive QA)
- The candidate question is submitted to the RoBERTa extractive QA model (`deepset/roberta-base-squad2`) with the source chunk as evidence.
- The model must determine `answerable == True` with positive confidence.
- The QA model's predicted span must contain every expected answer content token in the same order after normalization. Function-word variation is tolerated, numeric tokens must match exactly (e.g., $1998 \neq 1999$), and a one-word answer cannot match only as a fragment of a much longer predicted span.

---

## 5. Duplicate Detection & Suppression

To prevent redundant questions, `QuestionDeduplicator` enforces three tiers of duplicate checks:
1. **Exact Duplicates**: Character-for-character identical questions.
2. **Normalized Duplicates**: Normalizes casing, punctuation, and whitespace.
3. **Semantically Near-Duplicate Questions**: Computes token-level Jaccard similarity. If $\text{Jaccard} \ge 0.75$, the candidate is suppressed and recorded with its similarity score in diagnostics.

---

## 6. Provenance Contracts

Every returned `GeneratedQuestion` retains durable provenance:
- `document_id`: Source book identifier.
- `page_number`: Primary page where the answer appears.
- `page_numbers`: List of all pages contributing to the question (crucial for multi-page synthesis).
- `chunk_id`: Primary chunk ID.
- `chunk_ids`: All contributing chunk IDs.
- `source_text`: The exact passage containing the evidence.
- `start_offset` / `end_offset`: Exact character offsets of the target answer span.

---

## 7. API Reference

### Generate Questions
`POST /api/v1/questions/generate`

#### Request Payload
```json
{
  "document_id": "doc_ai_handbook_01",
  "count": 5,
  "question_type": "definition",
  "difficulty": "medium",
  "include_rejected": true,
  "ensure_diversity": true
}
```

#### Response Payload
```json
{
  "document_id": "doc_ai_handbook_01",
  "requested_count": 5,
  "generated_candidates": 18,
  "validated_count": 5,
  "returned_count": 5,
  "questions": [
    {
      "question_id": "qgen_doc_ai_handbook_01_c0004_1241",
      "question": "What is the definition of CNN?",
      "answer": "Convolutional Neural Networks",
      "question_type": "definition",
      "difficulty": "easy",
      "document_id": "doc_ai_handbook_01",
      "chunk_id": "doc_ai_handbook_01_p004_c0004",
      "chunk_ids": ["doc_ai_handbook_01_p004_c0004"],
      "page_number": 4,
      "page_numbers": [4],
      "source_text": "Convolutional Neural Networks (CNNs) exploit the local spatial geometry...",
      "qa_predicted_answer": "Convolutional Neural Networks",
      "qa_confidence_score": 13.7459,
      "validation_status": "validated"
    }
  ],
  "rejected_candidates": [
    {
      "candidate_id": "cand_001",
      "question_text": "What is the ReLU activation function?",
      "answer_text": "The Rectified Linear Unit",
      "rejection_reason": "near_duplicate_similarity_0.75",
      "rejection_category": "duplicate",
      "chunk_id": "doc_ai_handbook_01_p003_c0003",
      "chunk_ids": ["doc_ai_handbook_01_p003_c0003"],
      "page_number": 3,
      "page_numbers": [3],
      "question_type": "what"
    }
  ],
  "rejection_summary": {
    "duplicate": 4,
    "answer_mismatch": 2,
    "unanswerable": 2,
    "quality_failure": 1
  },
  "latency_ms": 27436.91
}
```

---

## 8. UI Usage

The Book Detail page (`/books/:id`) features a dedicated **"Generate Questions"** tab:
1. Select target question count (3, 5, 8, 10).
2. Optionally filter by Question Type (`Definition`, `Comparison`, `Explanation`, etc.) or Difficulty (`Easy`, `Medium`, `Hard`).
3. Toggle `Ensure Type Diversity` and `Show Diagnostics`.
4. Click **"Generate Questions"** to trigger the generation and validation pipeline.
5. Inspect the generated questions with color-coded type and difficulty badges, page citations, verified answer spans, and expandable source evidence passages.
6. Click **"Query Book"** on any generated question to immediately switch to Grounded QA mode and synthesize a full NLI-grounded answer.
