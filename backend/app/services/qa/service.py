"""Extractive Question Answering Service for BookRAG AI Phase 7.

Orchestrates question-context tokenization with sliding-window overflow handling,
context token mask validation, candidate start/end span generation, SQuAD 2.0
no-answer thresholding, and complete provenance preservation.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.qa import QAResponse
from app.schemas.search import SearchResult
from app.services.qa.exceptions import (
    InvalidAnswerSpanError,
    InvalidQAConfigError,
    InvalidQAEvidenceError,
    InvalidQAQueryError,
    QAError,
    QAInferenceError,
)

from app.services.qa.model import QAModel

logger = get_logger(__name__)


class QAService:
    """Orchestrates extractive question answering over retrieved and reranked evidence chunks."""

    def __init__(
        self,
        model: Optional[QAModel] = None,
        max_length: Optional[int] = None,
        doc_stride: Optional[int] = None,
        max_answer_length: Optional[int] = None,
        n_best_size: Optional[int] = None,
        no_answer_threshold: Optional[float] = None,
    ) -> None:
        """Initialize QAService with configurable or injected model and parameters.

        Args:
            model: Optional QAModel instance (defaults to cached singleton).
            max_length: Maximum sequence length for question + context window (default 512).
            doc_stride: Sliding-window stride when chunk context overflows max_length (default 128).
            max_answer_length: Maximum token length of an extracted answer span (default 64).
            n_best_size: Number of top candidate start/end logits to explore per window (default 20).
            no_answer_threshold: SQuAD 2.0 score difference threshold for answerability (default 0.0).
        """
        self._model = model
        self.max_length = max_length if max_length is not None else settings.QA_MAX_LENGTH
        self.doc_stride = doc_stride if doc_stride is not None else settings.QA_DOC_STRIDE
        self.max_answer_length = (
            max_answer_length if max_answer_length is not None else settings.QA_MAX_ANSWER_LENGTH
        )
        self.n_best_size = n_best_size if n_best_size is not None else settings.QA_N_BEST_SIZE
        self.no_answer_threshold = (
            no_answer_threshold
            if no_answer_threshold is not None
            else settings.QA_NO_ANSWER_THRESHOLD
        )


        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Verify that sliding window and answer length parameters are mathematically sound."""
        if self.max_length <= 0:
            raise InvalidQAConfigError(f"max_length must be positive (got {self.max_length}).")
        if self.doc_stride <= 0:
            raise InvalidQAConfigError(f"doc_stride must be positive (got {self.doc_stride}).")
        if self.doc_stride >= self.max_length:
            raise InvalidQAConfigError(
                f"doc_stride ({self.doc_stride}) must be strictly less than max_length ({self.max_length})."
            )
        if self.max_answer_length <= 0:
            raise InvalidQAConfigError(
                f"max_answer_length must be positive (got {self.max_answer_length})."
            )

    @property
    def model(self) -> QAModel:
        """Lazy load or return the injected QAModel instance."""
        if self._model is None:
            self._model = QAModel.get_instance()
        return self._model

    def answer_question(
        self,
        query: str,
        evidence: Sequence[Any],
        no_answer_threshold: Optional[float] = None,
    ) -> QAResponse:
        """Extract the best supported answer span across candidate evidence chunks.

        Sliding-Window and SQuAD 2.0 Evaluation Workflow:
        1. Validate query string and sanitize input.
        2. Handle empty evidence gracefully by returning an unanswerable structured response.
        3. For each candidate chunk:
           a. Tokenize (query, chunk_text) using truncation='only_second' and sliding stride.
           b. Generate input_ids, attention_mask, offset_mapping, and sequence_ids.
           c. Forward inference pass to obtain start_logits and end_logits.
           d. For each sliding window:
              - Extract the null score from token 0 (start_logits[0] + end_logits[0]).
              - Identify context tokens (sequence_ids == 1).
              - Search top n_best_size start/end token combinations.
              - Filter: start <= end, length <= max_answer_length, valid context offsets.
              - Compute span_score = start_logits[start] + end_logits[end].
              - Compute score_diff = span_score - null_score.
              - Map token boundaries back to character offsets in source text.
        4. Aggregate candidates across all windows and all evidence chunks.
        5. Evaluate best candidate against no-answer threshold.
        6. Return QAResponse with complete source provenance.

        Args:
            query: Natural language question string.
            evidence: Sequence of SearchResult or dict objects representing retrieved chunks.
            no_answer_threshold: Optional override for the answerability threshold.

        Returns:
            QAResponse containing extracted answer span or structured no-answer indicators.
        """
        # 1. Query validation
        if query is None or not isinstance(query, str) or not query.strip():
            raise InvalidQAQueryError("Question query cannot be empty or whitespace-only.")

        clean_query = query.strip()
        effective_threshold = (
            no_answer_threshold
            if no_answer_threshold is not None
            else self.no_answer_threshold
        )

        # 2. Empty evidence handling
        if not evidence:
            logger.info("No evidence chunks provided for query '%s'. Returning unanswerable.", clean_query)
            return QAResponse(
                query=clean_query,
                answer=None,
                answerable=False,
                qa_score=None,
                no_answer_score=None,
                document_id=None,
                chunk_id=None,
                page_number=None,
                chunk_index=None,
                answer_start=None,
                answer_end=None,
                source_text=None,
                evidence_rank=None,
                similarity_score=None,
                reranker_score=None,
                total_evidence_evaluated=0,
            )

        tokenizer = self.model.tokenizer
        all_candidates: List[Dict[str, Any]] = []
        best_null_score: Optional[float] = None

        # 3. Process each evidence chunk
        for rank_idx, chunk in enumerate(evidence, start=1):
            chunk_text, chunk_id, doc_id, page_num, chunk_idx, sim_score, rerank_score = self._extract_chunk_attributes(chunk)

            if not chunk_text or not chunk_text.strip():
                continue

            try:
                # Tokenize with sliding window for long context
                encoded = tokenizer(
                    clean_query,
                    chunk_text,
                    max_length=self.max_length,
                    stride=self.doc_stride,
                    truncation="only_second",
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="longest",
                    return_tensors="pt",
                )
            except Exception as exc:
                logger.exception("Tokenization failed for chunk '%s': %s", chunk_id, exc)
                raise QAInferenceError(
                    f"Tokenizer failed to process evidence chunk '{chunk_id}'.",
                    details=str(exc),
                ) from exc

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            num_windows = input_ids.shape[0]

            # Forward pass
            try:
                start_logits, end_logits = self.model.predict_logits(input_ids, attention_mask)
            except QAError:
                raise
            except Exception as exc:
                logger.exception("Inference failed for chunk '%s': %s", chunk_id, exc)
                raise QAInferenceError(
                    f"Model inference failed for evidence chunk '{chunk_id}'.",
                    details=str(exc),
                ) from exc


            # Evaluate each sliding window
            for w in range(num_windows):
                window_start_logits = start_logits[w]
                window_end_logits = end_logits[w]
                window_offsets = encoded["offset_mapping"][w].tolist()
                window_seq_ids = encoded.sequence_ids(w)

                # Token 0 is the null/no-answer token for SQuAD 2.0 (<s> in RoBERTa)
                null_score = float(window_start_logits[0].item() + window_end_logits[0].item())
                if best_null_score is None or null_score > best_null_score:
                    best_null_score = null_score

                # Identify context token indices (sequence_id == 1)
                context_indices = [
                    idx for idx, seq_id in enumerate(window_seq_ids)
                    if seq_id == 1
                ]

                if not context_indices:
                    continue

                # Filter top n_best_size logits strictly within context tokens
                context_start_logits = [(idx, float(window_start_logits[idx].item())) for idx in context_indices]
                context_end_logits = [(idx, float(window_end_logits[idx].item())) for idx in context_indices]

                context_start_logits.sort(key=lambda x: x[1], reverse=True)
                context_end_logits.sort(key=lambda x: x[1], reverse=True)

                top_starts = [idx for idx, _ in context_start_logits[:self.n_best_size]]
                top_ends = [idx for idx, _ in context_end_logits[:self.n_best_size]]

                for s_idx in top_starts:
                    for e_idx in top_ends:
                        if s_idx > e_idx:
                            continue

                        token_span_len = e_idx - s_idx + 1
                        if token_span_len > self.max_answer_length:
                            continue

                        # Extract character offsets
                        char_start, _ = window_offsets[s_idx]
                        _, char_end = window_offsets[e_idx]

                        if char_start >= char_end or char_start < 0 or char_end > len(chunk_text):
                            continue

                        span_text = chunk_text[char_start:char_end]
                        cleaned_span = span_text.strip()
                        if not cleaned_span:
                            continue

                        # Adjust character offsets to match trimmed text
                        leading_ws = len(span_text) - len(span_text.lstrip())
                        trailing_ws = len(span_text) - len(span_text.rstrip())
                        final_char_start = char_start + leading_ws
                        final_char_end = char_end - trailing_ws

                        span_score = float(window_start_logits[s_idx].item() + window_end_logits[e_idx].item())
                        score_diff = span_score - null_score

                        all_candidates.append({
                            "answer": cleaned_span,
                            "qa_score": span_score,
                            "no_answer_score": null_score,
                            "score_diff": score_diff,
                            "document_id": doc_id,
                            "chunk_id": chunk_id,
                            "page_number": page_num,
                            "chunk_index": chunk_idx,
                            "answer_start": final_char_start,
                            "answer_end": final_char_end,
                            "source_text": chunk_text,
                            "evidence_rank": rank_idx,
                            "similarity_score": sim_score,
                            "reranker_score": rerank_score,
                        })

        # 4. Determine best candidate and answerability
        if not all_candidates:
            logger.info("No valid answer spans extracted from evidence for query '%s'.", clean_query)
            return QAResponse(
                query=clean_query,
                answer=None,
                answerable=False,
                qa_score=None,
                no_answer_score=round(best_null_score, 4) if best_null_score is not None else None,
                document_id=None,
                chunk_id=None,
                page_number=None,
                chunk_index=None,
                answer_start=None,
                answer_end=None,
                source_text=None,
                evidence_rank=None,
                similarity_score=None,
                reranker_score=None,
                total_evidence_evaluated=len(evidence),
            )

        # Sort candidate spans by score_diff descending (preference over null_score)
        all_candidates.sort(key=lambda c: (c["score_diff"], c["qa_score"]), reverse=True)
        best = all_candidates[0]

        # 5. Check against no-answer threshold
        is_answerable = best["score_diff"] > effective_threshold

        if not is_answerable:
            logger.info(
                "Best span '%s' score_diff (%.4f) fell below threshold (%.4f). Returning unanswerable.",
                best["answer"][:30],
                best["score_diff"],
                effective_threshold,
            )
            return QAResponse(
                query=clean_query,
                answer=None,
                answerable=False,
                qa_score=round(best["qa_score"], 4),
                no_answer_score=round(best["no_answer_score"], 4),
                document_id=best["document_id"],
                chunk_id=best["chunk_id"],
                page_number=best["page_number"],
                chunk_index=best["chunk_index"],
                answer_start=None,
                answer_end=None,
                source_text=best["source_text"],
                evidence_rank=best["evidence_rank"],
                similarity_score=best["similarity_score"],
                reranker_score=best["reranker_score"],
                total_evidence_evaluated=len(evidence),
            )

        logger.info(
            "Extracted answer '%s' (qa_score=%.4f, score_diff=%.4f) from chunk '%s', page %s.",
            best["answer"][:40],
            best["qa_score"],
            best["score_diff"],
            best["chunk_id"],
            best["page_number"],
        )

        return QAResponse(
            query=clean_query,
            answer=best["answer"],
            answerable=True,
            qa_score=round(best["qa_score"], 4),
            no_answer_score=round(best["no_answer_score"], 4),
            document_id=best["document_id"],
            chunk_id=best["chunk_id"],
            page_number=best["page_number"],
            chunk_index=best["chunk_index"],
            answer_start=best["answer_start"],
            answer_end=best["answer_end"],
            source_text=best["source_text"],
            evidence_rank=best["evidence_rank"],
            similarity_score=best["similarity_score"],
            reranker_score=best["reranker_score"],
            total_evidence_evaluated=len(evidence),
        )

    @staticmethod
    def _extract_chunk_attributes(chunk: Any) -> Tuple[str, str, str, int, int, Optional[float], Optional[float]]:
        """Extract standardized chunk fields from SearchResult, dict, or object."""
        if isinstance(chunk, dict):
            text = str(chunk.get("text", ""))
            chunk_id = str(chunk.get("chunk_id", ""))
            doc_id = str(chunk.get("document_id", ""))
            page_num = int(chunk.get("page_number", 1))
            chunk_idx = int(chunk.get("chunk_index", 0))
            sim_score = chunk.get("similarity_score")
            rerank_score = chunk.get("reranker_score")
        else:
            text = getattr(chunk, "text", "")
            chunk_id = getattr(chunk, "chunk_id", "")
            doc_id = getattr(chunk, "document_id", "")
            page_num = getattr(chunk, "page_number", 1)
            chunk_idx = getattr(chunk, "chunk_index", 0)
            sim_score = getattr(chunk, "similarity_score", None)
            rerank_score = getattr(chunk, "reranker_score", None)

        return (
            str(text or ""),
            str(chunk_id or ""),
            str(doc_id or ""),
            int(page_num or 1),
            int(chunk_idx or 0),
            float(sim_score) if sim_score is not None else None,
            float(rerank_score) if rerank_score is not None else None,
        )
