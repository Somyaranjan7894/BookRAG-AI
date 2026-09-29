"""Cross-Encoder Reranker Service for BookRAG AI Phase 6.

Coordinates (query, passage) cross-encoder scoring, rank reassignment,
provenance preservation, and candidate pool reranking.
"""

import math
from typing import Any, List, Optional, Sequence

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.search import SearchResult
from app.services.reranking.exceptions import (
    InvalidRerankingConfigError,
    RerankingExecutionError,
    ScoreAlignmentError,
)
from app.services.reranking.model import RerankerModel

logger = get_logger(__name__)


class RerankerService:
    """Orchestrates second-stage cross-encoder precision reranking for candidate chunks."""

    def __init__(
        self,
        model: Optional[RerankerModel] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the RerankerService with an injectable CrossEncoder model.

        Args:
            model: Optional pre-initialized RerankerModel instance (useful for testing/fakes).
            batch_size: Inference batch size for scoring pairs.
            max_length: Maximum sequence length for the cross-encoder transformer.
            model_name: Model identifier (defaults to configuration).
            device: Device target ('auto', 'cpu', 'cuda').
        """
        self.batch_size = batch_size or settings.RERANKER_BATCH_SIZE
        self.max_length = max_length or settings.RERANKER_MAX_LENGTH
        self.model = model or RerankerModel.get_instance(
            model_name=model_name,
            device=device,
            max_length=self.max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank candidate chunks using joint query-passage transformer scoring.

        Execution Pipeline:
        1. Validate query string and candidates.
        2. Construct (query, text) sentence pairs.
        3. Execute batched cross-encoder inference under torch.inference_mode.
        4. Validate strict 1-to-1 score-to-candidate alignment.
        5. Sort candidates in descending order of reranker_score.
        6. Reassign new final ranks while preserving original_rank and chunk provenance.
        7. Return top_n highest-scoring candidates.

        Args:
            query: Natural language search query string.
            candidates: Sequence of first-stage candidate objects (RetrievalResult or SearchResult).
            top_n: Optional maximum number of final reranked candidates to return.

        Returns:
            List of SearchResult objects re-ranked and populated with reranker_score.

        Raises:
            InvalidRerankingConfigError: If query is empty or top_n is non-positive.
            RerankingExecutionError: If cross-encoder inference fails or candidate is malformed.
            ScoreAlignmentError: If returned scores count does not strictly match candidate count.
        """
        if query is None or not isinstance(query, str) or not query.strip():
            raise InvalidRerankingConfigError("Query for reranking cannot be empty or whitespace-only.")

        clean_query = query.strip()

        if not candidates:
            return []

        if top_n is not None and top_n <= 0:
            raise InvalidRerankingConfigError(f"top_n must be strictly positive (got {top_n}).")

        # 1. Build pairs and validate candidate integrity
        pairs: List[List[str]] = []
        for idx, c in enumerate(candidates):
            text = getattr(c, "text", None)
            if text is None or not isinstance(text, str):
                raise RerankingExecutionError(
                    f"Candidate at position {idx} has invalid or missing 'text' attribute."
                )
            pairs.append([clean_query, text])

        # 2. Run batched model inference
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        except Exception as exc:
            logger.exception("CrossEncoder prediction forward pass failed: %s", exc)
            raise RerankingExecutionError(
                f"Cross-encoder scoring failed: {exc}",
                details=str(exc),
            ) from exc

        # 3. Strict 1-to-1 score alignment validation
        if len(scores) != len(candidates):
            raise ScoreAlignmentError(
                f"Score count ({len(scores)}) does not match candidate count ({len(candidates)})."
            )

        # 4. Construct preliminary result records with original FAISS rank
        scored_candidates: List[SearchResult] = []
        for idx, (candidate, score) in enumerate(zip(candidates, scores)):
            float_score = float(score)
            if math.isnan(float_score) or math.isinf(float_score):
                raise RerankingExecutionError(
                    f"Invalid non-finite score {float_score} produced for candidate '{getattr(candidate, 'chunk_id', idx)}'."
                )

            # Preserve original first-stage rank
            orig_rank = getattr(candidate, "rank", idx + 1)
            sim_score = float(getattr(candidate, "similarity_score", 0.0))

            scored_candidates.append(
                SearchResult(
                    rank=idx + 1,  # Temporary slot, updated below after sort
                    original_rank=orig_rank,
                    chunk_id=getattr(candidate, "chunk_id", f"chunk_{idx}"),
                    document_id=getattr(candidate, "document_id", "unknown_doc"),
                    page_number=getattr(candidate, "page_number", 1),
                    text=getattr(candidate, "text", ""),
                    similarity_score=sim_score,
                    reranker_score=float_score,
                    chunk_index=getattr(candidate, "chunk_index", 0),
                    metadata=getattr(candidate, "metadata", {}),
                )
            )

        # 5. Sort candidates strictly in descending order of reranker_score
        # Tie-breaker: original similarity_score (descending)
        scored_candidates.sort(
            key=lambda item: (item.reranker_score if item.reranker_score is not None else -float("inf"), item.similarity_score),
            reverse=True,
        )

        # 6. Reassign final rank 1..N
        for new_rank, item in enumerate(scored_candidates, start=1):
            item.rank = new_rank

        # 7. Apply top_n cutoff if specified
        final_results = scored_candidates[:top_n] if top_n is not None else scored_candidates

        logger.info(
            "Reranked %d candidates for query '%s' -> returning top %d.",
            len(candidates),
            clean_query[:40],
            len(final_results),
        )
        return final_results
