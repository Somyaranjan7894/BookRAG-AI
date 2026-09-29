"""Deterministic claim decomposition for BookRAG AI Phase 9.

Decomposes generated answers into discrete sentence-level claim units for individual
NLI groundedness verification against retrieved book evidence.

ARCHITECTURAL LIMITATIONS:
Sentence splitting produces sentence-level propositions as a deterministic heuristic.
It does NOT produce true semantic atomic claims (e.g., compound sentences containing
multiple independent factual clauses are evaluated as a single unit). This modular
design allows a more advanced syntactic or dependency-based atomic claim decomposer
to be introduced in future iterations without changing the downstream NLI pipeline.
"""

import re
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.grounding import ExtractedClaim

logger = get_logger(__name__)

# Heuristic sentence boundary split: punctuation followed by space or line break
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


class ClaimDecomposer:
    """Deterministic sentence-level claim decomposer."""

    def __init__(self, min_claim_length: Optional[int] = None) -> None:
        """Initialize the ClaimDecomposer.

        Args:
            min_claim_length: Minimum character length for a fragment to be treated as a claim.
                              Defaults to settings.GROUNDING_MIN_CLAIM_LENGTH.
        """
        self.min_claim_length = (
            min_claim_length
            if min_claim_length is not None
            else settings.GROUNDING_MIN_CLAIM_LENGTH
        )

    def decompose(self, text: Optional[str]) -> List[ExtractedClaim]:
        """Decompose text into normalized, indexed sentence claims.

        Args:
            text: Input string (e.g., generated answer from FLAN-T5).

        Returns:
            List of ExtractedClaim objects with deterministic 0-based indices.
        """
        if text is None or not isinstance(text, str):
            return []

        cleaned = text.strip()
        if not cleaned:
            return []

        # Split on sentence boundaries and newlines
        raw_fragments = SENTENCE_SPLIT_REGEX.split(cleaned)

        claims: List[ExtractedClaim] = []
        current_index = 0

        for fragment in raw_fragments:
            # Normalize whitespace: collapse internal tabs/spaces/newlines into single space
            norm = " ".join(fragment.strip().split())

            # Discard empty fragments and fragments shorter than minimum length
            if not norm or len(norm) < self.min_claim_length:
                continue

            claims.append(
                ExtractedClaim(
                    claim_index=current_index,
                    claim_text=norm,
                )
            )
            current_index += 1

        logger.debug(
            "Decomposed text into %d claim(s) (min_length=%d).",
            len(claims),
            self.min_claim_length,
        )
        return claims
