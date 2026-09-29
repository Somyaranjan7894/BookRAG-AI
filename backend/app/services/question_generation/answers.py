"""Answer candidate extraction from book evidence chunks for Phase 12 Question Generation.

Extracts meaningful, high-value answer candidates (entities, dates, years, numbers,
technical terms, key noun phrases) grounded directly within source text passages.
Preserves exact text spans, character offsets, and source chunk provenance.
"""

import re
from typing import List, Optional, Set
from app.core.logging import get_logger
from app.schemas.chunk import Chunk
from app.schemas.question_generation import AnswerCandidate

logger = get_logger(__name__)

STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now", "it", "its", "they", "them", "their",
    "we", "us", "our", "he", "him", "his", "she", "her", "hers", "i", "me",
    "my", "you", "your", "yours", "was", "were", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "would", "could"
}

# Regex patterns for high-value answer candidates
PATTERNS = [
    # Full dates: e.g. October 14, 1947 or 14 October 1947
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    # Standalone 4-digit years: e.g. 1991, 2020
    re.compile(r"\b(?:17|18|19|20)\d{2}\b"),
    # Quantities, numbers with units, percentages: e.g. 1.41 billion, 42%, 100 meters
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|percent|billion|million|trillion|thousand|hundred|meters|km|miles|kg|seconds|minutes|hours|days|years)\b",
        re.IGNORECASE,
    ),
    # Capitalized Multi-word Named Entities: e.g. Guido van Rossum, Alan Turing, United States
    re.compile(r"\b[A-Z][a-z]+(?:\s+(?:van|von|de|la|of|the|and)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+){1,3}\b"),
    # Capitalized Acronyms / Organizations: e.g. NASA, UNESCO, MIT, IBM, COVID-19
    re.compile(r"\b[A-Z]{2,}(?:-[0-9A-Z]+)?\b"),
    # Quoted terms: e.g. "backpropagation", 'quantum entanglement'
    re.compile(r"[\"']([A-Za-z0-9\s\-]{3,40})[\"']"),
    # Single Capitalized Proper Nouns (min 3 chars, not at the absolute start of sentences if generic):
    re.compile(r"(?<=[a-z0-9,\(\)]\s)\b[A-Z][a-z]{2,}\b"),
]


class AnswerCandidateExtractor:
    """Extracts grounded answer candidate spans from evidence passages."""

    def __init__(
        self,
        min_length: int = 2,
        max_length: int = 80,
        max_candidates_per_chunk: int = 6,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.max_candidates_per_chunk = max_candidates_per_chunk

    def extract_from_chunk(self, chunk: Chunk) -> List[AnswerCandidate]:
        """Extract candidate answers directly present in the chunk's text."""
        return self.extract_from_text(
            text=chunk.text,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
        )

    def extract_from_text(
        self,
        text: str,
        document_id: str,
        chunk_id: str,
        page_number: int,
        chunk_index: Optional[int] = None,
    ) -> List[AnswerCandidate]:
        """Extract candidate answers directly present in raw text with offsets."""
        candidates: List[AnswerCandidate] = []
        seen_spans: Set[str] = set()

        if not text or not text.strip():
            return candidates

        clean_text = text

        for pattern in PATTERNS:
            for match in pattern.finditer(clean_text):
                # If pattern has group 1 (e.g. quotes), use it
                span_text = match.group(1) if pattern.groups > 0 and match.group(1) else match.group(0)
                span_text = span_text.strip(" ,.;:'\"()[]")

                if not self._is_valid_candidate(span_text, clean_text):
                    continue

                normalized_key = span_text.lower()
                if normalized_key in seen_spans:
                    continue

                start_offset = clean_text.find(span_text)
                end_offset = start_offset + len(span_text) if start_offset >= 0 else None

                candidates.append(
                    AnswerCandidate(
                        answer_text=span_text,
                        source_text=clean_text,
                        start_offset=start_offset if start_offset >= 0 else None,
                        end_offset=end_offset,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        page_number=page_number,
                        chunk_index=chunk_index,
                    )
                )
                seen_spans.add(normalized_key)

                if len(candidates) >= self.max_candidates_per_chunk:
                    return candidates

        return candidates

    def _is_valid_candidate(self, span: str, full_text: str) -> bool:
        """Validate candidate against quality and stop-word rules."""
        if not span:
            return False
        if len(span) < self.min_length or len(span) > self.max_length:
            return False
        # Do not allow candidate to be the entire passage
        if span.strip() == full_text.strip():
            return False
        # Check against stop words
        words = span.lower().split()
        if len(words) == 1 and words[0] in STOP_WORDS:
            return False
        # Do not allow purely punctuation or whitespace
        if not any(c.isalnum() for c in span):
            return False
        return True
