"""Answer candidate extraction from book evidence chunks for Phase 22 Question Generation.

Extracts meaningful, high-value answer candidates across all supported question types:
- Direct facts & named entities (persons, locations, organizations, dates, years)
- Technical definitions (concepts, terms, formal formulations)
- Explanations & causes (mechanisms, rationale, why/how clauses)
- Comparisons & contrasts (distinctions between two models or methods)
- Numerical facts & mathematical constants (derivatives, percentages, dimensions)
- Multi-step reasoning & multi-page synthesis

Preserves exact text spans, character offsets, and complete chunk/page provenance.
"""

import re
from typing import List, Optional, Sequence, Set, Tuple
from app.core.logging import get_logger
from app.schemas.chunk import Chunk
from app.schemas.question_generation import AnswerCandidate, QuestionType

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
    "has", "had", "having", "do", "does", "did", "doing", "would", "could",
}


# Typed extraction pattern tuple: (Regex, TargetGroup, QuestionType)
TYPED_PATTERNS: List[Tuple[re.Pattern, int, QuestionType]] = [
    # 1. Full dates: e.g. October 14, 1947 or 14 October 1947
    (
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b",
            re.IGNORECASE,
        ),
        0,
        QuestionType.WHEN,
    ),
    (
        re.compile(
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            re.IGNORECASE,
        ),
        0,
        QuestionType.WHEN,
    ),
    # 2. Standalone 4-digit calendar years: e.g. 1991, 2020
    (
        re.compile(r"\b(?:17|18|19|20)\d{2}\b"),
        0,
        QuestionType.WHEN,
    ),
    # Keep a numeric range intact rather than extracting its upper bound as a standalone fact.
    (
        re.compile(r"\b\d+(?:\.\d+)?\s*[-\u2013]\s*\d+(?:\.\d+)?\s*%"),
        0,
        QuestionType.NUMERICAL,
    ),
    # Capture the explanatory predicate, not just the quoted term being defined.
    (
        re.compile(
            r"\b(?:our|the)\s+definition\s+of\s+([A-Za-z][A-Za-z0-9-]*(?:\s+[a-z][A-Za-z0-9-]*){0,3})\s+is\s+([^,.;\n]{10,160})(?=[,.;\n]|$)",
            re.IGNORECASE,
        ),
        2,
        QuestionType.DEFINITION,
    ),
    # 3. Numerical values, constants, percentages, units: e.g. "a constant 1", "42%", "1.41 billion"
    (
        re.compile(r"\ba\s+constant\s+\d+(?:\.\d+)?\b", re.IGNORECASE),
        0,
        QuestionType.NUMERICAL,
    ),
    (
        re.compile(
            r"(?<![\d-])\b\d+(?:\.\d+)?\s*(?:%|(?:\b(?:percent|billion|million|trillion|thousand|hundred|meters|km|miles|kg|seconds|minutes|hours|days|years)\b))",
            re.IGNORECASE,
        ),
        0,
        QuestionType.NUMERICAL,
    ),
    # 4. Mathematical relations: e.g. "derivative is a constant 1"
    (
        re.compile(
            r"(?:derivative\s+is\s+|derivative\s+equals\s+)(a\s+constant\s+\d+|\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        1,
        QuestionType.NUMERICAL,
    ),
    # 5. Definitions: e.g. "X is defined as Y" or "X refers to Y"
    (
        re.compile(
            r"\b([A-Z][A-Za-z0-9\s\-]{2,30})\s+(?:is defined as(?!\s+follows\b)|refers to|is a method for|is an algorithm that)\s+([^;\.\n]{10,160})(?=[;\.\n]|$)",
            re.IGNORECASE,
        ),
        2,
        QuestionType.DEFINITION,
    ),
    # 6. Explanations / Causes / Rationale: e.g. "because its derivative is a constant 1"
    (
        re.compile(
            r"\b(because|in order to|as a result of|by recursively applying)\s+([^;\.\n]{12,160})(?=[;\.\n]|$)",
            re.IGNORECASE,
        ),
        2,
        QuestionType.EXPLANATION,
    ),
    # 7. Comparisons: e.g. "In contrast to batch gradient descent, stochastic gradient descent..."
    (
        re.compile(
            r"(?:in contrast(?:\s+to)?|unlike|whereas|compared to)\s+([^,\.\n]{3,100}),\s+([^,\.\n]{3,100})(?=[,\.\n]|$)",
            re.IGNORECASE,
        ),
        2,
        QuestionType.COMPARISON,
    ),
    # 8. Subject-action / Direct fact mappings retain the subject and predicate.
    (
        re.compile(
            r"\b([A-Z][A-Za-z0-9]*(?:\s+[a-z][A-Za-z0-9-]*){0,3}\s+(?:maps?|optimizes?|computes?|calculates?|transforms?|connects?|minimizes?|maximizes?)\s+[^,;\.\n(]{10,160})(?=[,;\.\n(]|$)"
        ),
        1,
        QuestionType.DIRECT_FACT,
    ),
    # 9. Capitalized Multi-word Named Entities (Person, Place, Institution)
    (
        re.compile(r"\b[A-Z][a-z]+(?:\s+(?:van|von|de|la|of|the|and)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+){1,3}\b"),
        0,
        QuestionType.WHO,
    ),
    # 9. Capitalized Acronyms / Organizations: e.g. NASA, UNESCO, MIT, IBM, COVID-19, CNNs, ReLU, NLI
    (
        re.compile(r"\b[A-Z]{2,}(?:-[0-9A-Z]+|s)?\b"),
        0,
        QuestionType.FACTUAL,
    ),
    # 10. Quoted Technical Terms: e.g. "backpropagation", 'quantum entanglement'
    (
        re.compile(r"(?<!\w)([\"'])([A-Za-z0-9][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9\s\-]*){1,3})\1(?!\w)"),
        2,
        QuestionType.DEFINITION,
    ),
    # 11. Single Capitalized Proper Nouns (min 3 chars, not at start of sentence)
    (
        re.compile(r"(?<=[a-z0-9,\(\)]\s)\b[A-Z][a-z]{2,}\b"),
        0,
        QuestionType.FACTUAL,
    ),
]


GENERIC_FRAGMENT_WORDS = {
    "linear",
    "unit",
    "model",
    "function",
    "system",
    "layer",
    "method",
    "process",
    "value",
    "term",
    "component",
    "program",
    "programs",
    "computer",
    "computers",
    "algorithm",
    "algorithms",
    "data",
    "problem",
    "problems",
    "process",
    "processes",
}

_STRUCTURAL_UPPERCASE_TOKENS = {
    "CHAPTER", "CONTENTS", "INTRODUCTION", "APPENDIX", "FIGURE", "TABLE", "SECTION",
}


class AnswerCandidateExtractor:
    """Extracts evidence-grounded answer candidate spans across multiple question types."""

    def __init__(
        self,
        min_length: int = 2,
        max_length: int = 160,
        max_candidates_per_chunk: int = 8,
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
        """Extract candidate answers directly present in raw text with offsets and typed classifications."""
        candidates: List[AnswerCandidate] = []
        seen_spans: Set[str] = set()

        if not text or not text.strip():
            return candidates

        clean_text = text

        for pattern, group_idx, q_type in TYPED_PATTERNS:
            for match in pattern.finditer(clean_text):
                capture_idx = group_idx if group_idx <= pattern.groups else 0
                try:
                    raw_span = match.group(capture_idx)
                    span_start, span_end = match.span(capture_idx)
                except IndexError:
                    raw_span = match.group(0)
                    span_start, span_end = match.span(0)
                if raw_span is None:
                    continue

                span_text = raw_span.strip(" ,.;:'\"()[]")
                left_trim = len(raw_span) - len(raw_span.lstrip(" ,.;:'\"()[]"))
                right_trim = len(raw_span) - len(raw_span.rstrip(" ,.;:'\"()[]"))
                span_start += left_trim
                span_end -= right_trim

                if not self._is_valid_candidate(span_text, clean_text, span_start):
                    continue

                normalized_key = span_text.lower()
                if normalized_key in seen_spans:
                    continue

                start_offset = span_start
                end_offset = span_end
                candidate_metadata = {
                    "extraction_pattern": str(pattern.pattern[:30]),
                    "inferred_type": q_type.value,
                }
                if q_type == QuestionType.DEFINITION and "definition\\s+of" in pattern.pattern:
                    candidate_metadata["definition_term"] = match.group(1)
                if q_type == QuestionType.EXPLANATION:
                    candidate_metadata["extraction_trigger"] = match.group(1).lower()

                candidates.append(
                    AnswerCandidate(
                        answer_text=span_text,
                        source_text=clean_text,
                        start_offset=start_offset if start_offset >= 0 else None,
                        end_offset=end_offset,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        chunk_ids=[chunk_id],
                        page_number=page_number,
                        page_numbers=[page_number],
                        chunk_index=chunk_index,
                        candidate_type=q_type,
                        metadata=candidate_metadata,
                    )
                )
                seen_spans.add(normalized_key)

                if len(candidates) >= self.max_candidates_per_chunk:
                    return candidates

        return candidates

    def extract_multipage_candidates(self, chunks: Sequence[Chunk]) -> List[AnswerCandidate]:
        """Extract multi-page synthesis answer candidates across contiguous chunks from adjacent pages."""
        multipage_candidates: List[AnswerCandidate] = []
        if len(chunks) < 2:
            return multipage_candidates

        # Sort chunks by page_number, then chunk_index
        sorted_chunks = sorted(chunks, key=lambda c: (c.page_number, c.chunk_index or 0))

        for i in range(len(sorted_chunks) - 1):
            c1 = sorted_chunks[i]
            c2 = sorted_chunks[i + 1]

            # Check if chunks are from consecutive pages or within 2 pages
            if c1.page_number == c2.page_number or abs(c2.page_number - c1.page_number) > 2:
                continue

            combined_text = f"{c1.text.strip()}\n\n{c2.text.strip()}"

            # 1. Check for cross-chunk contrast / comparison markers
            contrast_match = re.search(
                r"\b(?:in contrast(?:\s+to)?|unlike|whereas|compared to|on the other hand)\s*,?\s*([A-Za-z0-9][A-Za-z0-9-]*(?:\s+[A-Za-z0-9][A-Za-z0-9-]*){1,3})\s+(?=(?:is|are|was|were|process|processes|use|uses|rely|relies|employ|employs|extract|extracts|update|updates)\b)",
                c2.text,
                re.IGNORECASE,
            )
            if contrast_match:
                term = contrast_match.group(1).strip(" ,.;:'\"")
                if self._is_valid_candidate(term, c2.text):
                    multipage_candidates.append(
                        AnswerCandidate(
                            answer_text=term,
                            source_text=combined_text,
                            start_offset=combined_text.find(term) if combined_text.find(term) >= 0 else None,
                            end_offset=None,
                            document_id=c1.document_id,
                            chunk_id=c1.chunk_id,
                            chunk_ids=[c1.chunk_id, c2.chunk_id],
                            page_number=c1.page_number,
                            page_numbers=[c1.page_number, c2.page_number],
                            chunk_index=c1.chunk_index,
                            candidate_type=QuestionType.MULTI_PAGE_SYNTHESIS,
                            metadata={
                                "is_multipage": True,
                                "contrast_term": term,
                                "source_pages": [c1.page_number, c2.page_number],
                            },
                        )
                    )

        return multipage_candidates

    def _is_valid_candidate(self, span: str, full_text: str, start_offset: Optional[int] = None) -> bool:
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
        # Reject isolated generic fragments that lack sufficient semantic context
        if len(words) == 1 and words[0] in GENERIC_FRAGMENT_WORDS:
            return False
        # Reject sub-word fragments extracted from hyphenated compounds (e.g. "Linear" from "Non-linear")
        if re.search(rf"\b[A-Za-z0-9]+-{re.escape(span)}\b", full_text, re.IGNORECASE) or re.search(
            rf"\b{re.escape(span)}-[A-Za-z0-9]+\b", full_text, re.IGNORECASE
        ):
            return False
        # Do not allow purely punctuation or whitespace
        if not any(c.isalnum() for c in span):
            return False
        # Reject candidates that end or start with dangling punctuation or conjunctions
        if words[0] in {
            "and", "or", "but", "so", "then", "of", "to", "in", "with", "if",
            "because", "while", "whereas", "unlike", "although", "when", "as",
        }:
            return False

        # Reject metadata noise (headers, author names, copyright lines, library codes)
        span_lower = span.lower()
        if span_lower.startswith("follows:"):
            return False
        if span in _STRUCTURAL_UPPERCASE_TOKENS:
            return False
        metadata_noise_words = {
            "mitchell", "tom mitchell", "tom m. mitchell", "mcgraw-hill", "mcgraw hill",
            "copyright", "rights reserved", "library of congress", "isbn", "dlc",
            "product details", "publisher", "edition", "printed in"
        }
        for noise in metadata_noise_words:
            if noise in span_lower:
                return False

        candidate_start = start_offset if start_offset is not None else full_text.find(span)
        candidate_end = candidate_start + len(span)
        if re.match(r"\s*:", full_text[candidate_end:]) and len(words) <= 3:
            return False
        if span.isupper():
            before = full_text[max(0, candidate_start - 28):candidate_start]
            after = full_text[candidate_end:candidate_end + 45]
            numbered_heading = re.search(r"\b\d+(?:\.\d+)*\s+$", before)
            upper_heading_continuation = re.match(r"(?:\s+[A-Z][A-Z0-9-]*){1,4}\s+(?=[A-Z])", after)
            if span.lower() not in {"nli", "cnn", "cnns", "relu", "lms", "svm", "svms", "id3", "alvinn", "sphinx", "ace"} and (
                numbered_heading is not None or upper_heading_continuation is not None
            ):
                return False
        if candidate_start >= 0 and candidate_end < len(full_text):
            if full_text[candidate_end - 1].isalnum() and full_text[candidate_end].isalnum():
                return False

        return True
