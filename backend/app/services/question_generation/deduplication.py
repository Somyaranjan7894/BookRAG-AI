"""Question deduplication mechanisms for Phase 12 Question Generation in BookRAG AI."""

import re
from typing import List, Optional, Set
from app.schemas.question_generation import GeneratedQuestion, QuestionCandidate


def normalize_question_for_dedup(question_text: str) -> str:
    """Normalize question text for stable duplicate comparison.

    Strips whitespace, converts to lowercase, collapses internal spaces,
    and removes trailing question marks or punctuation.
    """
    text = question_text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class QuestionDeduplicator:
    """Deduplicates question candidates within a single generation request."""

    def __init__(self) -> None:
        self._seen_normalized_questions: Set[str] = set()

    def is_duplicate(self, question_text: str) -> bool:
        """Check if a question is a duplicate of a previously seen question."""
        normalized = normalize_question_for_dedup(question_text)
        if not normalized:
            return True
        return normalized in self._seen_normalized_questions

    def record(self, question_text: str) -> None:
        """Record a question as seen."""
        normalized = normalize_question_for_dedup(question_text)
        if normalized:
            self._seen_normalized_questions.add(normalized)

    def filter_candidates(self, candidates: List[QuestionCandidate]) -> List[QuestionCandidate]:
        """Filter out duplicates from a list of QuestionCandidate objects."""
        unique: List[QuestionCandidate] = []
        for candidate in candidates:
            if not self.is_duplicate(candidate.question_text):
                self.record(candidate.question_text)
                unique.append(candidate)
        return unique

    def filter_questions(self, questions: List[GeneratedQuestion]) -> List[GeneratedQuestion]:
        """Filter out duplicates from a list of GeneratedQuestion objects."""
        unique: List[GeneratedQuestion] = []
        for q in questions:
            if not self.is_duplicate(q.question):
                self.record(q.question)
                unique.append(q)
        return unique
