"""Question deduplication mechanisms for Phase 22 Controlled Question Generation in BookRAG AI.

Detects:
1. Exact duplicates (character-for-character)
2. Normalized duplicates (case, punctuation, whitespace normalized)
3. Semantically near-duplicate questions (token Jaccard similarity >= threshold)

Records rejected duplicate candidates with specific diagnostic reasons to prevent
redundant question generation while maintaining complete diagnostic transparency.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
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


QUESTION_STOP_WORDS: Set[str] = {
    "is", "the", "are", "was", "were", "a", "an", "of", "in", "to",
    "for", "with", "by", "on", "at", "from", "does", "do", "did", "can",
    "could", "would", "should", "definition", "meaning", "concept", "type",
    "types", "it", "its", "they", "them", "their", "this", "that", "these",
    "those", "be", "been", "being",
}


def get_token_set(text: str) -> Set[str]:
    """Extract normalized content-word token set for Jaccard similarity."""
    norm = normalize_question_for_dedup(text)
    if not norm:
        return set()
    tokens = set(norm.split())
    content_tokens = {t for t in tokens if t not in QUESTION_STOP_WORDS}
    return content_tokens if content_tokens else tokens


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Compute content-word token Jaccard similarity between two question strings."""
    tokens1 = get_token_set(text1)
    tokens2 = get_token_set(text2)
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    if not union:
        return 0.0
    return len(intersection) / len(union)


class QuestionDeduplicator:
    """Deduplicates question candidates within a single generation request."""

    def __init__(
        self,
        near_duplicate_threshold: float = 0.88,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        self.near_duplicate_threshold = (
            similarity_threshold if similarity_threshold is not None else near_duplicate_threshold
        )
        self._seen_exact_questions: Set[str] = set()
        self._seen_normalized_questions: Set[str] = set()
        self._seen_raw_questions: List[str] = []
        self._seen_raw_answers: List[Optional[str]] = []
        self._answers_by_normalized_question: Dict[str, Set[str]] = defaultdict(set)
        self.exact_duplicate_count = 0
        self.near_duplicate_count = 0

    def find_duplicate_reason(
        self,
        question_text: str,
        answer_text: Optional[str] = None,
    ) -> Optional[str]:
        """Determine if question is a duplicate and return the exact diagnostic reason."""
        clean = question_text.strip()
        if not clean:
            return "empty_question"

        normalized = normalize_question_for_dedup(question_text)
        if not normalized:
            return "empty_normalized_question"

        normalized_answer = normalize_question_for_dedup(answer_text) if answer_text else None
        same_question_answer_seen = (
            normalized_answer is None
            or normalized_answer in self._answers_by_normalized_question.get(normalized, set())
        )

        if clean in self._seen_exact_questions and same_question_answer_seen:
            return "exact_duplicate"

        if normalized in self._seen_normalized_questions and same_question_answer_seen:
            return "normalized_duplicate"

        # Near-duplicate check via Jaccard token similarity
        for previous_q, previous_answer in zip(self._seen_raw_questions, self._seen_raw_answers):
            if (
                normalized_answer is not None
                and previous_answer is not None
                and normalized_answer != previous_answer
            ):
                continue
            sim = compute_jaccard_similarity(question_text, previous_q)
            if sim >= self.near_duplicate_threshold:
                return f"near_duplicate_similarity_{round(sim, 2)}"

        return None

    def is_duplicate(self, question_text: str) -> bool:
        """Check if a question is an exact, normalized, or near-duplicate of a previously seen question."""
        return self.find_duplicate_reason(question_text) is not None

    def record(self, question_text: str, answer_text: Optional[str] = None) -> None:
        """Record a question as seen."""
        clean = question_text.strip()
        normalized = normalize_question_for_dedup(question_text)
        normalized_answer = normalize_question_for_dedup(answer_text) if answer_text else None
        if clean:
            self._seen_exact_questions.add(clean)
        if normalized:
            self._seen_normalized_questions.add(normalized)
            self._seen_raw_questions.append(question_text)
            self._seen_raw_answers.append(normalized_answer)
            if normalized_answer is not None:
                self._answers_by_normalized_question[normalized].add(normalized_answer)

    def filter_candidates(self, candidates: List[QuestionCandidate]) -> List[QuestionCandidate]:
        """Filter out duplicates from a list of QuestionCandidate objects."""
        unique, _ = self.filter_candidates_with_rejected(candidates)
        return unique

    def filter_candidates_with_rejected(
        self,
        candidates: List[QuestionCandidate],
    ) -> Tuple[List[QuestionCandidate], List[Tuple[QuestionCandidate, str]]]:
        """Filter candidates returning both accepted unique items and rejected duplicates with reasons."""
        unique: List[QuestionCandidate] = []
        rejected: List[Tuple[QuestionCandidate, str]] = []

        for candidate in candidates:
            answer_text = candidate.answer_candidate.answer_text
            reason = self.find_duplicate_reason(candidate.question_text, answer_text)
            if reason is None:
                self.record(candidate.question_text, answer_text)
                unique.append(candidate)
            else:
                if "exact" in reason:
                    self.exact_duplicate_count += 1
                else:
                    self.near_duplicate_count += 1
                rejected.append((candidate, reason))

        return unique, rejected

    def filter_questions(self, questions: List[GeneratedQuestion]) -> List[GeneratedQuestion]:
        """Filter out duplicates from a list of GeneratedQuestion objects."""
        unique: List[GeneratedQuestion] = []
        for q in questions:
            if not self.is_duplicate(q.question):
                self.record(q.question)
                unique.append(q)
        return unique
