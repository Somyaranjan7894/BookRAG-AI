"""Question validation and answer matching engine for Phase 12 Question Generation in BookRAG AI.

Ensures that every generated question:
1. Passes deterministic structural and linguistic quality checks.
2. Is strictly grounded in the source book evidence passage.
3. Is independently answerable by the extractive QA model (RoBERTa SQuAD2).
4. Matches the expected answer using semantics-preserving normalization and numeric verification.
"""

import re
from typing import List, Optional, Tuple
from app.core.logging import get_logger
from app.schemas.qa import QAResponse
from app.schemas.question_generation import (
    AnswerCandidate,
    GeneratedQuestion,
    QuestionCandidate,
    QuestionDifficulty,
    QuestionType,
)
from app.services.qa.service import QAService

logger = get_logger(__name__)


def normalize_answer_text(text: str) -> str:
    """Normalize text for answer comparison while preserving numbers and letters."""
    clean = text.strip().lower()
    clean = re.sub(r"[^\w\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def extract_numbers(text: str) -> List[str]:
    """Extract all numeric tokens from text."""
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def matches_expected_answer(predicted: str, expected: str) -> bool:
    """Determine whether the QA model's predicted answer matches the expected answer.

    Rules:
    1. Empty predictions never match.
    2. Numbers must match strictly (e.g. 1998 != 1999).
    3. Exact normalized equality passes immediately.
    4. Controlled partial overlap/substring match passes if token overlap is high.
    """
    if not predicted or not expected:
        return False

    norm_pred = normalize_answer_text(predicted)
    norm_exp = normalize_answer_text(expected)

    if not norm_pred or not norm_exp:
        return False

    # Exact normalized match
    if norm_pred == norm_exp:
        return True

    # Numeric consistency: numbers in expected must match numbers in predicted
    exp_nums = extract_numbers(expected)
    pred_nums = extract_numbers(predicted)

    if exp_nums or pred_nums:
        # If numbers are present, all expected numbers must be in predicted numbers
        if set(exp_nums) != set(pred_nums):
            return False

    # Substring / containment check
    if norm_exp in norm_pred or norm_pred in norm_exp:
        return True

    # Token overlap check (Jaccard similarity >= 0.5)
    pred_tokens = set(norm_pred.split())
    exp_tokens = set(norm_exp.split())
    intersection = pred_tokens.intersection(exp_tokens)
    union = pred_tokens.union(exp_tokens)
    if union and (len(intersection) / len(union)) >= 0.5:
        return True

    return False


def classify_question_type(question_text: str) -> QuestionType:
    """Classify question type based on grammatical structure."""
    q_lower = question_text.strip().lower()

    if q_lower.startswith("who"):
        return QuestionType.WHO
    elif q_lower.startswith("where"):
        return QuestionType.WHERE
    elif q_lower.startswith("when") or "what year" in q_lower or "what date" in q_lower:
        return QuestionType.WHEN
    elif q_lower.startswith("why") or "what reason" in q_lower:
        return QuestionType.WHY
    elif q_lower.startswith("how many") or q_lower.startswith("how much"):
        return QuestionType.HOW_MANY
    elif q_lower.startswith("how"):
        return QuestionType.HOW
    elif "compare" in q_lower or "difference between" in q_lower or "versus" in q_lower:
        return QuestionType.COMPARISON
    elif "definition of" in q_lower or "what is meant by" in q_lower or "define" in q_lower:
        return QuestionType.DEFINITION
    elif q_lower.startswith("what"):
        return QuestionType.WHAT
    return QuestionType.FACTUAL


def classify_question_difficulty(
    question_type: QuestionType,
    answer_text: str,
    source_text: str,
) -> QuestionDifficulty:
    """Assign heuristic difficulty classification to a question."""
    answer_words = answer_text.strip().split()
    word_count = len(answer_words)

    if question_type in (QuestionType.WHO, QuestionType.WHEN, QuestionType.WHERE, QuestionType.HOW_MANY):
        if word_count <= 4:
            return QuestionDifficulty.EASY
        return QuestionDifficulty.MEDIUM

    if question_type in (QuestionType.WHY, QuestionType.HOW, QuestionType.COMPARISON):
        return QuestionDifficulty.HARD

    if word_count > 6 or len(source_text) > 800:
        return QuestionDifficulty.MEDIUM

    return QuestionDifficulty.EASY


class QuestionValidator:
    """Validates question candidates against source evidence and extractive QA."""

    def __init__(self, qa_service: Optional[QAService] = None) -> None:
        self._qa_service = qa_service

    @property
    def qa_service(self) -> QAService:
        if self._qa_service is None:
            self._qa_service = QAService()
        return self._qa_service

    def validate_candidate(
        self,
        candidate: QuestionCandidate,
    ) -> Tuple[bool, Optional[GeneratedQuestion], Optional[str]]:
        """Validate a single QuestionCandidate against quality checks and extractive QA.

        Returns:
            Tuple of (is_valid, validated_question_or_none, rejection_reason_or_none)
        """
        q_text = candidate.question_text.strip()
        ans_cand = candidate.answer_candidate
        source_text = ans_cand.source_text.strip()
        expected_ans = ans_cand.answer_text.strip()

        # 1. Basic Quality Checks
        quality_err = self._check_quality(q_text, expected_ans, source_text)
        if quality_err:
            return False, None, quality_err

        # 2. Source Grounding Check
        if expected_ans.lower() not in source_text.lower():
            return False, None, "Expected answer does not occur in source evidence."

        # 3. Extractive QA Verification
        try:
            # Pass source chunk as single evidence item
            qa_evidence_item = {
                "chunk_id": ans_cand.chunk_id,
                "document_id": ans_cand.document_id,
                "page_number": ans_cand.page_number,
                "chunk_index": ans_cand.chunk_index or 0,
                "text": source_text,
                "similarity_score": 1.0,
                "reranker_score": 1.0,
                "rank": 1,
            }
            qa_res: QAResponse = self.qa_service.answer_question(
                query=q_text,
                evidence=[qa_evidence_item],
            )
        except Exception as exc:
            logger.warning("QA validation failed with exception for question '%s': %s", q_text, exc)
            return False, None, f"QA execution failed: {exc}"

        if not qa_res.answerable or not qa_res.answer:
            return False, None, "QA model determined the question is unanswerable from the evidence."

        predicted_ans = qa_res.answer.strip()
        if not matches_expected_answer(predicted=predicted_ans, expected=expected_ans):
            return (
                False,
                None,
                f"QA predicted answer '{predicted_ans}' does not match expected answer '{expected_ans}'.",
            )

        # 4. Construct Validated Question
        q_type = classify_question_type(q_text)
        difficulty = classify_question_difficulty(q_type, expected_ans, source_text)

        validated = GeneratedQuestion(
            question=q_text,
            answer=expected_ans,
            question_type=q_type,
            difficulty=difficulty,
            document_id=ans_cand.document_id,
            chunk_id=ans_cand.chunk_id,
            page_number=ans_cand.page_number,
            source_text=source_text,
            start_offset=ans_cand.start_offset,
            end_offset=ans_cand.end_offset,
            qa_predicted_answer=predicted_ans,
            qa_confidence_score=qa_res.qa_score,
            metadata={
                "qa_score": qa_res.qa_score,
                "no_answer_score": qa_res.no_answer_score,
            },
        )
        return True, validated, None

    def _check_quality(self, question: str, answer: str, source_text: str) -> Optional[str]:
        """Perform deterministic quality checks on question candidate."""
        if not question:
            return "Question text is empty."
        if len(question) < 8:
            return f"Question is too short ({len(question)} characters)."
        if len(question) > 250:
            return f"Question is excessively long ({len(question)} characters)."
        if not question.endswith("?"):
            return "Question does not end with a question mark."

        # Cannot be identical to source text
        if question.lower() == source_text.lower():
            return "Question is identical to source text."

        # Circular question: question literally is just the answer or contains answer in trivial circular way
        norm_q = normalize_answer_text(question)
        norm_ans = normalize_answer_text(answer)
        if norm_q == norm_ans:
            return "Question is identical to the target answer."

        # Avoid questions where the entire question is "Who is {answer}?" or "What is {answer}?"
        if norm_q in {f"who is {norm_ans}", f"what is {norm_ans}", f"where is {norm_ans}"}:
            # Allowed only if question type is definition, but trivial tautologies should be flagged
            pass

        return None
