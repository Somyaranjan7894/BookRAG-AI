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

_ANSWER_FUNCTION_WORDS = frozenset(
    "a an the and or but if as because while when where why how all any both each few more most other some such "
    "no nor not only own same so than too very can will just should now it its they them their we us our he him "
    "his she her hers i me my you your yours was were been being have has had having do does did doing would could "
    "of to in out on off from for with at by is are".split()
)


def normalize_answer_text(text: str) -> str:
    """Normalize text for answer comparison while preserving numbers and letters."""
    clean = text.strip().lower()
    clean = re.sub(r"[^\w\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def extract_numbers(text: str) -> List[str]:
    """Extract all numeric tokens from text."""
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def _is_grounded_definition_expansion(
    predicted: str,
    expected: str,
    question: Optional[str],
    source_text: Optional[str],
) -> bool:
    """Allow a longer extractive span only for a term explicitly defined in that same source sentence."""
    if not question or not source_text:
        return False
    if not re.search(r"\b(?:definition|define|meaning)\b", question, re.IGNORECASE):
        return False

    expected_content = [
        token
        for token in normalize_answer_text(expected).split()
        if token not in _ANSWER_FUNCTION_WORDS
    ]
    if len(expected_content) != 1:
        return False

    normalized_prediction = normalize_answer_text(predicted)
    if expected_content[0] not in normalized_prediction.split():
        return False

    for sentence in re.split(r"(?<=[.!?])\s+", source_text):
        normalized_sentence = normalize_answer_text(sentence)
        if normalized_prediction not in normalized_sentence:
            continue
        term = re.escape(expected_content[0])
        if re.search(rf"\bdefinition\s+of\s+{term}\s+is\b", normalized_sentence):
            return True
        if re.search(rf"\b{term}\s+(?:is\s+defined\s+as|refers\s+to)\b", normalized_sentence):
            return True
    return False


def _is_grounded_subordinate_clause_match(
    predicted: str,
    expected: str,
    question: Optional[str],
) -> bool:
    """Allow concise answer clause when expected was an overly broad sentence containing question premise."""
    if not predicted or not expected or not question:
        return False
    exp_nums = extract_numbers(expected)
    pred_nums = extract_numbers(predicted)
    if exp_nums or pred_nums:
        if set(exp_nums) != set(pred_nums):
            return False

    norm_pred = normalize_answer_text(predicted)
    norm_exp = normalize_answer_text(expected)
    pred_tokens = [t for t in norm_pred.split() if t not in _ANSWER_FUNCTION_WORDS]
    exp_tokens = [t for t in norm_exp.split() if t not in _ANSWER_FUNCTION_WORDS]
    if len(pred_tokens) < 3 or len(exp_tokens) <= len(pred_tokens):
        return False

    is_contiguous = any(
        exp_tokens[i:i + len(pred_tokens)] == pred_tokens
        for i in range(len(exp_tokens) - len(pred_tokens) + 1)
    )
    if not is_contiguous:
        return False

    norm_q = set(normalize_answer_text(question).split())
    remaining_tokens = set(exp_tokens) - set(pred_tokens)
    overlap_with_q = len(remaining_tokens & norm_q) / max(1, len(remaining_tokens))
    return overlap_with_q >= 0.5


def matches_expected_answer(
    predicted: str,
    expected: str,
    *,
    question: Optional[str] = None,
    source_text: Optional[str] = None,
) -> bool:
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

    if _is_grounded_definition_expansion(predicted, expected, question, source_text):
        return True

    if _is_grounded_subordinate_clause_match(predicted, expected, question):
        return True

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

    expected_content = [
        token for token in norm_exp.split() if token not in _ANSWER_FUNCTION_WORDS
    ]
    predicted_content = [
        token for token in norm_pred.split() if token not in _ANSWER_FUNCTION_WORDS
    ]
    if (
        len(expected_content) == 1
        and len(predicted_content) >= 3
        and expected_content[0] in predicted_content
        and not (exp_nums and pred_nums and set(exp_nums) == set(pred_nums))
    ):
        return False

    if len(expected_content) > len(predicted_content):
        return False
    return any(
        predicted_content[index:index + len(expected_content)] == expected_content
        for index in range(len(predicted_content) - len(expected_content) + 1)
    )


def classify_question_type(
    question_text: str,
    candidate_type: Optional[QuestionType] = None,
    is_multipage: bool = False,
) -> QuestionType:
    """Classify question type based on grammatical structure, semantics, and candidate provenance."""
    if is_multipage:
        return QuestionType.MULTI_PAGE_SYNTHESIS

    q_lower = question_text.strip().lower()

    # Explicit candidate type hints from evidence extraction
    if candidate_type == QuestionType.MULTI_PAGE_SYNTHESIS:
        return QuestionType.MULTI_PAGE_SYNTHESIS

    # Comparison
    if (
        "compare" in q_lower
        or "difference between" in q_lower
        or "versus" in q_lower
        or "in contrast" in q_lower
        or "distinguish between" in q_lower
        or candidate_type == QuestionType.COMPARISON
    ):
        return QuestionType.COMPARISON

    # Definition
    if (
        "definition of" in q_lower
        or "what is meant by" in q_lower
        or "define" in q_lower
        or "defined as" in q_lower
        or "what is the concept of" in q_lower
        or candidate_type == QuestionType.DEFINITION
    ):
        return QuestionType.DEFINITION

    # Numerical fact / numerical
    if (
        q_lower.startswith("how many")
        or q_lower.startswith("how much")
        or "what percentage" in q_lower
        or "how high" in q_lower
        or "what rate" in q_lower
        or candidate_type in (QuestionType.NUMERICAL, QuestionType.NUMERICAL_FACT)
    ):
        if candidate_type == QuestionType.NUMERICAL_FACT:
            return QuestionType.NUMERICAL_FACT
        if q_lower.startswith("how many") and candidate_type is None:
            return QuestionType.HOW_MANY
        return QuestionType.NUMERICAL_FACT

    # Explanation / Why
    if "explain" in q_lower or "what causes" in q_lower or "what is the mechanism" in q_lower:
        return QuestionType.EXPLANATION

    if q_lower.startswith("why"):
        return QuestionType.EXPLANATION if candidate_type == QuestionType.EXPLANATION else QuestionType.WHY

    # Reasoning
    if (
        "reason" in q_lower
        or "result of" in q_lower
        or "consequence of" in q_lower
        or "impact of" in q_lower
        or "what follows from" in q_lower
        or candidate_type == QuestionType.REASONING
    ):
        return QuestionType.REASONING

    # Wh-question types (for backward compatibility & fine-grained classification)
    if q_lower.startswith("who"):
        return QuestionType.WHO
    elif q_lower.startswith("where"):
        return QuestionType.WHERE
    elif q_lower.startswith("when") or "what year" in q_lower or "what date" in q_lower:
        return QuestionType.WHEN
    elif q_lower.startswith("how"):
        return QuestionType.HOW
    elif q_lower.startswith("what"):
        if candidate_type == QuestionType.DIRECT_FACT:
            return QuestionType.DIRECT_FACT
        return QuestionType.WHAT

    if candidate_type in (QuestionType.DIRECT_FACT, QuestionType.FACTUAL):
        return candidate_type

    return QuestionType.FACTUAL


def classify_question_difficulty(
    question_type: QuestionType,
    answer_text: str,
    source_text: str,
) -> QuestionDifficulty:
    """Assign heuristic difficulty classification to a question based on structural and cognitive load."""
    answer_words = answer_text.strip().split()
    word_count = len(answer_words)

    # Multi-page synthesis, reasoning, explanations, and comparisons inherently demand higher synthesis
    if question_type in (
        QuestionType.MULTI_PAGE_SYNTHESIS,
        QuestionType.REASONING,
        QuestionType.COMPARISON,
        QuestionType.EXPLANATION,
        QuestionType.WHY,
        QuestionType.HOW,
    ):
        return QuestionDifficulty.HARD

    if question_type in (
        QuestionType.WHO,
        QuestionType.WHEN,
        QuestionType.WHERE,
        QuestionType.HOW_MANY,
        QuestionType.NUMERICAL,
        QuestionType.NUMERICAL_FACT,
        QuestionType.DIRECT_FACT,
    ):
        if word_count <= 4 and len(source_text) < 700:
            return QuestionDifficulty.EASY
        return QuestionDifficulty.MEDIUM

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

        if self._purpose_clause_mismatches_question(candidate):
            return False, None, "Extracted purpose clause does not answer a how/method question."

        # 2. Source Grounding Check
        if expected_ans.lower() not in source_text.lower():
            return False, None, "Expected answer does not occur in source evidence."

        # 3. Outside Knowledge Check (Ensure no introduced entity names absent from evidence)
        outside_err = self._check_outside_knowledge(q_text, source_text)
        if outside_err:
            return False, None, outside_err

        # 4. Extractive QA Verification
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
        definition_expansion = _is_grounded_definition_expansion(
            predicted_ans,
            expected_ans,
            q_text,
            source_text,
        )
        if not matches_expected_answer(
            predicted=predicted_ans,
            expected=expected_ans,
            question=q_text,
            source_text=source_text,
        ):
            return (
                False,
                None,
                f"QA predicted answer '{predicted_ans}' does not match expected answer '{expected_ans}'.",
            )

        # 5. Construct Validated Question with Complete Provenance & ID
        is_multipage = bool(ans_cand.metadata.get("is_multipage", False) or len(ans_cand.chunk_ids) > 1)
        q_type = classify_question_type(
            question_text=q_text,
            candidate_type=ans_cand.candidate_type,
            is_multipage=is_multipage,
        )
        difficulty = classify_question_difficulty(q_type, expected_ans, source_text)
        question_id = f"qgen_{ans_cand.document_id}_{ans_cand.chunk_id}_{abs(hash(q_text)) % 10000:04d}"

        chunk_ids = ans_cand.chunk_ids if ans_cand.chunk_ids else [ans_cand.chunk_id]
        page_numbers = ans_cand.page_numbers if ans_cand.page_numbers else [ans_cand.page_number]

        subordinate_clause = _is_grounded_subordinate_clause_match(
            predicted_ans, expected_ans, q_text
        )
        use_predicted = definition_expansion or subordinate_clause
        answer_text = predicted_ans if use_predicted else expected_ans
        answer_start = qa_res.answer_start if use_predicted else ans_cand.start_offset
        answer_end = qa_res.answer_end if use_predicted else ans_cand.end_offset
        validated = GeneratedQuestion(
            question_id=question_id,
            question=q_text,
            answer=answer_text,
            question_type=q_type,
            difficulty=difficulty,
            document_id=ans_cand.document_id,
            chunk_id=ans_cand.chunk_id,
            chunk_ids=chunk_ids,
            page_number=ans_cand.page_number,
            page_numbers=page_numbers,
            source_text=source_text,
            start_offset=answer_start,
            end_offset=answer_end,
            qa_predicted_answer=predicted_ans,
            qa_confidence_score=qa_res.qa_score,
            validation_status="validated",
            metadata={
                "qa_score": qa_res.qa_score,
                "no_answer_score": qa_res.no_answer_score,
                "candidate_type": ans_cand.candidate_type.value if hasattr(ans_cand.candidate_type, "value") else str(ans_cand.candidate_type),
                "is_multipage": is_multipage,
                "answer_match_type": (
                    "grounded_definition_expansion"
                    if definition_expansion
                    else ("grounded_subordinate_clause" if subordinate_clause else "content_tokens")
                ),
            },
        )

        return True, validated, None

    @staticmethod
    def _purpose_clause_mismatches_question(candidate: QuestionCandidate) -> bool:
        """Do not treat an intended outcome as an explanation of how a process works."""
        trigger = str(candidate.answer_candidate.metadata.get("extraction_trigger", "")).lower()
        if trigger != "in order to":
            return False
        question = normalize_answer_text(candidate.question_text)
        return question.startswith(("how ", "by what method ", "what method ", "which method "))

    def _check_quality(self, question: str, answer: str, source_text: str) -> Optional[str]:
        """Perform deterministic quality checks on question candidate."""
        if not question or not question.strip():
            return "Question text is empty."
        clean_q = question.strip()
        if len(clean_q) < 8:
            return f"Question is too short ({len(clean_q)} characters)."
        if len(clean_q) > 250:
            return f"Question is excessively long ({len(clean_q)} characters)."
        if not clean_q.endswith("?"):
            return "Question does not end with a question mark."

        # Malformed start with punctuation
        if clean_q[0] in "?!.,;:/-_":
            return "Question begins with invalid punctuation."
        if clean_q[0].isalpha() and not clean_q[0].isupper():
            return "Question must begin with an uppercase letter."

        # Reject concatenated questions or multiple question marks
        if clean_q.count("?") > 1:
            return "Question contains multiple question marks or concatenated questions."
        if re.search(r"([?!.,])\s*([?!.,])", clean_q):
            return "Question contains invalid repeated punctuation."

        # Cannot be identical to source text (ignoring trailing '?')

        clean_q_notrail = clean_q[:-1].strip() if clean_q.endswith("?") else clean_q
        if clean_q_notrail.lower() == source_text.strip().lower() or clean_q.lower() == source_text.strip().lower():
            return "Question is identical to source text."

        # Circular question: question literally is just the answer or contains answer in trivial circular way
        norm_q = normalize_answer_text(clean_q)
        norm_ans = normalize_answer_text(answer)
        if norm_q == norm_ans:
            return "Question is identical to the target answer."

        # Avoid questions where the entire question is "Who is {answer}?" or "What is {answer}?"
        if norm_q in {f"who is {norm_ans}", f"what is {norm_ans}", f"where is {norm_ans}"}:
            return "Question is a trivial tautological restatement of the answer."

        # Generic and context-dependent questions that cannot stand on their own.
        generic_prefixes = (
            "what is this", "what is that", "what is it", "what is this about",
            "what does this mean", "what is the meaning of this", "what does this chapter discuss",
            "what is the book about", "what is learning", "how do computers learn",
            "what is this book about", "what are these",
        )
        if norm_q.startswith(generic_prefixes):
            return "Question is too generic or context-dependent to be independently answerable."

        if re.search(r"\b(?:this|that|these|those|it|they|them|such|the above|the following|the former|the latter)\b", clean_q, re.IGNORECASE):
            # Allow references only when they are anchored by nearby concrete content.
            if not re.search(r"\b(?:what|which|who|when|where|why|how)\s+(?:is|are|was|were|did|does|do|can|could|would|should)\s+[^?]*\b(?:[A-Z][a-z]+|\d)\b", clean_q):
                return "Question relies on ambiguous references that cannot be understood independently."

        # Incomplete / dangling questions (e.g. ending in dangling preposition or conjunction before '?')
        dangling_patterns = [
            r"\b(role of|impact of|effect of|because of|because|and|or|but|that|with|from|by|about|in|at|to|which|for|as)\s*\?$",
        ]
        for pattern in dangling_patterns:
            if re.search(pattern, clean_q, re.IGNORECASE):
                return "Question has an incomplete sentence structure or dangling preposition/conjunction."

        # Vague questions
        vague_phrases = {
            "what is this", "what is that", "what is it", "what happened",
            "what did they do", "why is that", "who did this", "what are these",
            "how did this happen", "what does it do",
        }
        if norm_q in vague_phrases:
            return "Question is too vague or generic to be answerable."

        return None

    def _check_outside_knowledge(self, question: str, source_text: str) -> Optional[str]:
        """Verify that question does not introduce outside entity names not found in source text."""
        # Find proper nouns (capitalized words excluding sentence starter if generic)
        tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", question)
        source_lower = source_text.lower()

        # Stop list of common question words capitalized at start of sentence
        common_starts = {
            "What", "Which", "Where", "When", "Who", "Whom", "Why", "How",
            "Name", "State", "Define", "Explain", "Compare", "Describe", "Identify",
            "Give", "List", "Does", "According", "In", "Based", "From",
        }

        for tok in tokens:
            if tok in common_starts:
                continue
            if tok.lower() not in source_lower:
                return f"Question introduces outside entity or unsupported concept '{tok}'."
        return None

