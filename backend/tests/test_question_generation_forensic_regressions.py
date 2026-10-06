"""Forensic Regression Tests for Question Generation System in BookRAG AI.

Covers all 17 root causes identified during the 24-candidate forensic audit:
1. short valid answer
2. longer valid answer (e.g. grounded definition expansion, subordinate clause)
3. false substring match (e.g. 'linear' inside 'rectified linear unit')
4. fragment answer (e.g. 'closest', 'follows: LMS weight update rule')
5. hyphenated fragment (e.g. sub-word fragments like 'Linear' in 'Non-linear')
6. malformed question (multiple question marks, repeated punctuation)
7. generic question (e.g. 'what is this', 'what did they do')
8. context-dependent question (ambiguous 'this series', 'this book')
9. unanswerable question (QA model predicts unanswerable/low confidence)
10. duplicate question
11. normalized duplicate (case, quotes, punctuation variations)
12. technical terminology (properly preserving uppercase acronyms and concepts)
13. equation answer / heading prefix noise (e.g. 'follows: LMS weight update rule')
14. provenance (complete document_id, page_numbers, chunk_ids, start_offset)
15. eligibility (filtering front matter/references while preserving technical content)
16. RoBERTa false negative recovery (entity suffixes like 'SPHINX' vs 'SPHINX system')
17. answer mismatch rejection (e.g. speed '70 miles' vs distance '90 miles')
"""

import pytest
from app.schemas.chunk import Chunk
from app.schemas.qa import QAResponse
from app.schemas.question_generation import (
    AnswerCandidate,
    QuestionCandidate,
    QuestionType,
    QuestionDifficulty,
)
from app.services.question_generation.answers import AnswerCandidateExtractor
from app.services.question_generation.deduplication import (
    QuestionDeduplicator,
    normalize_question_for_dedup,
)
from app.services.question_generation.eligibility import is_eligible
from app.services.question_generation.validator import (
    QuestionValidator,
    matches_expected_answer,
    _is_grounded_definition_expansion,
    _is_grounded_subordinate_clause_match,
)


def _make_chunk(text: str, chunk_id: str = "c01", page: int = 1, doc_id: str = "doc_test") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        page_number=page,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        char_count=len(text),
        word_count=len(text.split()),
    )


class _MockQAService:
    def __init__(self, answer="test answer", answerable=True, score=0.9, no_ans_score=0.1):
        self._answer = answer
        self._answerable = answerable
        self._score = score
        self._no_ans_score = no_ans_score

    def answer_question(self, query, evidence, **kwargs):
        return QAResponse(
            query=query,
            answer=self._answer,
            answerable=self._answerable,
            qa_score=self._score,
            no_answer_score=self._no_ans_score,
            document_id=evidence[0]["document_id"] if evidence else "doc_1",
            chunk_id=evidence[0]["chunk_id"] if evidence else "c_1",
            page_number=evidence[0].get("page_number", 1) if evidence else 1,
            chunk_index=0,
            answer_start=0,
            answer_end=len(self._answer) if self._answer else 0,
            source_text=evidence[0]["text"] if evidence else "",
            evidence_rank=1,
        )


# 1. Short valid answer
def test_short_valid_answer():
    assert matches_expected_answer(predicted="1997", expected="1997") is True
    assert matches_expected_answer(predicted="SPHINX", expected="SPHINX") is True


# 2. Longer valid answer (definition expansion & subordinate clause)
def test_longer_valid_answer_definition_expansion():
    source = 'Our definition of learning is broad enough to include most tasks that we would conventionally call "learning" tasks.'
    q = "What is the definition of learning?"
    expected = 'broad enough to include most tasks that we would conventionally call "learning" tasks'
    predicted = expected
    assert matches_expected_answer(predicted, expected, question=q, source_text=source) is True


def test_longer_valid_answer_subordinate_clause():
    expected = "the game can be lost even when early moves are optimal, if these are followed later by poor moves"
    predicted = "if these are followed later by poor moves"
    q = "How can the game be lost when early moves are optimal?"
    assert matches_expected_answer(predicted, expected, question=q) is True


# 3. False substring match (linear inside rectified linear unit)
def test_false_substring_match_strictly_rejected():
    assert matches_expected_answer(predicted="rectified linear unit", expected="linear") is False
    assert matches_expected_answer(predicted="linear", expected="rectified linear unit") is False


# 4. Fragment answer rejection at extractor
def test_fragment_answer_rejection():
    extractor = AnswerCandidateExtractor()
    # "closest" in quotes should not be extracted as standalone answer
    text = 'We choose the "closest" stored prototype to classify any unseen instance.'
    chunk = _make_chunk(text)
    cands = extractor.extract_from_chunk(chunk)
    assert not any(c.answer_text.lower() == "closest" for c in cands)


# 5. Hyphenated fragment rejection
def test_hyphenated_fragment_rejection():
    extractor = AnswerCandidateExtractor()
    text = "The system uses non-linear decision boundaries for high dimensional classification."
    chunk = _make_chunk(text)
    cands = extractor.extract_from_chunk(chunk)
    # Sub-word "linear" should not be extracted from "non-linear"
    assert not any(c.answer_text.lower() == "linear" for c in cands)


# 6. Malformed question rejection (multiple question marks, repeated punctuation)
def test_malformed_question_rejected():
    validator = QuestionValidator(qa_service=_MockQAService())
    cand1 = QuestionCandidate(
        candidate_id="c1",
        question_text="What does V map to? ? In the next section of the article?",
        answer_candidate=AnswerCandidate(
            answer_text="real values",
            source_text="V maps states to real values.",
            document_id="doc_1",
            chunk_id="c_1",
            page_number=1,
            candidate_type=QuestionType.FACTUAL,
        ),
        question_type=QuestionType.FACTUAL,
        difficulty=QuestionDifficulty.MEDIUM,
    )
    is_valid, _, reason = validator.validate_candidate(cand1)
    assert is_valid is False
    assert "multiple question marks" in reason or "repeated punctuation" in reason


# 7. Generic question rejection
def test_generic_question_rejected():
    validator = QuestionValidator(qa_service=_MockQAService())
    cand = QuestionCandidate(
        candidate_id="c1",
        question_text="What does it do?",
        answer_candidate=AnswerCandidate(
            answer_text="it updates weights",
            source_text="It updates weights according to the gradient.",
            document_id="doc_1",
            chunk_id="c_1",
            page_number=1,
            candidate_type=QuestionType.FACTUAL,
        ),
        question_type=QuestionType.FACTUAL,
        difficulty=QuestionDifficulty.EASY,
    )
    is_valid, _, reason = validator.validate_candidate(cand)
    assert is_valid is False
    assert "vague" in reason or "generic" in reason or "ambiguous" in reason


# 8. Context-dependent question rejection (this series, this book)
def test_context_dependent_question_rejected():
    validator = QuestionValidator(qa_service=_MockQAService())
    cand = QuestionCandidate(
        candidate_id="c1",
        question_text="What is the most recent article in this series?",
        answer_candidate=AnswerCandidate(
            answer_text="1989",
            source_text="Recent articles: Waibel 1989, Lee 1989.",
            document_id="doc_1",
            chunk_id="c_1",
            page_number=1,
            candidate_type=QuestionType.WHEN,
        ),
        question_type=QuestionType.WHEN,
        difficulty=QuestionDifficulty.EASY,
    )
    is_valid, _, reason = validator.validate_candidate(cand)
    assert is_valid is False
    assert "ambiguous reference" in reason.lower()


# 9. Unanswerable question rejection
def test_unanswerable_question_rejected():
    qa = _MockQAService(answer="", answerable=False, score=-2.0, no_ans_score=6.0)
    validator = QuestionValidator(qa_service=qa)
    cand = QuestionCandidate(
        candidate_id="c1",
        question_text="What learning systems combine both?",
        answer_candidate=AnswerCandidate(
            answer_text="some learning systems combine both",
            source_text="Whereas ID3 exhibits preference bias, some learning systems combine both.",
            document_id="doc_1",
            chunk_id="c_1",
            page_number=1,
            candidate_type=QuestionType.FACTUAL,
        ),
        question_type=QuestionType.FACTUAL,
        difficulty=QuestionDifficulty.HARD,
    )
    is_valid, _, reason = validator.validate_candidate(cand)
    assert is_valid is False
    assert "unanswerable" in reason.lower()


# 10. Duplicate question rejection
def test_duplicate_question_rejection():
    dedup = QuestionDeduplicator()
    ans = AnswerCandidate(
        answer_text="SPHINX",
        source_text="SPHINX speech recognition.",
        document_id="doc_1",
        chunk_id="c_1",
        page_number=1,
        candidate_type=QuestionType.FACTUAL,
    )
    c1 = QuestionCandidate(
        candidate_id="cand_1",
        question_text="What is the most successful speech recognition system?",
        answer_candidate=ans,
        question_type=QuestionType.FACTUAL,
        difficulty=QuestionDifficulty.EASY,
    )
    c2 = QuestionCandidate(
        candidate_id="cand_2",
        question_text="What is the most successful speech recognition system?",
        answer_candidate=ans,
        question_type=QuestionType.FACTUAL,
        difficulty=QuestionDifficulty.EASY,
    )
    unique, dups = dedup.filter_candidates_with_rejected([c1, c2])
    assert len(unique) == 1
    assert len(dups) == 1
    assert dups[0][1] == "exact_duplicate"


# 11. Normalized duplicate rejection (quotes, punctuation, capitalization)
def test_normalized_duplicate_rejection():
    dedup = QuestionDeduplicator()
    ans = AnswerCandidate(
        answer_text="learning",
        source_text="Definition of learning.",
        document_id="doc_1",
        chunk_id="c_1",
        page_number=1,
        candidate_type=QuestionType.DEFINITION,
    )
    c1 = QuestionCandidate(
        candidate_id="cand_1",
        question_text="What is the definition of learning?",
        answer_candidate=ans,
        question_type=QuestionType.DEFINITION,
        difficulty=QuestionDifficulty.MEDIUM,
    )
    c2 = QuestionCandidate(
        candidate_id="cand_2",
        question_text="what is the definition of 'learning'?",
        answer_candidate=ans,
        question_type=QuestionType.DEFINITION,
        difficulty=QuestionDifficulty.MEDIUM,
    )
    unique, dups = dedup.filter_candidates_with_rejected([c1, c2])
    assert len(unique) == 1
    assert len(dups) == 1
    assert "duplicate" in dups[0][1]


# 12. Technical terminology preserved
def test_technical_terminology_preserved():
    extractor = AnswerCandidateExtractor()
    text = "The Backpropagation algorithm trains Multilayer Perceptrons using gradient descent."
    chunk = _make_chunk(text)
    cands = extractor.extract_from_chunk(chunk)
    terms = [c.answer_text for c in cands]
    assert any("Backpropagation" in t or "Multilayer" in t for t in terms)


# 13. Equation answer / heading prefix noise rejection
def test_equation_and_heading_noise_rejected():
    extractor = AnswerCandidateExtractor()
    text = "The LMS algorithm is defined as follows: LMS weight update rule wi = wi + delta_w."
    chunk = _make_chunk(text)
    cands = extractor.extract_from_chunk(chunk)
    assert not any(c.answer_text.lower().startswith("follows:") for c in cands)


# 14. Provenance preservation
def test_provenance_preservation():
    qa = _MockQAService(answer="1997", answerable=True, score=0.99)
    validator = QuestionValidator(qa_service=qa)
    ans = AnswerCandidate(
        answer_text="1997",
        source_text="Cooper et al. 1997 predicted pneumonia recovery.",
        document_id="doc_mitchell",
        chunk_id="p014_c0010",
        chunk_ids=["p014_c0010"],
        page_number=14,
        page_numbers=[14],
        start_offset=14,
        end_offset=18,
        candidate_type=QuestionType.WHEN,
    )
    cand = QuestionCandidate(
        candidate_id="cand_1",
        question_text="What year did Cooper et al. predict recovery rates of pneumonia patients?",
        answer_candidate=ans,
        question_type=QuestionType.WHEN,
        difficulty=QuestionDifficulty.EASY,
    )
    is_valid, q_val, reason = validator.validate_candidate(cand)
    assert is_valid is True
    assert q_val.document_id == "doc_mitchell"
    assert q_val.chunk_id == "p014_c0010"
    assert q_val.page_number == 14
    assert q_val.page_numbers == [14]
    assert q_val.chunk_ids == ["p014_c0010"]


# 15. Eligibility checks
def test_eligibility_preserves_technical_content_rejects_metadata():
    front_matter = _make_chunk("TABLE OF CONTENTS\nChapter 1 Introduction\nChapter 2 Concept Learning")
    assert is_eligible(front_matter).eligible is False

    ref_chunk = _make_chunk("REFERENCES\nAhn, W., & Brewer, W. F. (1993). Psychological studies of learning.")
    assert is_eligible(ref_chunk).eligible is False

    tech_chunk = _make_chunk("Overfitting is a significant practical difficulty for decision tree learning, decreasing accuracy by 10-25%.")
    assert is_eligible(tech_chunk).eligible is True


# 16. RoBERTa false negative recovery (entity suffixes)
def test_roberta_entity_suffix_match():
    assert matches_expected_answer(predicted="SPHINX system", expected="SPHINX") is True
    assert matches_expected_answer(predicted="h2 is more general", expected="more general than") is True


# 17. Answer mismatch rejection (numbers strictly checked)
def test_answer_mismatch_strictly_rejected():
    assert matches_expected_answer(predicted="90 miles", expected="70 miles") is False
    assert matches_expected_answer(
        predicted="learning algorithms developed for computers",
        expected="Chi and Bassock",
    ) is False
