from app.schemas.chunk import Chunk
from app.schemas.qa import QAResponse
from app.schemas.question_generation import AnswerCandidate, QuestionCandidate
from app.services.question_generation.answers import AnswerCandidateExtractor
from app.services.question_generation.eligibility import is_eligible
from app.services.question_generation.service import QuestionGenerationService
from app.services.question_generation.validator import QuestionValidator


class _MockQGModel:
    def __init__(self, mapping=None):
        self.mapping = mapping or {
            "Guido van Rossum": "Who created Python?",
            "1991": "When was Python first released?",
            "Netherlands": "Where was Python created?",
        }

    def generate_questions_batch(self, pairs):
        results = []
        for answer, _ in pairs:
            results.append(self.mapping.get(answer, f"What is {answer}?"))
        return results


class _MockQAService:
    def answer_question(self, query, evidence, **kwargs):
        answer = "Guido van Rossum" if "Who" in query else ("1991" if "When" in query else "Netherlands")
        return QAResponse(
            query=query,
            answer=answer,
            answerable=True,
            qa_score=0.95,
            no_answer_score=0.0,
            document_id="doc_test",
            chunk_id="chunk_1",
            page_number=1,
            chunk_index=0,
            answer_start=0,
            answer_end=len(answer),
            source_text=evidence[0]["text"] if evidence else "",
            evidence_rank=1,
        )


def _make_chunk(text: str, *, chunk_id: str = "chunk_1", page_number: int = 1, document_id: str = "doc_test") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page_number=page_number,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        char_count=len(text),
        word_count=len(text.split()),
    )


def test_eligibility_rejects_front_matter_and_metadata_but_accepts_technical_content():
    assert is_eligible(_make_chunk("Preface\nThis book introduces the problem.")).eligible is False
    assert is_eligible(_make_chunk("References\nSmith, J. 2020. Deep Learning.")).eligible is False
    assert is_eligible(_make_chunk("The optimizer updates weights using gradient descent and backpropagation on each training step.")).eligible is True


def test_ineligible_chunks_never_reach_extractor():
    model = _MockQGModel()
    qa = _MockQAService()
    service = QuestionGenerationService(model=model, validator=QuestionValidator(qa_service=qa))
    chunks = [
        _make_chunk("Abstract\nThis book introduces the topic.", chunk_id="chunk_abs"),
        _make_chunk("Python was created by Guido van Rossum in 1991 in the Netherlands.", chunk_id="chunk_tech"),
    ]

    response = service.generate_from_chunks(chunks, document_id="doc_test", count=1)

    assert response.returned_count == 1
    assert response.questions[0].chunk_id == "chunk_tech"
    assert response.rejection_summary["ABSTRACT"] == 1
    assert response.questions[0].document_id == "doc_test"


def test_service_keeps_provenance_and_eligibility_summary():
    model = _MockQGModel()
    service = QuestionGenerationService(model=model, validator=QuestionValidator(qa_service=_MockQAService()))
    chunks = [
        _make_chunk("Abstract\nThis book introduces the topic.", chunk_id="chunk_abs"),
        _make_chunk("Python was created by Guido van Rossum in 1991 in the Netherlands.", chunk_id="chunk_tech"),
    ]

    response = service.generate_from_chunks(chunks, document_id="doc_test", count=1)
    assert response.returned_count == 1
    assert response.questions[0].document_id == "doc_test"
    assert response.questions[0].chunk_id == "chunk_tech"
    assert response.rejection_summary["total_chunks"] == 2
    assert response.rejection_summary["eligible_chunks"] == 1


def test_validator_rejects_generic_and_ambiguous_questions():
    validator = QuestionValidator(qa_service=_MockQAService())

    generic = QuestionCandidate(
        question_text="What is this?",
        answer_candidate=AnswerCandidate(
            answer_text="gradient descent",
            source_text="Gradient descent updates parameters by following the negative gradient of the loss function.",
            document_id="doc_test",
            chunk_id="chunk_1",
            page_number=1,
        ),
    )
    ambiguous = QuestionCandidate(
        question_text="Why is it difficult?",
        answer_candidate=AnswerCandidate(
            answer_text="overfitting",
            source_text="Overfitting occurs when a model fits noise instead of the underlying pattern.",
            document_id="doc_test",
            chunk_id="chunk_2",
            page_number=2,
        ),
    )

    is_valid, _, generic_reason = validator.validate_candidate(generic)
    assert is_valid is False
    assert generic_reason is not None

    is_valid, _, ambiguous_reason = validator.validate_candidate(ambiguous)
    assert is_valid is False
    assert ambiguous_reason is not None
