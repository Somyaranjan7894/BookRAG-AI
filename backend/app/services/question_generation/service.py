"""High-level orchestration service for Phase 12 Question Generation in BookRAG AI.

Coordinates:
1. Evidence chunk retrieval for a target book document (with document isolation).
2. Grounded answer candidate extraction from evidence passages.
3. Conditioned question generation with iarfmoose/t5-base-question-generator.
4. Duplicate candidate detection.
5. Extractive QA verification with RoBERTa SQuAD2 and answer matching.
6. Controlled count limits without fabrication.
"""

from typing import List, Optional, Sequence
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.chunk import Chunk
from app.schemas.question_generation import (
    GeneratedQuestion,
    QuestionCandidate,
    QuestionDifficulty,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
)
from app.services.qa.service import QAService
from app.services.question_generation.answers import AnswerCandidateExtractor
from app.services.question_generation.deduplication import QuestionDeduplicator
from app.services.question_generation.exceptions import QuestionGenerationError
from app.services.question_generation.model import QuestionGenerationModel
from app.services.question_generation.validator import (
    QuestionValidator,
    classify_question_difficulty,
    classify_question_type,
)
from app.services.retrieval.service import RetrievalService
from app.services.search.exceptions import DocumentNotFoundError
from app.services.search.service import SearchService

logger = get_logger(__name__)


class QuestionGenerationService:
    """Orchestrates evidence-grounded question generation and validation."""

    def __init__(
        self,
        model: Optional[QuestionGenerationModel] = None,
        extractor: Optional[AnswerCandidateExtractor] = None,
        validator: Optional[QuestionValidator] = None,
        qa_service: Optional[QAService] = None,
        search_service: Optional[SearchService] = None,
        retrieval_service: Optional[RetrievalService] = None,
    ) -> None:
        self._model = model
        self._extractor = extractor or AnswerCandidateExtractor()
        self._qa_service = qa_service
        self._validator = validator
        self._search_service = search_service
        self._retrieval_service = retrieval_service

    @property
    def model(self) -> QuestionGenerationModel:
        if self._model is None:
            self._model = QuestionGenerationModel.get_instance()
        return self._model

    @property
    def extractor(self) -> AnswerCandidateExtractor:
        return self._extractor

    @property
    def validator(self) -> QuestionValidator:
        if self._validator is None:
            self._validator = QuestionValidator(qa_service=self._qa_service)
        return self._validator

    @property
    def search_service(self) -> SearchService:
        if self._search_service is None:
            self._search_service = SearchService()
        return self._search_service

    @property
    def retrieval_service(self) -> RetrievalService:
        if self._retrieval_service is None:
            self._retrieval_service = self.search_service.retrieval_service
        return self._retrieval_service

    def generate_questions(
        self,
        request: QuestionGenerationRequest,
    ) -> QuestionGenerationResponse:
        """Generate validated, evidence-grounded questions for a document."""
        doc_id = request.document_id.strip()
        count = request.count

        # 1. Retrieve chunks for the specified document
        chunks = self._get_document_chunks(
            document_id=doc_id,
            chapter=request.chapter,
            page_number=request.page_number,
        )

        return self.generate_from_chunks(
            chunks=chunks,
            document_id=doc_id,
            count=count,
            target_difficulty=request.difficulty,
        )

    def generate_from_chunks(
        self,
        chunks: Sequence[Chunk],
        document_id: str,
        count: int,
        target_difficulty: Optional[QuestionDifficulty] = None,
    ) -> QuestionGenerationResponse:
        """Generate validated questions from an explicit list of chunks."""
        if not chunks:
            return QuestionGenerationResponse(
                document_id=document_id,
                requested_count=count,
                generated_candidates=0,
                validated_count=0,
                returned_count=0,
                questions=[],
            )

        # 2. Extract answer candidates across chunks
        multiplier = settings.QUESTION_GEN_CANDIDATE_MULTIPLIER
        target_candidate_pool = max(count * multiplier, count + 2)

        answer_candidates = []
        for chunk in chunks:
            extracted = self.extractor.extract_from_chunk(chunk)
            answer_candidates.extend(extracted)
            if len(answer_candidates) >= target_candidate_pool * 2:
                break

        if not answer_candidates:
            return QuestionGenerationResponse(
                document_id=document_id,
                requested_count=count,
                generated_candidates=0,
                validated_count=0,
                returned_count=0,
                questions=[],
            )

        # 3. Generate questions for candidates in batches
        generation_pairs = [
            (ac.answer_text, ac.source_text) for ac in answer_candidates[:target_candidate_pool]
        ]
        generated_texts = self.model.generate_questions_batch(generation_pairs)

        raw_candidates: List[QuestionCandidate] = []
        for i, q_text in enumerate(generated_texts):
            ans_cand = answer_candidates[i]
            q_type = classify_question_type(q_text)
            diff = classify_question_difficulty(q_type, ans_cand.answer_text, ans_cand.source_text)
            raw_candidates.append(
                QuestionCandidate(
                    question_text=q_text,
                    answer_candidate=ans_cand,
                    question_type=q_type,
                    difficulty=diff,
                    raw_model_output=q_text,
                )
            )

        # 4. Deduplicate candidate questions
        deduplicator = QuestionDeduplicator()
        unique_candidates = deduplicator.filter_candidates(raw_candidates)

        # 5. Validate candidates with Extractive QA & quality checks
        validated_questions: List[GeneratedQuestion] = []
        validated_count = 0

        for candidate in unique_candidates:
            is_valid, validated_q, rejection_reason = self.validator.validate_candidate(candidate)
            if is_valid and validated_q is not None:
                validated_count += 1
                # Check difficulty filter if specified
                if target_difficulty is not None and validated_q.difficulty != target_difficulty:
                    continue

                validated_questions.append(validated_q)
                if len(validated_questions) >= count:
                    break

        final_questions = validated_questions[:count]

        return QuestionGenerationResponse(
            document_id=document_id,
            requested_count=count,
            generated_candidates=len(raw_candidates),
            validated_count=validated_count,
            returned_count=len(final_questions),
            questions=final_questions,
        )

    def _get_document_chunks(
        self,
        document_id: str,
        chapter: Optional[int] = None,
        page_number: Optional[int] = None,
    ) -> List[Chunk]:
        """Fetch chunks for a specific document from the loaded vector index mapping."""
        # Find index containing this document
        target_index = None
        for idx in self.retrieval_service._indices.values():
            if document_id in idx.metadata.document_ids:
                target_index = idx
                break

        if target_index is None:
            # Check default index if initialized
            if self.retrieval_service._default_index_id:
                default_idx = self.retrieval_service.get_index()
                if document_id in default_idx.metadata.document_ids:
                    target_index = default_idx

        if target_index is None:
            raise DocumentNotFoundError(f"Document '{document_id}' not found in any active index.")

        matching_chunks: List[Chunk] = []
        for item in target_index._mapping:
            if item.document_id == document_id:
                if page_number is not None and item.page_number != page_number:
                    continue
                if chapter is not None and item.metadata.get("chapter") != chapter:
                    continue
                matching_chunks.append(
                    Chunk(
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        page_number=item.page_number,
                        chunk_index=item.chunk_index,
                        text=item.text,
                        start_char=0,
                        end_char=len(item.text),
                        char_count=len(item.text),
                        word_count=len(item.text.split()),
                    )
                )

        return matching_chunks
