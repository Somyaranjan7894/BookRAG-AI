"""High-level orchestration service for Phase 12 Question Generation in BookRAG AI.

Coordinates:
1. Evidence chunk retrieval for a target book document (with document isolation).
2. Grounded answer candidate extraction from evidence passages.
3. Conditioned question generation with iarfmoose/t5-base-question-generator.
4. Duplicate candidate detection.
5. Extractive QA verification with RoBERTa SQuAD2 and answer matching.
6. Controlled count limits without fabrication.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.chunk import Chunk
from app.schemas.question_generation import (
    GeneratedQuestion,
    QuestionCandidate,
    QuestionDifficulty,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionType,
    RejectedCandidateInfo,
)
from app.services.persistence import DocumentPersistenceService
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
        persistence: Optional[DocumentPersistenceService] = None,
    ) -> None:
        self._model = model
        self._extractor = extractor or AnswerCandidateExtractor()
        self._qa_service = qa_service
        self._validator = validator
        self._search_service = search_service
        self._retrieval_service = retrieval_service
        self._persistence = persistence

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
            target_type=request.question_type,
            include_rejected=request.include_rejected,
            ensure_diversity=request.ensure_diversity,
        )

    def generate_from_chunks(
        self,
        chunks: Sequence[Chunk],
        document_id: str,
        count: int,
        target_difficulty: Optional[QuestionDifficulty] = None,
        target_type: Optional[QuestionType] = None,
        include_rejected: bool = False,
        ensure_diversity: bool = True,
    ) -> QuestionGenerationResponse:
        """Generate validated questions from an explicit list of chunks with complete provenance."""
        start_time = time.perf_counter()

        if not chunks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return QuestionGenerationResponse(
                document_id=document_id,
                requested_count=count,
                generated_candidates=0,
                validated_count=0,
                returned_count=0,
                questions=[],
                rejected_candidates=[],
                rejection_summary={"total_chunks": 0, "eligible_chunks": 0, "ineligible_chunks": 0},
                latency_ms=round(elapsed_ms, 2),
            )

        # 2. Filter chunks by eligibility before extraction
        from app.services.question_generation.eligibility import is_eligible, EligibilityResult
        multiplier = settings.QUESTION_GEN_CANDIDATE_MULTIPLIER
        target_candidate_pool = max(count * multiplier, count + 2)

        # Track ineligible statistics
        ineligible_counts: Dict[str, int] = defaultdict(int)
        eligible_chunks: List[Chunk] = []
        for chunk in chunks:
            result: EligibilityResult = is_eligible(chunk)
            if result.eligible:
                eligible_chunks.append(chunk)
            else:
                ineligible_counts[result.category] += 1

        rejection_summary: Dict[str, int] = {
            "total_chunks": len(chunks),
            "eligible_chunks": len(eligible_chunks),
            "ineligible_chunks": len(chunks) - len(eligible_chunks),
        }
        rejection_summary.update({k: v for k, v in ineligible_counts.items()})

        if not eligible_chunks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return QuestionGenerationResponse(
                document_id=document_id,
                requested_count=count,
                generated_candidates=0,
                validated_count=0,
                returned_count=0,
                questions=[],
                rejected_candidates=[],
                rejection_summary=rejection_summary,
                latency_ms=round(elapsed_ms, 2),
            )

        raw_extracted_candidates = []
        for chunk in eligible_chunks:
            extracted = self.extractor.extract_from_chunk(chunk)
            raw_extracted_candidates.extend(extracted)

        # Extract multi-page candidates across consecutive eligible chunks
        if len(eligible_chunks) > 1 and (target_type is None or target_type == QuestionType.MULTI_PAGE_SYNTHESIS):
            multipage_cands = self.extractor.extract_multipage_candidates(eligible_chunks)
            raw_extracted_candidates.extend(multipage_cands)

        if not raw_extracted_candidates:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return QuestionGenerationResponse(
                document_id=document_id,
                requested_count=count,
                generated_candidates=0,
                validated_count=0,
                returned_count=0,
                questions=[],
                rejected_candidates=[],
                rejection_summary={
                    "total_chunks": len(chunks),
                    "eligible_chunks": len(eligible_chunks),
                    "ineligible_chunks": len(chunks) - len(eligible_chunks),
                    **{k: v for k, v in ineligible_counts.items()},
                },
                latency_ms=round(elapsed_ms, 2),
            )

        # Apply target_type filter if specified
        if target_type is not None:
            filtered_cands = [c for c in raw_extracted_candidates if c.candidate_type == target_type]
            raw_extracted_candidates = filtered_cands

        if not raw_extracted_candidates:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return QuestionGenerationResponse(
                document_id=document_id,
                requested_count=count,
                generated_candidates=0,
                validated_count=0,
                returned_count=0,
                questions=[],
                rejected_candidates=[],
                rejection_summary=rejection_summary,
                latency_ms=round(elapsed_ms, 2),
            )

        # Select candidates respecting diversity across question types
        if ensure_diversity and len(raw_extracted_candidates) > target_candidate_pool:
            type_groups: Dict[str, List] = defaultdict(list)
            for cand in raw_extracted_candidates:
                t_key = cand.candidate_type.value if hasattr(cand.candidate_type, "value") else str(cand.candidate_type)
                type_groups[t_key].append(cand)

            selected_candidates = []
            keys = list(type_groups.keys())
            while len(selected_candidates) < target_candidate_pool and any(type_groups[k] for k in keys):
                for k in keys:
                    if type_groups[k] and len(selected_candidates) < target_candidate_pool:
                        selected_candidates.append(type_groups[k].pop(0))
            answer_candidates = selected_candidates
        else:
            answer_candidates = raw_extracted_candidates[:target_candidate_pool]

        # 3. Generate questions for candidates in batches
        generation_pairs = [
            (ac.answer_text, ac.source_text) for ac in answer_candidates
        ]
        generated_texts = self.model.generate_questions_batch(generation_pairs)

        raw_candidates: List[QuestionCandidate] = []
        for i, q_text in enumerate(generated_texts):
            ans_cand = answer_candidates[i]
            is_mp = bool(ans_cand.metadata.get("is_multipage", False) or len(ans_cand.chunk_ids) > 1)
            q_type = classify_question_type(
                question_text=q_text,
                candidate_type=ans_cand.candidate_type,
                is_multipage=is_mp,
            )
            diff = classify_question_difficulty(q_type, ans_cand.answer_text, ans_cand.source_text)
            candidate_id = f"cand_{ans_cand.document_id}_{ans_cand.chunk_id}_{i:03d}_{abs(hash(q_text)) % 10000:04d}"
            raw_candidates.append(
                QuestionCandidate(
                    candidate_id=candidate_id,
                    question_text=q_text,
                    answer_candidate=ans_cand,
                    question_type=q_type,
                    difficulty=diff,
                    raw_model_output=q_text,
                )
            )

        # 4. Deduplicate candidate questions
        deduplicator = QuestionDeduplicator()
        unique_candidates, duplicate_rejections = deduplicator.filter_candidates_with_rejected(raw_candidates)

        rejection_summary = defaultdict(int, {"total_chunks": len(chunks), "eligible_chunks": len(eligible_chunks), "ineligible_chunks": len(chunks) - len(eligible_chunks)})
        for key, value in ineligible_counts.items():
            rejection_summary[key] += value
        rejected_candidates: List[RejectedCandidateInfo] = []

        # Record duplicate rejections
        for dup_cand, dup_reason in duplicate_rejections:
            rejection_summary["duplicate"] += 1
            cid = dup_cand.candidate_id or f"cand_{dup_cand.answer_candidate.chunk_id}_{abs(hash(dup_cand.question_text)) % 10000:04d}"
            rejected_candidates.append(
                RejectedCandidateInfo(
                    candidate_id=cid,
                    question_text=dup_cand.question_text,
                    answer_text=dup_cand.answer_candidate.answer_text,
                    rejection_reason=dup_reason,
                    rejection_category="duplicate",
                    chunk_id=dup_cand.answer_candidate.chunk_id,
                    chunk_ids=dup_cand.answer_candidate.chunk_ids or [dup_cand.answer_candidate.chunk_id],
                    page_number=dup_cand.answer_candidate.page_number,
                    page_numbers=dup_cand.answer_candidate.page_numbers or [dup_cand.answer_candidate.page_number],
                    question_type=dup_cand.question_type,
                )
            )

        # 5. Validate candidates with Extractive QA & quality checks
        validated_questions: List[GeneratedQuestion] = []
        matching_filter_questions: List[GeneratedQuestion] = []
        validated_count = 0

        for candidate in unique_candidates:
            is_valid, validated_q, rejection_reason = self.validator.validate_candidate(candidate)
            if is_valid and validated_q is not None:
                validated_count += 1
                validated_questions.append(validated_q)

                # Check if it matches strict target filters
                matches_diff = target_difficulty is None or validated_q.difficulty == target_difficulty
                matches_type = target_type is None or validated_q.question_type == target_type

                if matches_diff and matches_type:
                    matching_filter_questions.append(validated_q)
            else:
                reason = rejection_reason or "Candidate validation failed."
                r_lower = reason.lower()
                if "unanswerable" in r_lower:
                    cat = "unanswerable"
                elif "outside entity" in r_lower or "unsupported concept" in r_lower or "outside" in r_lower:
                    cat = "outside_knowledge"
                elif "does not occur in source evidence" in r_lower:
                    cat = "grounding_failure"
                elif "does not match expected answer" in r_lower:
                    cat = "answer_mismatch"
                elif (
                    "too short" in r_lower
                    or "too long" in r_lower
                    or "empty" in r_lower
                    or "punctuation" in r_lower
                    or "dangling" in r_lower
                    or "vague" in r_lower
                    or "tautological" in r_lower
                    or "identical" in r_lower
                ):
                    cat = "quality_failure"
                else:
                    cat = "validation_failure"

                rejection_summary[cat] += 1
                cid = candidate.candidate_id or f"cand_{candidate.answer_candidate.chunk_id}_{abs(hash(candidate.question_text)) % 10000:04d}"
                rejected_candidates.append(
                    RejectedCandidateInfo(
                        candidate_id=cid,
                        question_text=candidate.question_text,
                        answer_text=candidate.answer_candidate.answer_text,
                        rejection_reason=reason,
                        rejection_category=cat,
                        chunk_id=candidate.answer_candidate.chunk_id,
                        chunk_ids=candidate.answer_candidate.chunk_ids or [candidate.answer_candidate.chunk_id],
                        page_number=candidate.answer_candidate.page_number,
                        page_numbers=candidate.answer_candidate.page_numbers or [candidate.answer_candidate.page_number],
                        question_type=candidate.question_type,
                    )
                )

        # Resolve answer alternatives before suppressing duplicate accepted questions.
        final_deduplicator = QuestionDeduplicator()
        deduplicated_validated: List[GeneratedQuestion] = []
        retained_question_objects = set()
        for validated_question in validated_questions:
            duplicate_reason = final_deduplicator.find_duplicate_reason(validated_question.question)
            if duplicate_reason is None:
                final_deduplicator.record(validated_question.question)
                deduplicated_validated.append(validated_question)
                retained_question_objects.add(id(validated_question))
                continue

            rejection_summary["duplicate"] += 1
            rejected_candidates.append(
                RejectedCandidateInfo(
                    candidate_id=validated_question.question_id or validated_question.chunk_id,
                    question_text=validated_question.question,
                    answer_text=validated_question.answer,
                    rejection_reason=f"duplicate_after_answer_validation:{duplicate_reason}",
                    rejection_category="duplicate",
                    chunk_id=validated_question.chunk_id,
                    chunk_ids=validated_question.chunk_ids,
                    page_number=validated_question.page_number,
                    page_numbers=validated_question.page_numbers,
                    question_type=validated_question.question_type,
                )
            )

        validated_questions = deduplicated_validated
        matching_filter_questions = [
            question for question in matching_filter_questions
            if id(question) in retained_question_objects
        ]

        # Prioritize matching filter questions, falling back to all validated questions if needed
        final_questions = matching_filter_questions[:count]
        if len(final_questions) < count:
            remaining_needed = count - len(final_questions)
            already_selected_ids = {q.question_id for q in final_questions}
            for v_q in validated_questions:
                if v_q.question_id not in already_selected_ids:
                    final_questions.append(v_q)
                    if len(final_questions) >= count:
                        break
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return QuestionGenerationResponse(
            document_id=document_id,
            requested_count=count,
            generated_candidates=len(raw_candidates),
            validated_count=validated_count,
            returned_count=len(final_questions),
            questions=final_questions,
            rejected_candidates=rejected_candidates if include_rejected else [],
            rejection_summary=dict(rejection_summary),
            latency_ms=round(elapsed_ms, 2),
        )

    def _get_document_chunks(
        self,
        document_id: str,
        chapter: Optional[int] = None,
        page_number: Optional[int] = None,
    ) -> List[Chunk]:
        """Fetch chunks for a specific document from PostgreSQL or active vector index mapping."""
        # 1. Try PostgreSQL DocumentPersistenceService first if available
        temp_session = None
        persistence = self._persistence

        if persistence is None or not getattr(persistence.session, "is_active", False):
            try:
                from app.db.session import get_session_factory
                temp_session = get_session_factory()()
                persistence = DocumentPersistenceService(temp_session)
            except Exception as exc:
                logger.debug("Could not initialize DB session for question generation persistence: %s", exc)

        if persistence is not None:
            try:
                db_chunks = persistence.get_document_chunks(document_id)
                if db_chunks:
                    matching_chunks: List[Chunk] = []
                    for c in db_chunks:
                        if page_number is not None and c.page_number != page_number:
                            continue
                        matching_chunks.append(
                            Chunk(
                                chunk_id=c.chunk_id,
                                document_id=c.document_id,
                                page_number=c.page_number,
                                chunk_index=c.chunk_index,
                                text=c.text,
                                start_char=0,
                                end_char=len(c.text),
                                char_count=c.char_count,
                                word_count=c.word_count,
                            )
                        )
                    if matching_chunks:
                        return matching_chunks
            except Exception as exc:
                logger.debug("Could not fetch document '%s' chunks from PostgreSQL: %s", document_id, exc)
            finally:
                if temp_session is not None:
                    try:
                        temp_session.close()
                    except Exception:
                        pass

        # 2. Fallback to in-memory vector indices
        target_index = None
        for idx in self.retrieval_service._indices.values():
            if document_id in idx.metadata.document_ids:
                target_index = idx
                break

        if target_index is None and self.retrieval_service._default_index_id:
            default_idx = self.retrieval_service.get_index()
            if document_id in default_idx.metadata.document_ids:
                target_index = default_idx

        if target_index is None:
            raise DocumentNotFoundError(f"Document '{document_id}' not found in database or any active index.")

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

