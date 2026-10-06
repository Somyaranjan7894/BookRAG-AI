"""Abstractive Question Answering (Generation) Service for BookRAG AI Phase 8.

Orchestrates query validation, evidence context budgeting, grounded prompt formatting,
and deterministic Seq2Seq generation via FLAN-T5, ensuring answers are grounded
exclusively in retrieved book evidence with complete provenance.
"""

from typing import Any, List, Optional, Sequence
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.generation import GenerationEvidenceItem, GenerationResponse
from app.services.generation.evidence import BuiltPrompt, EvidenceBuilder
from app.services.generation.exceptions import (
    GenerationError,
    GenerationInferenceError,
    InvalidGenerationConfigError,
    InvalidGenerationQueryError,
)
from app.services.generation.model import GenerationModel

logger = get_logger(__name__)


class GenerationService:
    """Orchestrates abstractive question answering using retrieved book evidence and FLAN-T5."""

    def __init__(
        self,
        model: Optional[GenerationModel] = None,
        evidence_builder: Optional[EvidenceBuilder] = None,
        max_input_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Initialize GenerationService with configurable or injected components.

        Args:
            model: Optional GenerationModel instance (defaults to cached singleton).
            evidence_builder: Optional EvidenceBuilder instance.
            max_input_tokens: Maximum allowed prompt token budget.
            max_new_tokens: Maximum tokens to generate (default: 128).
            num_beams: Beam search width (default: 4).
            do_sample: Whether to sample (default: False for deterministic output).
            temperature: Sampling temperature if do_sample is True.
        """
        self._model = model
        self._evidence_builder = evidence_builder
        self.max_input_tokens = (
            max_input_tokens if max_input_tokens is not None else settings.GENERATION_MAX_INPUT_TOKENS
        )
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else settings.GENERATION_MAX_NEW_TOKENS
        )
        self.num_beams = (
            num_beams if num_beams is not None else settings.GENERATION_NUM_BEAMS
        )
        self.do_sample = (
            do_sample if do_sample is not None else settings.GENERATION_DO_SAMPLE
        )
        self.temperature = (
            temperature if temperature is not None else settings.GENERATION_TEMPERATURE
        )


        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Verify that generation configuration values are mathematically sound."""
        if self.max_input_tokens <= 0:
            raise InvalidGenerationConfigError(
                f"max_input_tokens must be positive (got {self.max_input_tokens})."
            )
        if self.max_new_tokens <= 0:
            raise InvalidGenerationConfigError(
                f"max_new_tokens must be positive (got {self.max_new_tokens})."
            )
        if self.num_beams <= 0:
            raise InvalidGenerationConfigError(
                f"num_beams must be positive (got {self.num_beams})."
            )
        if self.temperature <= 0.0:
            raise InvalidGenerationConfigError(
                f"temperature must be strictly positive (got {self.temperature})."
            )

    @property
    def model(self) -> GenerationModel:
        """Lazy load or return the injected GenerationModel instance."""
        if self._model is None:
            self._model = GenerationModel.get_instance()
        return self._model

    @property
    def evidence_builder(self) -> EvidenceBuilder:
        """Return the injected or lazily constructed EvidenceBuilder."""
        if self._evidence_builder is None:
            self._evidence_builder = EvidenceBuilder(
                tokenizer=self.model.tokenizer,
                max_input_tokens=self.max_input_tokens,
            )
        return self._evidence_builder

    def generate_answer(
        self,
        query: str,
        evidence: Sequence[Any],
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        prompt_instruction: Optional[str] = None,
        prompt_template: Optional[str] = None,
        query_type: Optional[Any] = None,
    ) -> GenerationResponse:
        """Synthesize an abstractive answer from retrieved evidence chunks using FLAN-T5.

        Execution Pipeline:
        1. Validate query string (reject empty or whitespace).
        2. Empty evidence handling: If no valid evidence is supplied, return unanswerable
           immediately without calling the model (preventing ungrounded hallucination).
        3. Assemble context and budget tokens using EvidenceBuilder.
        4. Check if any evidence could be accommodated within the token budget.
        5. Invoke FLAN-T5 forward pass with deterministic beam search.
        6. Clean output text (strip surrounding formatting/whitespace).
        7. Assemble and return GenerationResponse with complete evidence provenance.

        Args:
            query: Natural language question string.
            evidence: Sequence of SearchResult or dict objects representing retrieved chunks.
            max_new_tokens: Optional override for generated token limit.
            num_beams: Optional override for beam count.
            do_sample: Optional override for sampling flag.
            temperature: Optional override for temperature.
            prompt_instruction: Optional targeted prompt instruction (e.g. for controlled regeneration).
            prompt_template: Optional complete prompt template override.
            query_type: Optional classified query type to select specialized prompt templates.

        Returns:
            GenerationResponse containing generated text or structured no-answer.
        """
        # 1. Query validation
        if query is None or not isinstance(query, str) or not query.strip():
            raise InvalidGenerationQueryError("Question query cannot be empty or whitespace-only.")

        clean_query = query.strip()
        model_name = getattr(self.model, "model_name", settings.GENERATION_MODEL_NAME)

        # 2. Empty evidence handling (CRITICAL RULE: Never call FLAN-T5 with empty evidence)
        if not evidence:
            logger.info("No evidence chunks provided for query '%s'. Returning unanswerable.", clean_query)
            return GenerationResponse(
                query=clean_query,
                answer=None,
                answerable=False,
                model_name=model_name,
                evidence=[],
                evidence_count=0,
            )

        # 3. Assemble prompt and budget evidence context
        from app.services.generation.evidence import COMPARISON_PROMPT_TEMPLATE, MULTI_PART_PROMPT_TEMPLATE

        resolved_template = prompt_template
        if resolved_template is None and prompt_instruction is None and query_type is not None:
            qt_str = str(getattr(query_type, "value", query_type)).lower()
            if qt_str == "comparison":
                resolved_template = COMPARISON_PROMPT_TEMPLATE
            elif qt_str in ("multi_part", "multi_page"):
                resolved_template = MULTI_PART_PROMPT_TEMPLATE

        built_prompt: BuiltPrompt = self.evidence_builder.build_prompt(
            question=clean_query,
            evidence=evidence,
            prompt_template=resolved_template,
            instruction=prompt_instruction,
        )

        if not built_prompt.included_evidence:
            logger.info("No evidence could fit into prompt budget for query '%s'.", clean_query)
            return GenerationResponse(
                query=clean_query,
                answer=None,
                answerable=False,
                model_name=model_name,
                evidence=[],
                evidence_count=0,
            )

        # 4. Resolve generation hyperparameters
        eff_max_new = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        eff_beams = num_beams if num_beams is not None else self.num_beams
        eff_sample = do_sample if do_sample is not None else self.do_sample
        eff_temp = temperature if temperature is not None else self.temperature

        # 5. Invoke model forward pass
        try:
            raw_answer = self.model.generate(
                prompt=built_prompt.prompt,
                max_new_tokens=eff_max_new,
                num_beams=eff_beams,
                do_sample=eff_sample,
                temperature=eff_temp,
            )
        except GenerationError:
            raise
        except Exception as exc:
            logger.exception("FLAN-T5 generation forward pass failed: %s", exc)
            raise GenerationInferenceError(
                "Generation inference failed.",
                details=str(exc),
            ) from exc

        # 6. Clean generated answer
        clean_answer = raw_answer.strip()
        if not clean_answer:
            logger.info("FLAN-T5 returned empty output for query '%s'.", clean_query)
            return GenerationResponse(
                query=clean_query,
                answer=None,
                answerable=False,
                model_name=model_name,
                evidence=built_prompt.included_evidence,
                evidence_count=len(built_prompt.included_evidence),
            )

        logger.info(
            "Generated answer of %d chars using %d evidence chunks for query '%s'.",
            len(clean_answer),
            len(built_prompt.included_evidence),
            clean_query,
        )

        # 7. Return structured response with full provenance
        return GenerationResponse(
            query=clean_query,
            answer=clean_answer,
            answerable=True,
            model_name=model_name,
            evidence=built_prompt.included_evidence,
            evidence_count=len(built_prompt.included_evidence),
        )
