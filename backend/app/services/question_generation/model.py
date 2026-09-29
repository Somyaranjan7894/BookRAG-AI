"""Question generation model wrapper and lifecycle caching for BookRAG AI Phase 12.

Encapsulates Hugging Face AutoTokenizer and AutoModelForSeq2SeqLM
(iarfmoose/t5-base-question-generator) with singleton caching, device negotiation,
and deterministic inference under torch.inference_mode().
"""

import re
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.model import resolve_device
from app.services.question_generation.exceptions import (
    QuestionInferenceError,
    QuestionModelLoadError,
)

logger = get_logger(__name__)

DEFAULT_QUESTION_GEN_MODEL_NAME: str = "iarfmoose/t5-base-question-generator"


class QuestionGenerationModel:
    """Encapsulates a loaded Seq2Seq question generation model with singleton caching."""

    _instances: Dict[str, "QuestionGenerationModel"] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_QUESTION_GEN_MODEL_NAME,
        device: str = "auto",
    ) -> None:
        """Initialize and load the Seq2Seq question generation model.

        Args:
            model_name: Hugging Face model repository identifier.
            device: 'auto', 'cpu', or 'cuda' target device.
        """
        self.model_name = model_name
        self.target_device = resolve_device(device)
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._model: Optional[PreTrainedModel] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model weights onto the resolved target device."""
        try:
            logger.info(
                "Loading question generation model '%s' onto device '%s'...",
                self.model_name,
                self.target_device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.target_device)
            self._model.eval()
            logger.info(
                "Question generation model '%s' loaded successfully onto '%s'.",
                self.model_name,
                self.target_device,
            )
        except Exception as exc:
            logger.exception("Failed to load question generation model '%s': %s", self.model_name, exc)
            raise QuestionModelLoadError(
                f"Could not load question generation model '{self.model_name}' on device '{self.target_device}'.",
                details=str(exc),
            ) from exc

    @classmethod
    def get_instance(
        cls,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "QuestionGenerationModel":
        """Retrieve a cached model instance or instantiate a new one."""
        resolved_name = model_name or settings.QUESTION_GEN_MODEL_NAME
        resolved_dev = resolve_device(device or settings.QUESTION_GEN_DEVICE)
        cache_key = f"{resolved_name}::{resolved_dev}"

        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(
                model_name=resolved_name,
                device=resolved_dev,
            )
        return cls._instances[cache_key]

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Return the loaded tokenizer."""
        if self._tokenizer is None:
            raise QuestionModelLoadError("Tokenizer is not initialized.")
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """Return the underlying generation model."""
        if self._model is None:
            raise QuestionModelLoadError("Model is not initialized.")
        return self._model

    def generate_question(
        self,
        answer: str,
        context: str,
        max_input_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> str:
        """Generate a single question conditioned on answer and context."""
        results = self.generate_questions_batch(
            pairs=[(answer, context)],
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return results[0] if results else ""

    def generate_questions_batch(
        self,
        pairs: List[tuple[str, str]],
        max_input_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> List[str]:
        """Generate questions for a batch of (answer, context) pairs.

        Args:
            pairs: List of (answer, context) string tuples.
            max_input_length: Maximum input sequence length (defaults to settings).
            max_new_tokens: Maximum generated tokens (defaults to settings).
            num_beams: Number of beams for deterministic beam search.

        Returns:
            List of generated question strings.
        """
        if not pairs:
            return []

        resolved_max_input = max_input_length or settings.QUESTION_GEN_MAX_INPUT_LENGTH
        resolved_max_new = max_new_tokens or settings.QUESTION_GEN_MAX_NEW_TOKENS
        resolved_num_beams = num_beams or settings.QUESTION_GEN_NUM_BEAMS

        # Construct input sequences: "<answer> {answer} <context> {context}"
        input_texts = [f"<answer> {ans} <context> {ctx}" for ans, ctx in pairs]

        try:
            with torch.inference_mode():
                encoded_inputs = self.tokenizer(
                    input_texts,
                    padding=True,
                    truncation=True,
                    max_length=resolved_max_input,
                    return_tensors="pt",
                )
                input_ids = encoded_inputs["input_ids"].to(self.target_device)
                attention_mask = encoded_inputs["attention_mask"].to(self.target_device)

                output_tokens = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=resolved_max_new,
                    num_beams=resolved_num_beams,
                    do_sample=False,
                    early_stopping=True,
                )

                decoded = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
                return [self._clean_question(q) for q in decoded]

        except Exception as exc:
            logger.exception("Question generation batch inference failed: %s", exc)
            raise QuestionInferenceError(
                f"Failed to generate questions for batch of {len(pairs)} items: {exc}",
                details=str(exc),
            ) from exc

    @staticmethod
    def _clean_question(raw_text: str) -> str:
        """Clean and normalize generated question string."""
        text = raw_text.strip()
        # Collapse repeated spaces
        text = re.sub(r"\s+", " ", text)
        # Fix duplicated punctuation at the end: e.g. "? ?" or "??" -> "?"
        text = re.sub(r"[\s\?]+$", "?", text)
        if text and not text.endswith("?"):
            text += "?"
        return text
