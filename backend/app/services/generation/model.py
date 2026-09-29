"""Abstractive generation model wrapper and lifecycle caching for BookRAG AI Phase 8.

Encapsulates Hugging Face Transformers AutoTokenizer and AutoModelForSeq2SeqLM
(google/flan-t5-base) with process-level singleton caching, device negotiation,
and deterministic inference under torch.inference_mode().
"""

from typing import Dict, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.model import resolve_device
from app.services.generation.exceptions import GenerationInferenceError, GenerationModelLoadError

logger = get_logger(__name__)

DEFAULT_GENERATION_MODEL_NAME: str = "google/flan-t5-base"


class GenerationModel:
    """Encapsulates a loaded Seq2Seq generation model and tokenizer with singleton caching."""

    _instances: Dict[str, "GenerationModel"] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATION_MODEL_NAME,
        device: str = "auto",
    ) -> None:
        """Initialize and load the Seq2Seq generation model and tokenizer.

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
        """Load tokenizer and weights onto the resolved device."""
        try:
            logger.info(
                "Loading Seq2Seq generation model '%s' onto device '%s'...",
                self.model_name,
                self.target_device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.target_device)
            self._model.eval()
            logger.info(
                "Seq2Seq model '%s' loaded successfully onto '%s'.",
                self.model_name,
                self.target_device,
            )
        except Exception as exc:
            logger.exception("Failed to load Seq2Seq model '%s': %s", self.model_name, exc)
            raise GenerationModelLoadError(
                f"Could not load generation model '{self.model_name}' on device '{self.target_device}'.",
                details=str(exc),
            ) from exc

    @classmethod
    def get_instance(
        cls,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "GenerationModel":
        """Retrieve a cached model instance or instantiate a new one if not loaded."""
        resolved_name = model_name or settings.GENERATION_MODEL_NAME
        resolved_dev = resolve_device(device or settings.GENERATION_DEVICE)
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
            raise GenerationModelLoadError("Tokenizer is not initialized.")
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """Return the underlying generation model."""
        if self._model is None:
            raise GenerationModelLoadError("Model is not initialized.")
        return self._model

    @property
    def device(self) -> str:
        """Return the active execution device."""
        return self.target_device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """Execute controlled abstractive text generation over the given prompt.

        Args:
            prompt: Grounded prompt string containing task instructions and evidence context.
            max_new_tokens: Maximum number of tokens to generate.
            num_beams: Number of beams for deterministic beam search.
            do_sample: Whether to use multinomial sampling (default False).
            temperature: Sampling temperature if do_sample is True.

        Returns:
            Clean decoded string response.
        """
        if self._model is None or self._tokenizer is None:
            raise GenerationModelLoadError("Generation model is not loaded.")

        try:
            with torch.inference_mode():
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self.target_device)

                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "num_beams": num_beams,
                    "do_sample": do_sample,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature

                outputs = self._model.generate(**inputs, **gen_kwargs)
                decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                return decoded.strip()
        except Exception as exc:
            logger.exception("Inference failed during FLAN-T5 generation: %s", exc)
            raise GenerationInferenceError(
                "Failed to execute abstractive generation.",
                details=str(exc),
            ) from exc
