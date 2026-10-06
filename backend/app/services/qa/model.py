"""Extractive Question Answering model wrapper and caching registry for BookRAG AI Phase 7.

Wraps Hugging Face Transformers AutoTokenizer and AutoModelForQuestionAnswering (deepset/roberta-base-squad2),
providing process-level singleton lifecycle caching, automatic device resolution, and
thread-safe inference under torch.inference_mode().
"""

from typing import Dict, Optional, Tuple
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.model import resolve_device
from app.services.qa.exceptions import QAInferenceError, QAModelLoadError

logger = get_logger(__name__)

DEFAULT_QA_MODEL_NAME: str = "deepset/roberta-base-squad2"


class QAModel:
    """Encapsulates a loaded Hugging Face QA model and tokenizer with singleton caching."""

    _instances: Dict[str, "QAModel"] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_QA_MODEL_NAME,
        device: str = "auto",
    ) -> None:
        """Initialize and load the Hugging Face QA model and tokenizer.

        Args:
            model_name: HuggingFace model repository identifier.
            device: 'auto', 'cpu', or 'cuda' target device.
        """
        self.model_name = model_name
        self.target_device = resolve_device(device)
        self._tokenizer: Optional[PreTrainedTokenizerFast] = None
        self._model: Optional[PreTrainedModel] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model weights onto the resolved device."""
        try:
            logger.info(
                "Loading extractive QA model '%s' onto device '%s'...",
                self.model_name,
                self.target_device,
            )
            # Load fast tokenizer for offset mapping and overflow mapping support
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )
            self._model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name,
            )
            self._model.to(self.target_device)
            self._model.eval()
            logger.info(
                "Extractive QA model '%s' loaded successfully on '%s'.",
                self.model_name,
                self.target_device,
            )
        except Exception as exc:
            logger.exception("Failed to load extractive QA model '%s': %s", self.model_name, exc)
            raise QAModelLoadError(
                f"Could not load extractive QA model '{self.model_name}' on device '{self.target_device}'.",
                details=str(exc),
            ) from exc

    @classmethod
    def get_instance(
        cls,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "QAModel":
        """Retrieve a cached model instance or instantiate a new one if not loaded."""
        resolved_name = model_name or settings.QA_MODEL_NAME
        resolved_dev = resolve_device(device or settings.QA_DEVICE)
        cache_key = f"{resolved_name}::{resolved_dev}"

        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(
                model_name=resolved_name,
                device=resolved_dev,
            )
        return cls._instances[cache_key]

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """Return the loaded Hugging Face tokenizer."""
        if self._tokenizer is None:
            raise QAModelLoadError("QA tokenizer is not initialized.")
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """Return the loaded Hugging Face QA model."""
        if self._model is None:
            raise QAModelLoadError("QA model is not initialized.")
        return self._model

    @property
    def device(self) -> str:
        """Return the active execution device."""
        return self.target_device

    def predict_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward inference pass returning start and end logits.

        Args:
            input_ids: Token ID tensor of shape (batch_size, sequence_length).
            attention_mask: Attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            Tuple of (start_logits, end_logits) tensors moved to CPU.
        """
        if self._model is None:
            raise QAModelLoadError("QA model is not loaded.")

        from app.core.device import inference_context

        try:
            with inference_context(self.target_device):
                device_input_ids = input_ids.to(self.target_device)
                device_attention_mask = attention_mask.to(self.target_device)

                outputs = self._model(
                    input_ids=device_input_ids,
                    attention_mask=device_attention_mask,
                )
                start_logits = outputs.start_logits.detach().cpu()
                end_logits = outputs.end_logits.detach().cpu()
                return start_logits, end_logits
        except Exception as exc:
            logger.exception("Forward pass failed during QA inference: %s", exc)
            raise QAInferenceError(
                "Failed to execute extractive QA forward inference pass.",
                details=str(exc),
            ) from exc
