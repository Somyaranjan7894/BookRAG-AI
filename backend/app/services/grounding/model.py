"""NLI CrossEncoder model wrapper and lifecycle caching for BookRAG AI Phase 9.

Encapsulates sentence_transformers.CrossEncoder for natural language inference (NLI)
using cross-encoder/nli-deberta-v3-base with dynamic id2label discovery, process-level
singleton caching, configurable device negotiation, and deterministic inference.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.model import resolve_device
from app.services.grounding.exceptions import (
    NLIInferenceError,
    NLILabelMappingError,
    NLIModelLoadError,
)

logger = get_logger(__name__)

DEFAULT_NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-base"


@dataclass(frozen=True)
class NLIScores:
    """Normalized probability distribution over NLI classes for a premise-hypothesis pair."""

    entailment: float
    contradiction: float
    neutral: float


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax computation."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class NLIModel:
    """Encapsulates a loaded CrossEncoder NLI model with singleton caching and dynamic label mapping."""

    _instances: Dict[str, "NLIModel"] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL_NAME,
        device: str = "auto",
        max_length: Optional[int] = 512,
    ) -> None:
        """Initialize and load the CrossEncoder NLI model on the specified device.

        Args:
            model_name: Hugging Face model repository identifier.
            device: 'auto', 'cpu', or 'cuda' target execution device.
            max_length: Maximum sequence length for tokenization.
        """
        self.model_name = model_name
        self.target_device = resolve_device(device)
        self.max_length = max_length
        self._model: Optional[CrossEncoder] = None

        self.entailment_idx: Optional[int] = None
        self.contradiction_idx: Optional[int] = None
        self.neutral_idx: Optional[int] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the CrossEncoder weights and discover class label indices from configuration."""
        try:
            logger.info(
                "Loading CrossEncoder NLI model '%s' onto device '%s'...",
                self.model_name,
                self.target_device,
            )
            self._model = CrossEncoder(
                self.model_name,
                device=self.target_device,
                max_length=self.max_length,
            )
            logger.info(
                "CrossEncoder NLI model '%s' loaded successfully onto '%s'.",
                self.model_name,
                self.target_device,
            )
        except Exception as exc:
            logger.exception("Failed to load NLI model '%s': %s", self.model_name, exc)
            raise NLIModelLoadError(
                f"Could not load NLI model '{self.model_name}' on device '{self.target_device}'.",
                details=str(exc),
            ) from exc

        self._discover_label_indices()

    def _discover_label_indices(self) -> None:
        """Inspect model id2label configuration and map entailment, contradiction, neutral indices."""
        config_obj: Any = None
        # SentenceTransformers CrossEncoder stores the underlying HF transformer in .model
        if hasattr(self._model, "model") and hasattr(self._model.model, "config"):
            config_obj = self._model.model.config
        elif hasattr(self._model, "config"):
            config_obj = self._model.config

        if config_obj is None or not hasattr(config_obj, "id2label"):
            raise NLILabelMappingError(
                f"Model '{self.model_name}' has no identifiable id2label configuration for NLI classification."
            )

        id2label = config_obj.id2label
        if not id2label or not isinstance(id2label, dict):
            raise NLILabelMappingError(
                f"Model '{self.model_name}' id2label is empty or not a dictionary: {id2label}"
            )

        ent_idx: Optional[int] = None
        contra_idx: Optional[int] = None
        neut_idx: Optional[int] = None

        for raw_idx, label_name in id2label.items():
            idx = int(raw_idx)
            lbl = str(label_name).strip().lower()
            if "entail" in lbl:
                if ent_idx is not None:
                    raise NLILabelMappingError(
                        f"Ambiguous entailment label mapping in {id2label}: multiple candidates."
                    )
                ent_idx = idx
            elif "contra" in lbl:
                if contra_idx is not None:
                    raise NLILabelMappingError(
                        f"Ambiguous contradiction label mapping in {id2label}: multiple candidates."
                    )
                contra_idx = idx
            elif "neut" in lbl:
                if neut_idx is not None:
                    raise NLILabelMappingError(
                        f"Ambiguous neutral label mapping in {id2label}: multiple candidates."
                    )
                neut_idx = idx

        if ent_idx is None or contra_idx is None or neut_idx is None:
            raise NLILabelMappingError(
                f"Could not safely resolve all 3 NLI labels (entailment, contradiction, neutral) "
                f"from model config id2label: {id2label} (resolved: entailment={ent_idx}, "
                f"contradiction={contra_idx}, neutral={neut_idx})"
            )

        # Ensure all mapped indices are distinct
        if len({ent_idx, contra_idx, neut_idx}) != 3:
            raise NLILabelMappingError(
                f"Discovered duplicate label indices from id2label: {id2label}"
            )

        self.entailment_idx = ent_idx
        self.contradiction_idx = contra_idx
        self.neutral_idx = neut_idx

        logger.info(
            "NLI label indices dynamically resolved: entailment=%d, contradiction=%d, neutral=%d",
            self.entailment_idx,
            self.contradiction_idx,
            self.neutral_idx,
        )

    @classmethod
    def get_instance(
        cls,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = 512,
    ) -> "NLIModel":
        """Retrieve a cached model instance or instantiate a new one if not loaded."""
        resolved_name = model_name or settings.GROUNDING_MODEL_NAME
        resolved_dev = resolve_device(device or settings.GROUNDING_DEVICE)
        cache_key = f"{resolved_name}::{resolved_dev}::{max_length}"

        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(
                model_name=resolved_name,
                device=resolved_dev,
                max_length=max_length,
            )
        return cls._instances[cache_key]

    @property
    def model(self) -> CrossEncoder:
        """Return the underlying CrossEncoder instance."""
        if self._model is None:
            raise NLIModelLoadError("CrossEncoder NLI model is not initialized.")
        return self._model

    def predict(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[NLIScores]:
        """Perform batched NLI inference over (premise, hypothesis) string pairs.

        Args:
            pairs: Sequence of (premise, hypothesis) tuples.
                   premise = evidence text, hypothesis = claim text.
            batch_size: Inference forward-pass batch size. Defaults to INFERENCE_BATCH_SIZE.

        Returns:
            List of NLIScores with normalized entailment, contradiction, and neutral probabilities.
        """
        if not pairs:
            return []

        if self._model is None:
            raise NLIModelLoadError("CrossEncoder NLI model is not initialized.")

        try:
            eff_batch_size = batch_size if batch_size is not None else getattr(settings, "INFERENCE_BATCH_SIZE", 8)
            pair_list = [list(p) for p in pairs]
            from app.core.device import inference_context

            with inference_context(self.target_device):
                # Note: apply_softmax=True applies softmax inside CrossEncoder if supported
                raw_outputs = self._model.predict(
                    sentences=pair_list,
                    batch_size=eff_batch_size,
                    apply_softmax=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

            outputs_arr = np.asarray(raw_outputs, dtype=np.float32)

            # Handle 1D single pair output reshape to (1, num_classes)
            if outputs_arr.ndim == 1:
                outputs_arr = np.expand_dims(outputs_arr, axis=0)

            results: List[NLIScores] = []
            for row in outputs_arr:
                # Ensure probabilities sum to 1.0 (in case logits were returned)
                row_sum = np.sum(row)
                if abs(row_sum - 1.0) > 1e-2 or np.any(row < 0.0):
                    probs = softmax(row)
                else:
                    probs = row

                ent = float(np.clip(probs[self.entailment_idx], 0.0, 1.0))
                contra = float(np.clip(probs[self.contradiction_idx], 0.0, 1.0))
                neut = float(np.clip(probs[self.neutral_idx], 0.0, 1.0))

                results.append(
                    NLIScores(
                        entailment=ent,
                        contradiction=contra,
                        neutral=neut,
                    )
                )

            return results

        except Exception as exc:
            logger.exception("NLI prediction failed: %s", exc)
            raise NLIInferenceError(
                f"NLI model inference failed on {len(pairs)} pairs: {exc}",
                details=str(exc),
            ) from exc
