"""CrossEncoder model wrapper and caching registry for BookRAG AI Phase 6.

Provides single-load model lifecycle caching, automatic CPU/CUDA device negotiation,
and batched forward-pass scoring using Sentence Transformers CrossEncoder.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings.model import resolve_device
from app.services.reranking.exceptions import RerankerModelLoadError

logger = get_logger(__name__)

DEFAULT_RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_MAX_LENGTH: int = 512


class RerankerModel:
    """Encapsulates a loaded CrossEncoder model instance with lifecycle caching."""

    _instances: Dict[str, "RerankerModel"] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        device: str = "auto",
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
    ) -> None:
        """Initialize and load the CrossEncoder model.

        Args:
            model_name: HuggingFace / SentenceTransformers model identifier.
            device: 'auto', 'cpu', or 'cuda' target device.
            max_length: Maximum combined sequence length for (query, passage) pairs.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.target_device = resolve_device(device)
        self._model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the underlying CrossEncoder model onto the resolved device."""
        try:
            logger.info(
                "Loading cross-encoder model '%s' (max_length=%d) onto device '%s'...",
                self.model_name,
                self.max_length,
                self.target_device,
            )
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.target_device,
            )
            logger.info(
                "Cross-encoder model '%s' loaded successfully onto '%s'.",
                self.model_name,
                self.target_device,
            )
        except Exception as exc:
            logger.exception("Failed to load cross-encoder model '%s': %s", self.model_name, exc)
            raise RerankerModelLoadError(
                f"Could not load cross-encoder model '{self.model_name}' on device '{self.target_device}'.",
                details=str(exc),
            ) from exc

    @classmethod
    def get_instance(
        cls,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> "RerankerModel":
        """Retrieve a cached model instance or instantiate a new one if not loaded."""
        resolved_name = model_name or settings.RERANKER_MODEL_NAME
        resolved_dev = resolve_device(device or settings.RERANKER_DEVICE)
        resolved_len = max_length or settings.RERANKER_MAX_LENGTH
        cache_key = f"{resolved_name}::{resolved_dev}::{resolved_len}"

        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(
                model_name=resolved_name,
                device=resolved_dev,
                max_length=resolved_len,
            )
        return cls._instances[cache_key]

    @property
    def device(self) -> str:
        """Return the active execution device."""
        return self.target_device

    def predict(
        self,
        pairs: Sequence[Sequence[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Score candidate query-passage pairs via the CrossEncoder transformer.

        Args:
            pairs: List of [query, passage] string pairs.
            batch_size: Number of pairs per forward inference pass.

        Returns:
            1D np.ndarray of continuous float32 cross-encoder relevance scores.
        """
        if self._model is None:
            raise RerankerModelLoadError("CrossEncoder model is not loaded.")

        if not pairs:
            return np.empty((0,), dtype=np.float32)

        with torch.inference_mode():
            scores = self._model.predict(
                sentences=pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        scores_arr = np.asarray(scores, dtype=np.float32)
        # Ensure 1D shape (single score per pair)
        if scores_arr.ndim > 1:
            scores_arr = scores_arr.flatten()
        return scores_arr
