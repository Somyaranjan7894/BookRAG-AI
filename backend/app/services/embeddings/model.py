"""SentenceTransformer model wrapper and caching registry for BookRAG AI.

Provides single-load model management, automatic CPU/CUDA device negotiation,
and batch-encoded dense representations.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.logging import get_logger
from app.services.embeddings.exceptions import ModelLoadError

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


from app.core.device import resolve_device


class EmbeddingModel:
    """Encapsulates a loaded SentenceTransformer model instance with lifecycle caching."""

    _instances: Dict[str, "EmbeddingModel"] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "auto") -> None:
        self.model_name = model_name
        self.target_device = resolve_device(device)
        self._model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the underlying SentenceTransformer model onto the resolved device."""
        try:
            logger.info("Loading embedding model '%s' onto device '%s'...", self.model_name, self.target_device)
            self._model = SentenceTransformer(self.model_name, device=self.target_device)
            # Verify output dimension
            get_dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension"
            )
            dim = get_dim_fn()
            if dim != EXPECTED_DIMENSION and self.model_name == DEFAULT_MODEL_NAME:

                logger.warning(
                    "Model '%s' produced unexpected dimension %d (expected %d).",
                    self.model_name,
                    dim,
                    EXPECTED_DIMENSION,
                )
            logger.info("Embedding model '%s' loaded successfully (dimension: %d, device: '%s').", self.model_name, dim, self.target_device)
        except Exception as exc:
            logger.exception("Failed to load embedding model '%s': %s", self.model_name, exc)
            raise ModelLoadError(
                f"Could not load embedding model '{self.model_name}' on device '{self.target_device}'.",
                details=str(exc),
            ) from exc

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME, device: str = "auto") -> "EmbeddingModel":
        """Retrieve a cached model instance or instantiate a new one if not loaded."""
        resolved_dev = resolve_device(device)
        cache_key = f"{model_name}::{resolved_dev}"
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(model_name=model_name, device=resolved_dev)
        return cls._instances[cache_key]

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        if self._model is None:
            return EXPECTED_DIMENSION
        get_dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        dim = get_dim_fn()
        return int(dim) if dim is not None else EXPECTED_DIMENSION


    @property
    def device(self) -> str:
        """Return the active execution device."""
        return self.target_device

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode a batch of text strings into numpy embedding vectors.

        Args:
            texts: List of text passages to embed.
            batch_size: Number of texts per forward pass.
            normalize_embeddings: Whether to normalize vectors to L2 unit length.

        Returns:
            np.ndarray of shape (len(texts), dimension) and dtype float32.
        """
        if self._model is None:
            raise ModelLoadError("SentenceTransformer model is not loaded.")

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
