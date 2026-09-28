"""Semantic embedding service for BookRAG AI.

Transforms retrieval-ready Chunk objects into normalized dense vector EmbeddingRecords
while preserving complete document and page provenance.
"""

from typing import Any, List, Optional
import numpy as np

from app.core.logging import get_logger
from app.schemas.chunk import Chunk
from app.schemas.embedding import EmbeddingConfig, EmbeddingRecord
from app.services.embeddings.exceptions import InvalidChunkError
from app.services.embeddings.model import EmbeddingModel


logger = get_logger(__name__)


class EmbeddingService:
    """Orchestrates validation, batch inference, normalization, and provenance preservation."""

    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        default_config: Optional[EmbeddingConfig] = None,
    ) -> None:
        self.model = model
        self.default_config = default_config or EmbeddingConfig()

    def _validate_chunk(self, chunk: Any, index: int) -> None:
        """Ensure chunk is non-null, valid, and contains non-empty text content."""
        if chunk is None:
            raise InvalidChunkError(f"Chunk at index {index} is None.")

        if not hasattr(chunk, "chunk_id") or not hasattr(chunk, "text"):
            raise InvalidChunkError(f"Item at index {index} is not a valid Chunk instance.")

        if not chunk.text or not chunk.text.strip():
            chunk_identifier = getattr(chunk, "chunk_id", f"index_{index}")
            raise InvalidChunkError(
                f"Chunk '{chunk_identifier}' at index {index} contains empty or whitespace-only text."
            )

    def embed_chunk(
        self,
        chunk: Chunk,
        config: Optional[EmbeddingConfig] = None,
    ) -> EmbeddingRecord:
        """Generate an embedding for a single text chunk with provenance preservation.

        Args:
            chunk: Input Chunk from Phase 2.
            config: Optional configuration override.

        Returns:
            EmbeddingRecord containing 384-dimensional vector and metadata.
        """
        records = self.embed_chunks([chunk], config=config)
        return records[0]

    def embed_query(
        self,
        query: str,
        config: Optional[EmbeddingConfig] = None,
    ) -> np.ndarray:
        """Encode a user query string into a normalized dense vector for search.

        Args:
            query: Non-empty search query string.
            config: Optional configuration override.

        Returns:
            np.ndarray of shape (1, dimension) and dtype float32.
        """
        if query is None or not isinstance(query, str) or not query.strip():
            raise InvalidChunkError("Query text cannot be empty or whitespace-only.")

        cfg = config or self.default_config
        model = self.model or EmbeddingModel.get_instance(
            model_name=cfg.model_name,
            device=cfg.device,
        )
        return model.encode(
            texts=[query],
            batch_size=1,
            normalize_embeddings=cfg.normalize_embeddings,
        )

    def embed_chunks(
        self,
        chunks: List[Chunk],
        config: Optional[EmbeddingConfig] = None,
    ) -> List[EmbeddingRecord]:
        """Generate normalized vector embeddings for a collection of chunks in batches.

        Guarantees:
        1. Input validation: Rejects None, malformed, or empty-text chunks.
        2. Order preservation: Output list strictly matches input chunk order.
        3. Batch processing: Executes model inference respecting batch_size.
        4. Normalization: Normalizes vectors to L2 unit length when configured.
        5. Immutability: Input Chunk objects are never modified.
        6. Provenance: chunk_id, document_id, page_number are attached to each record.

        Args:
            chunks: List of Chunk objects to embed.
            config: Optional configuration override.

        Returns:
            List of EmbeddingRecord objects matching input sequence.
        """
        if chunks is None:
            raise InvalidChunkError("Chunks collection cannot be None.")

        if not chunks:
            return []

        cfg = config or self.default_config

        # 1. Validate all chunks prior to inference
        for idx, chunk in enumerate(chunks):
            self._validate_chunk(chunk, idx)

        # 2. Resolve embedding model
        model = self.model or EmbeddingModel.get_instance(
            model_name=cfg.model_name,
            device=cfg.device,
        )

        texts = [chunk.text for chunk in chunks]
        logger.info(
            "Embedding %d chunks (model: '%s', device: '%s', batch_size: %d, normalize: %s)...",
            len(texts),
            model.model_name,
            model.device,
            cfg.batch_size,
            cfg.normalize_embeddings,
        )

        # 3. Batch encode
        vectors = model.encode(
            texts=texts,
            batch_size=cfg.batch_size,
            normalize_embeddings=cfg.normalize_embeddings,
        )

        # 4. Assemble EmbeddingRecords preserving complete provenance
        records: List[EmbeddingRecord] = []
        for chunk, vector in zip(chunks, vectors):
            record = EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                text=chunk.text,
                embedding=vector.tolist(),
                dimension=len(vector),
                model_name=model.model_name,
                device=model.device,
                normalized=cfg.normalize_embeddings,
            )
            records.append(record)

        logger.info(
            "Successfully embedded %d chunks (dimension: %d, normalized: %s).",
            len(records),
            model.dimension,
            cfg.normalize_embeddings,
        )
        return records
