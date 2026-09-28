"""FAISS Vector Index wrapper and persistence manager for BookRAG AI.

Implements exact cosine-similarity vector retrieval using faiss.IndexFlatIP
over L2-normalized embeddings while maintaining synchronized chunk provenance mapping.
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import faiss
import numpy as np

from app.core.logging import get_logger
from app.schemas.embedding import EmbeddingRecord
from app.schemas.retrieval import IndexMetadata, RetrievalResult, VectorMappingItem
from app.services.retrieval.exceptions import (
    CorruptedIndexError,
    IndexDimensionMismatchError,
    IndexNotFoundError,
    IndexPersistenceError,
)
from app.services.retrieval.mapping import VectorToChunkMapping

logger = get_logger(__name__)

DEFAULT_DIMENSION = 384
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorIndex:
    """Encapsulates a FAISS IndexFlatIP instance, index metadata, and vector-to-chunk mappings."""

    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        model_name: str = DEFAULT_MODEL_NAME,
        index_id: str = "default",
        normalized: bool = True,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"Index dimension must be strictly positive (got {dimension}).")

        self.dimension = dimension
        self.model_name = model_name
        self.index_id = index_id
        self.normalized = normalized

        # Initialize exact inner product FAISS index (equivalent to cosine similarity on normalized vectors)
        self._index = faiss.IndexFlatIP(self.dimension)
        self._mapping = VectorToChunkMapping()
        self._document_ids: set[str] = set()

    @property
    def total_vectors(self) -> int:
        """Return the total number of vectors indexed in FAISS."""
        return self._index.ntotal

    @property
    def metadata(self) -> IndexMetadata:
        """Construct structured metadata representation for this index."""
        return IndexMetadata(
            index_id=self.index_id,
            model_name=self.model_name,
            dimension=self.dimension,
            total_vectors=self.total_vectors,
            document_ids=sorted(list(self._document_ids)),
            normalized=self.normalized,
        )

    def add_records(self, records: List[EmbeddingRecord]) -> int:
        """Add a batch of EmbeddingRecords into the FAISS index and update provenance mapping.

        Args:
            records: List of EmbeddingRecord objects produced in Phase 3.

        Returns:
            Number of newly added vectors.

        Raises:
            IndexDimensionMismatchError: If any record's vector dimension differs from the index dimension.
        """
        if not records:
            return 0

        # 1. Validate vector dimensions
        for r in records:
            if r.dimension != self.dimension or len(r.embedding) != self.dimension:
                raise IndexDimensionMismatchError(
                    f"Record chunk '{r.chunk_id}' dimension ({r.dimension}) "
                    f"does not match index dimension ({self.dimension})."
                )

        # 2. Extract vectors and convert to contiguous float32 numpy array
        vectors = np.array([r.embedding for r in records], dtype=np.float32)
        vectors = np.ascontiguousarray(vectors)

        start_slot = self.total_vectors

        # 3. Add to FAISS index
        self._index.add(vectors)

        # 4. Update mapping and document tracking
        for idx, r in enumerate(records):
            slot = start_slot + idx
            mapping_item = VectorMappingItem(
                vector_index=slot,
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                page_number=r.page_number,
                chunk_index=getattr(r, "chunk_index", idx),
                text=r.text,
                metadata={
                    "model_name": r.model_name,
                    "device": r.device,
                    "normalized": r.normalized,
                },
            )
            self._mapping.add_item(mapping_item)
            self._document_ids.add(r.document_id)

        logger.info(
            "Added %d vectors to index '%s' (total vectors now: %d).",
            len(records),
            self.index_id,
            self.total_vectors,
        )
        return len(records)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Perform similarity search for nearest vectors to query_vector.

        Args:
            query_vector: 1D or 2D numpy array containing the query embedding.
            top_k: Maximum number of top results to return.
            document_id: Optional document ID to isolate retrieval to a specific document.

        Returns:
            Ranked list of RetrievalResult objects (rank 1 = highest similarity).

        Raises:
            IndexDimensionMismatchError: If query vector dimension does not match index dimension.
        """
        if top_k <= 0:
            return []

        # Validate and shape query vector
        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        elif q_vec.ndim != 2 or q_vec.shape[0] != 1:
            raise IndexDimensionMismatchError(
                f"Query vector must be 1D (dimension,) or 2D (1, dimension). Got shape {q_vec.shape}."
            )

        if q_vec.shape[1] != self.dimension:
            raise IndexDimensionMismatchError(
                f"Query vector dimension ({q_vec.shape[1]}) does not match index dimension ({self.dimension})."
            )

        if self.total_vectors == 0:
            logger.debug("Search called on empty index '%s'. Returning empty result list.", self.index_id)
            return []

        q_vec = np.ascontiguousarray(q_vec)


        # Determine retrieval search limit
        search_k = min(self.total_vectors, max(top_k * 10, top_k) if document_id else top_k)

        distances, indices = self._index.search(q_vec, search_k)

        results: List[RetrievalResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue

            mapping_item = self._mapping.get(int(idx))
            if mapping_item is None:
                continue

            # Document isolation / filtering
            if document_id is not None and mapping_item.document_id != document_id:
                continue

            results.append(
                RetrievalResult(
                    rank=len(results) + 1,
                    chunk_id=mapping_item.chunk_id,
                    document_id=mapping_item.document_id,
                    page_number=mapping_item.page_number,
                    chunk_index=mapping_item.chunk_index,
                    text=mapping_item.text,
                    similarity_score=float(dist),
                    metadata=mapping_item.metadata,
                )
            )

            if len(results) == top_k:
                break

        return results

    def save(self, directory: Union[str, Path], base_name: str = "index") -> Tuple[Path, Path]:
        """Persist the FAISS index binary and metadata mapping to disk.

        Args:
            directory: Target directory path where artifacts will be written.
            base_name: Prefix for persisted files (<base_name>.faiss and <base_name>.json).

        Returns:
            Tuple of (index_file_path, metadata_file_path).
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_path = dir_path / f"{base_name}.faiss"
        meta_path = dir_path / f"{base_name}.json"

        try:
            # 1. Write FAISS index binary
            faiss.write_index(self._index, str(index_path))

            # 2. Write metadata and mapping JSON
            payload = {
                "metadata": self.metadata.model_dump(),
                "mapping": self._mapping.to_dict(),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            logger.info("Persisted index '%s' to '%s' and '%s'.", self.index_id, index_path, meta_path)
            return index_path, meta_path
        except Exception as exc:
            logger.exception("Failed to persist index '%s': %s", self.index_id, exc)
            raise IndexPersistenceError(f"Could not save index to '{directory}': {exc}") from exc

    @classmethod
    def load(cls, directory: Union[str, Path], base_name: str = "index") -> "VectorIndex":
        """Load a persisted FAISS index and reconstruct its vector-to-chunk mappings.

        Args:
            directory: Directory containing <base_name>.faiss and <base_name>.json.
            base_name: Prefix for persisted files.

        Returns:
            Reconstructed and validated VectorIndex instance.

        Raises:
            IndexNotFoundError: If index or metadata files do not exist.
            CorruptedIndexError: If metadata and FAISS index are inconsistent.
        """
        dir_path = Path(directory)
        index_path = dir_path / f"{base_name}.faiss"
        meta_path = dir_path / f"{base_name}.json"

        if not index_path.exists():
            raise IndexNotFoundError(f"FAISS index file not found at '{index_path}'.")

        if not meta_path.exists():
            raise IndexNotFoundError(f"Index metadata file not found at '{meta_path}'.")

        try:
            # 1. Read metadata and mappings
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if "metadata" not in payload or "mapping" not in payload:
                raise CorruptedIndexError(f"Metadata file at '{meta_path}' is missing required sections.")

            meta = IndexMetadata(**payload["metadata"])
            mapping = VectorToChunkMapping.from_dict(payload["mapping"])

            # 2. Read FAISS index
            loaded_faiss_index = faiss.read_index(str(index_path))

            # 3. Validate consistency between FAISS index and metadata
            if loaded_faiss_index.d != meta.dimension:
                raise CorruptedIndexError(
                    f"FAISS index dimension ({loaded_faiss_index.d}) does not match metadata dimension ({meta.dimension})."
                )

            if loaded_faiss_index.ntotal != meta.total_vectors:
                raise CorruptedIndexError(
                    f"FAISS vector count ({loaded_faiss_index.ntotal}) does not match metadata count ({meta.total_vectors})."
                )

            if loaded_faiss_index.ntotal != len(mapping):
                raise CorruptedIndexError(
                    f"FAISS vector count ({loaded_faiss_index.ntotal}) does not match mapping count ({len(mapping)})."
                )

            mapping.validate_consistency(loaded_faiss_index.ntotal)

            # 4. Instantiate VectorIndex
            instance = cls(
                dimension=meta.dimension,
                model_name=meta.model_name,
                index_id=meta.index_id,
                normalized=meta.normalized,
            )
            instance._index = loaded_faiss_index
            instance._mapping = mapping
            instance._document_ids = set(meta.document_ids)

            logger.info(
                "Successfully loaded index '%s' from '%s' (%d vectors, %d dimensions).",
                meta.index_id,
                dir_path,
                instance.total_vectors,
                instance.dimension,
            )
            return instance

        except (IndexNotFoundError, CorruptedIndexError):
            raise
        except Exception as exc:
            logger.exception("Unexpected error loading index from '%s': %s", directory, exc)
            raise IndexPersistenceError(f"Failed to load index from '{directory}': {exc}") from exc
