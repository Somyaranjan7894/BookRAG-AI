"""High-level Vector Retrieval Service for BookRAG AI.

Coordinates query embedding generation, FAISS vector search, and structured
RetrievalResult assembly with complete provenance and document isolation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from app.core.logging import get_logger
from app.schemas.chunk import Chunk
from app.schemas.embedding import EmbeddingRecord
from app.schemas.retrieval import RetrievalResult
from app.services.embeddings.service import EmbeddingService
from app.services.retrieval.exceptions import (
    EmptyIndexError,
    IndexNotFoundError,
    InvalidQueryError,
    RetrievalError,
)
from app.services.retrieval.index import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL_NAME,
    VectorIndex,
)

logger = get_logger(__name__)


class RetrievalService:
    """Orchestrates query embedding, index management, and top-K vector retrieval."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        default_index: Optional[VectorIndex] = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self._indices: Dict[str, VectorIndex] = {}
        self._default_index_id: Optional[str] = None

        if default_index is not None:
            self.register_index(default_index, set_as_default=True)

    def register_index(self, index: VectorIndex, set_as_default: bool = False) -> None:
        """Register a VectorIndex instance into the service registry."""
        self._indices[index.index_id] = index
        if set_as_default or self._default_index_id is None:
            self._default_index_id = index.index_id
        logger.debug("Registered index '%s' in RetrievalService.", index.index_id)

    def get_index(self, index_id: Optional[str] = None) -> VectorIndex:
        """Retrieve an index by ID or the default active index."""
        target_id = index_id or self._default_index_id
        if not target_id or target_id not in self._indices:
            raise IndexNotFoundError(
                f"No vector index found for ID '{target_id}'. Registered indices: {list(self._indices.keys())}."
            )
        return self._indices[target_id]

    def create_index(
        self,
        index_id: str = "default",
        dimension: int = DEFAULT_DIMENSION,
        model_name: str = DEFAULT_MODEL_NAME,
        set_as_default: bool = True,
    ) -> VectorIndex:
        """Create, register, and return a new empty VectorIndex."""
        index = VectorIndex(
            dimension=dimension,
            model_name=model_name,
            index_id=index_id,
            normalized=True,
        )
        self.register_index(index, set_as_default=set_as_default)
        return index

    def build_index_from_records(
        self,
        records: List[EmbeddingRecord],
        index_id: Optional[str] = None,
        set_as_default: bool = True,
    ) -> VectorIndex:
        """Build and populate a VectorIndex directly from Phase 3 EmbeddingRecord objects.

        Args:
            records: List of Phase 3 EmbeddingRecords.
            index_id: Optional index identifier (defaults to document_id of records or 'default').
            set_as_default: Whether to register this index as the primary default index.

        Returns:
            Populated and registered VectorIndex.
        """
        if not records:
            resolved_id = index_id or "default"
            return self.create_index(index_id=resolved_id, set_as_default=set_as_default)

        resolved_id = index_id or records[0].document_id or "default"
        dim = records[0].dimension
        model_name = records[0].model_name

        index = VectorIndex(
            dimension=dim,
            model_name=model_name,
            index_id=resolved_id,
            normalized=records[0].normalized,
        )
        index.add_records(records)
        self.register_index(index, set_as_default=set_as_default)
        return index

    def build_index_from_chunks(
        self,
        chunks: List[Chunk],
        index_id: Optional[str] = None,
        set_as_default: bool = True,
    ) -> VectorIndex:
        """Generate embeddings for chunks and construct a searchable VectorIndex.

        Args:
            chunks: List of Phase 2 Chunk objects.
            index_id: Optional index identifier.
            set_as_default: Whether to set as default active index.

        Returns:
            Populated VectorIndex.
        """
        records = self.embedding_service.embed_chunks(chunks)
        return self.build_index_from_records(records, index_id=index_id, set_as_default=set_as_default)

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        document_id: Optional[str] = None,
        index_id: Optional[str] = None,
        index: Optional[VectorIndex] = None,
    ) -> List[RetrievalResult]:
        """Perform semantic vector retrieval directly using a precomputed query embedding vector.

        Args:
            query_vector: 1D or 2D numpy array containing the query embedding.
            top_k: Maximum number of nearest candidate chunks to retrieve (default: 5).
            document_id: Optional document ID to restrict retrieval.
            index_id: Optional specific index ID to query.
            index: Optional direct VectorIndex instance override.

        Returns:
            List of RetrievalResult objects ranked by similarity score.

        Raises:
            RetrievalError: If top_k <= 0.
            IndexNotFoundError: If no target index is available.
        """
        if top_k <= 0:
            raise RetrievalError(f"top_k must be strictly positive (got {top_k}).")

        # Resolve target index
        target_index: VectorIndex
        if index is not None:
            target_index = index
        elif document_id and document_id in self._indices:
            target_index = self._indices[document_id]
        else:
            target_index = self.get_index(index_id)

        # Empty index check
        if target_index.total_vectors == 0:
            logger.debug("Retrieval requested on empty index '%s'. Returning empty list.", target_index.index_id)
            return []

        # Perform search and map to RetrievalResults
        results = target_index.search(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
        )

        logger.info(
            "Vector search retrieved %d candidates from index '%s' (top_k=%d, doc_filter=%s).",
            len(results),
            target_index.index_id,
            top_k,
            document_id,
        )
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        index_id: Optional[str] = None,
        index: Optional[VectorIndex] = None,
    ) -> List[RetrievalResult]:
        """Perform semantic vector retrieval for a natural language query.

        Execution Pipeline:
        1. Validate query string and top_k parameters.
        2. Resolve target VectorIndex (per-document, specified index_id, or default).
        3. If index has no vectors, return empty result list.
        4. Generate 384-dimensional L2-normalized query vector via EmbeddingService.
        5. Execute FAISS IndexFlatIP inner product search.
        6. Map top-K vector positions back to Chunk metadata with provenance.
        7. Return structured, ranked RetrievalResult list.

        Args:
            query: Natural language search query string.
            top_k: Maximum number of nearest candidate chunks to retrieve (default: 5).
            document_id: Optional document ID to restrict retrieval.
            index_id: Optional specific index ID to query.
            index: Optional direct VectorIndex instance override.

        Returns:
            List of RetrievalResult objects ranked by similarity score.

        Raises:
            InvalidQueryError: If query is empty or whitespace-only.
            RetrievalError: If top_k <= 0.
            IndexNotFoundError: If no target index is available.
        """
        # 1. Query and parameter validation
        if query is None or not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("Search query cannot be empty or whitespace-only.")

        if top_k <= 0:
            raise RetrievalError(f"top_k must be strictly positive (got {top_k}).")

        # 2. Generate query embedding using same model contract as chunks
        query_vector = self.embedding_service.embed_query(query.strip())

        # 3. Delegate to vector retrieval
        return self.search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            index_id=index_id,
            index=index,
        )

    def save_index(
        self,
        index_id: str,
        directory: Union[str, Path],
        base_name: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """Save a registered VectorIndex to disk."""
        target_index = self.get_index(index_id)
        name = base_name or index_id
        return target_index.save(directory, base_name=name)

    def load_index(
        self,
        directory: Union[str, Path],
        base_name: str = "index",
        register: bool = True,
        set_as_default: bool = False,
    ) -> VectorIndex:
        """Load a persisted VectorIndex from disk and optionally register it."""
        index = VectorIndex.load(directory, base_name=base_name)
        if register:
            self.register_index(index, set_as_default=set_as_default)
        return index
