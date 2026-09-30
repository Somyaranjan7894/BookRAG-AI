"""Repository for pgvector persistence and vector similarity search operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.schemas.retrieval import RetrievalResult

logger = get_logger(__name__)


class BaseVectorRepository(ABC):
    """Abstract interface for vector persistence and similarity search."""

    @abstractmethod
    def store_embedding(self, chunk_id: str, embedding: Sequence[float]) -> bool:
        """Store or update embedding vector for a single chunk."""
        pass

    @abstractmethod
    def store_embeddings(self, embeddings_by_chunk_id: Dict[str, Sequence[float]]) -> int:
        """Persist embeddings for multiple chunks in batch."""
        pass

    @abstractmethod
    def search_similar(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Perform vector similarity search, optionally filtered by document_id."""
        pass

    @abstractmethod
    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Retrieve the vector embedding for a specific chunk."""
        pass

    @abstractmethod
    def delete_embedding(self, chunk_id: str) -> bool:
        """Clear embedding vector for a chunk without deleting the chunk itself."""
        pass

    @abstractmethod
    def delete_embeddings_for_document(self, document_id: str) -> int:
        """Clear all embedding vectors for chunks belonging to a document."""
        pass

    @abstractmethod
    def count_embeddings(self, document_id: Optional[str] = None) -> int:
        """Count chunks that have non-null embeddings (optionally for a document)."""
        pass


class PGVectorRepository(BaseVectorRepository):
    """Encapsulates PostgreSQL pgvector storage, retrieval, and similarity search."""

    def __init__(
        self,
        session: Session,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ) -> None:
        """Initialize PGVectorRepository.

        Args:
            session: SQLAlchemy session instance.
            dimension: Expected vector dimension (defaults to 384 for all-MiniLM-L6-v2).
        """
        self.session = session
        self.dimension = dimension

    def _validate_vector_dimension(self, vector: Sequence[float], label: str = "vector") -> List[float]:
        """Validate that a vector matches the configured embedding dimensionality."""
        if hasattr(vector, "squeeze"):
            vector = vector.squeeze()
        if hasattr(vector, "tolist"):
            vec_list = vector.tolist()
        else:
            vec_list = list(vector)

        if vec_list and isinstance(vec_list[0], (list, tuple)):
            vec_list = list(vec_list[0])

        if not isinstance(vec_list, list) or len(vec_list) != self.dimension:
            actual_dim = len(vec_list) if hasattr(vec_list, "__len__") else "unknown"
            raise ValueError(
                f"Embedding dimension mismatch for {label}: expected {self.dimension}, got {actual_dim}"
            )
        return [float(x) for x in vec_list]

    def store_embedding(self, chunk_id: str, embedding: Sequence[float]) -> bool:
        """Store or update embedding vector for a single chunk.

        Args:
            chunk_id: Primary key of the target chunk.
            embedding: 384-dimensional dense vector sequence.

        Returns:
            True if chunk was found and updated, False otherwise.
        """
        vec_list = self._validate_vector_dimension(embedding, label=f"chunk '{chunk_id}'")
        stmt = (
            update(Chunk)
            .where(Chunk.chunk_id == chunk_id)
            .values(embedding=vec_list)
        )
        res = self.session.execute(stmt)
        self.session.flush()
        return res.rowcount > 0

    def store_embeddings(self, embeddings_by_chunk_id: Dict[str, Sequence[float]]) -> int:
        """Persist embeddings for multiple chunks in batch.

        Args:
            embeddings_by_chunk_id: Mapping of chunk_id to 384-dimensional vector sequence.

        Returns:
            Count of successfully updated chunks.
        """
        if not embeddings_by_chunk_id:
            return 0

        # Validate all dimensions before modifying database
        validated_items: List[tuple[str, List[float]]] = []
        for cid, emb in embeddings_by_chunk_id.items():
            valid_emb = self._validate_vector_dimension(emb, label=f"chunk '{cid}'")
            validated_items.append((cid, valid_emb))

        updated_count = 0
        for chunk_id, vec in validated_items:
            stmt = (
                update(Chunk)
                .where(Chunk.chunk_id == chunk_id)
                .values(embedding=vec)
            )
            res = self.session.execute(stmt)
            updated_count += res.rowcount

        self.session.flush()
        return updated_count

    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Retrieve embedding vector for a given chunk."""
        stmt = select(Chunk.embedding).where(Chunk.chunk_id == chunk_id)
        res = self.session.execute(stmt).scalar_one_or_none()
        if res is None:
            return None
        return list(res)

    def search_similar(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Perform vector similarity search against persisted chunks using pgvector cosine distance.

        Cosine Distance Operator: `<=>`
        Cosine Similarity = 1.0 - Cosine Distance
        Since vectors from sentence-transformers are L2-normalized (||v|| ≈ 1.0),
        cosine similarity is equivalent to inner product in FAISS.

        Args:
            query_vector: 384-dimensional dense query vector sequence.
            top_k: Maximum number of nearest candidates to return.
            document_id: Optional document ID to enforce document-level isolation.

        Returns:
            List of RetrievalResult objects ranked by descending similarity score.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be strictly positive (got {top_k}).")

        query_list = self._validate_vector_dimension(query_vector, label="query")

        stmt = (
            select(
                Chunk,
                Chunk.embedding.cosine_distance(query_list).label("distance"),
            )
            .where(Chunk.embedding.isnot(None))
        )

        if document_id is not None and document_id.strip():
            stmt = stmt.where(Chunk.document_id == document_id.strip())

        stmt = stmt.order_by("distance").limit(top_k)

        rows = self.session.execute(stmt).all()

        results: List[RetrievalResult] = []
        for rank_idx, (chunk, distance) in enumerate(rows, start=1):
            dist_val = float(distance) if distance is not None else 1.0
            # Cosine similarity in range [-1.0, 1.0]
            similarity_score = max(-1.0, min(1.0, 1.0 - dist_val))

            metadata: Dict[str, Any] = {
                "char_count": chunk.char_count,
                "word_count": chunk.word_count,
                "page_id": chunk.page_id,
            }

            results.append(
                RetrievalResult(
                    rank=rank_idx,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    similarity_score=round(similarity_score, 6),
                    metadata=metadata,
                )
            )

        return results

    def delete_embedding(self, chunk_id: str) -> bool:
        """Clear embedding for a single chunk."""
        stmt = (
            update(Chunk)
            .where(Chunk.chunk_id == chunk_id)
            .where(Chunk.embedding.isnot(None))
            .values(embedding=None)
        )
        res = self.session.execute(stmt)
        self.session.flush()
        return res.rowcount > 0

    def delete_embeddings_for_document(self, document_id: str) -> int:
        """Clear all embeddings for chunks belonging to a document."""
        stmt = (
            update(Chunk)
            .where(Chunk.document_id == document_id)
            .where(Chunk.embedding.isnot(None))
            .values(embedding=None)
        )
        res = self.session.execute(stmt)
        self.session.flush()
        return res.rowcount

    def count_embeddings(self, document_id: Optional[str] = None) -> int:
        """Count chunks having populated embeddings, optionally filtered by document."""
        stmt = select(func.count()).select_from(Chunk).where(Chunk.embedding.isnot(None))
        if document_id is not None and document_id.strip():
            stmt = stmt.where(Chunk.document_id == document_id.strip())
        return self.session.execute(stmt).scalar_one() or 0
