"""Vector search backend abstractions supporting FAISS and pgvector."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.repositories.vector_repository import BaseVectorRepository, PGVectorRepository
from app.schemas.retrieval import RetrievalResult
from app.services.retrieval.service import RetrievalService

logger = get_logger(__name__)


class VectorSearchBackend(ABC):
    """Abstract interface decoupling search services from specific vector engines."""

    @abstractmethod
    def search(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        top_k: int = 5,
        document_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Perform first-stage candidate retrieval using a dense query vector."""
        pass

    @abstractmethod
    def has_document(self, document_id: str, **kwargs: Any) -> bool:
        """Check whether the given document exists in this backend."""
        pass

    @abstractmethod
    def total_vectors(self, **kwargs: Any) -> int:
        """Return total number of searchable vectors in this backend."""
        pass


class FAISSVectorBackend(VectorSearchBackend):
    """FAISS-based vector search backend utilizing RetrievalService."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def search(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        top_k: int = 5,
        document_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        vec = np.asarray(query_vector, dtype=np.float32)
        index_id = kwargs.get("index_id")
        return self.retrieval_service.search_by_vector(
            query_vector=vec,
            top_k=top_k,
            document_id=document_id,
            index_id=index_id,
        )

    def has_document(self, document_id: str, **kwargs: Any) -> bool:
        index_id = kwargs.get("index_id")
        try:
            target_index = self.retrieval_service.get_index(index_id)
            known_docs = set(target_index.metadata.document_ids)
            has_doc_index = (
                hasattr(self.retrieval_service, "_indices")
                and document_id in self.retrieval_service._indices
            )
            return document_id in known_docs or has_doc_index
        except Exception:
            return False

    def total_vectors(self, **kwargs: Any) -> int:
        index_id = kwargs.get("index_id")
        target_index = self.retrieval_service.get_index(index_id)
        return target_index.total_vectors


class PGVectorBackend(VectorSearchBackend):
    """PostgreSQL pgvector database-native vector search backend."""

    def __init__(
        self,
        vector_repo: Optional[BaseVectorRepository] = None,
        session_factory=None,
    ) -> None:
        self._vector_repo = vector_repo
        self._session_factory = session_factory or get_session_factory()

    def _get_repo(self):
        if self._vector_repo is not None:
            return self._vector_repo, None
        session = self._session_factory()
        return PGVectorRepository(session), session

    def search(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        top_k: int = 5,
        document_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        repo, session = self._get_repo()
        try:
            return repo.search_similar(
                query_vector=query_vector,
                top_k=top_k,
                document_id=document_id,
            )
        finally:
            if session is not None:
                session.close()

    def has_document(self, document_id: str, **kwargs: Any) -> bool:
        repo, session = self._get_repo()
        try:
            return repo.count_embeddings(document_id=document_id) > 0
        finally:
            if session is not None:
                session.close()

    def total_vectors(self, **kwargs: Any) -> int:
        repo, session = self._get_repo()
        try:
            return repo.count_embeddings()
        finally:
            if session is not None:
                session.close()


def create_vector_backend(
    backend_type: Optional[str] = None,
    retrieval_service: Optional[RetrievalService] = None,
    vector_repo: Optional[BaseVectorRepository] = None,
) -> VectorSearchBackend:
    """Factory creating the appropriate vector search backend."""
    chosen = (backend_type or settings.VECTOR_BACKEND).lower()
    if chosen == "pgvector":
        return PGVectorBackend(vector_repo=vector_repo)
    if retrieval_service is None:
        retrieval_service = RetrievalService()
    return FAISSVectorBackend(retrieval_service=retrieval_service)
