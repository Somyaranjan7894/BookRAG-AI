"""Vector-to-chunk metadata mapping for FAISS index positions.

Maintains an explicit, deterministic 1-to-1 correspondence between FAISS internal
integer positions and chunk provenance metadata.
"""

from typing import Any, Dict, List, Optional
from app.schemas.retrieval import VectorMappingItem
from app.services.retrieval.exceptions import CorruptedIndexError


class VectorToChunkMapping:
    """Manages deterministic mapping between FAISS vector slots and source chunk metadata."""

    def __init__(self) -> None:
        self._mapping: Dict[int, VectorMappingItem] = {}

    def add_item(self, item: VectorMappingItem) -> None:
        """Register a new mapping entry for a given FAISS vector index position."""
        self._mapping[item.vector_index] = item

    def get(self, vector_index: int) -> Optional[VectorMappingItem]:
        """Retrieve chunk provenance for a specific FAISS vector index position."""
        return self._mapping.get(vector_index)

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self):
        return iter(self._mapping.values())

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize mapping to a list of dictionaries for JSON persistence."""
        sorted_keys = sorted(self._mapping.keys())
        return [self._mapping[k].model_dump() for k in sorted_keys]

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> "VectorToChunkMapping":
        """Reconstruct mapping from serialized list of dictionaries."""
        mapping = cls()
        for entry in data:
            item = VectorMappingItem(**entry)
            mapping.add_item(item)
        return mapping

    def validate_consistency(self, expected_vector_count: int) -> None:
        """Verify that mapping contains exactly contiguous indices from 0 to expected_count-1.

        Args:
            expected_vector_count: The number of vectors reported by FAISS (ntotal).

        Raises:
            CorruptedIndexError: If mapping size or keys do not match contiguous vector slots.
        """
        if len(self._mapping) != expected_vector_count:
            raise CorruptedIndexError(
                f"Mapping count ({len(self._mapping)}) does not match FAISS vector count ({expected_vector_count})."
            )

        for i in range(expected_vector_count):
            if i not in self._mapping:
                raise CorruptedIndexError(
                    f"Missing vector mapping entry for index position {i} (expected 0..{expected_vector_count - 1})."
                )
