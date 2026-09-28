"""Document-level text processing service for BookRAG AI.

Coordinates text cleaning and page-by-page chunking across complete documents
while strictly preserving raw page text integrity and provenance.
"""

from typing import List, Optional

from app.core.logging import get_logger
from app.schemas.chunk import Chunk, ChunkingConfig
from app.schemas.document import Document
from app.services.text.chunker import Chunker
from app.services.text.cleaner import TextCleaner

logger = get_logger(__name__)


class TextProcessingService:
    """Orchestrates end-to-end cleaning and chunking for full Document representations."""

    def __init__(
        self,
        cleaner: Optional[TextCleaner] = None,
        chunker: Optional[Chunker] = None,
        default_config: Optional[ChunkingConfig] = None,
    ) -> None:
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or Chunker(default_config=default_config)

    def process_document(
        self,
        document: Document,
        config: Optional[ChunkingConfig] = None,
    ) -> List[Chunk]:
        """Process all pages of a Document into an ordered collection of Chunks.

        CRITICAL PROVENANCE GUARANTEE:
        - Pages are processed strictly independently without cross-page merging.
        - The raw `page.text` of each Page in `document.pages` is NOT modified.
        - Each Chunk retains its exact `document_id` and `page_number`.

        Args:
            document: Ingested Document containing raw extracted pages.
            config: Optional ChunkingConfig override for chunk sizes and overlap.

        Returns:
            List[Chunk]: Complete ordered list of generated Chunks across all pages.
        """
        logger.info(
            "Starting text processing for document '%s' (%d pages)",
            document.document_id,
            document.page_count,
        )

        all_chunks: List[Chunk] = []

        for page in document.pages:
            # Skip empty or textless pages gracefully without raising errors
            if not page.has_text or not page.text.strip():
                logger.debug(
                    "Skipping chunk generation for empty page %d in document '%s'",
                    page.page_number,
                    document.document_id,
                )
                continue

            page_chunks = self.chunker.chunk_page(
                page=page,
                document_id=document.document_id,
                config=config,
                start_chunk_index=len(all_chunks),
            )

            all_chunks.extend(page_chunks)

        logger.info(
            "Completed text processing for document '%s': generated %d total chunks across %d pages",
            document.document_id,
            len(all_chunks),
            document.page_count,
        )

        return all_chunks
