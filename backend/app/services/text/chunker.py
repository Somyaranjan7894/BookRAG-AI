"""Paragraph and sentence-aware intelligent text chunker for BookRAG AI.

Transforms cleaned page text into bounded, semantically coherent text chunks
with strict document and page-level provenance.
"""

import re
from typing import List, Optional

from app.core.logging import get_logger
from app.schemas.chunk import Chunk, ChunkingConfig
from app.schemas.document import Page
from app.services.text.cleaner import TextCleaner
from app.services.text.exceptions import InvalidChunkingConfigError

logger = get_logger(__name__)

# Standard abbreviations to avoid false positive sentence breaks.
_KNOWN_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "e.g", "i.e",
    "fig", "al", "vol", "p", "pp", "no", "sec", "dept", "approx", "est",
}


# Regex to detect sentence boundaries: terminal punctuation followed by space and uppercase/number/quote.
_SENTENCE_BOUNDARY_RE = re.compile(
    r'([.!?]+[\"\'\u201d\u2019\)]*)\s+(?=[A-Z0-9\"\'\u201c\u2018])'
)


def split_sentences(text: str) -> List[str]:
    """Deterministically split text into sentences, protecting common abbreviations.

    Args:
        text: Paragraph or text block to segment.

    Returns:
        List of individual sentence strings.
    """
    text = text.strip()
    if not text:
        return []

    splits: List[int] = []
    last_idx = 0

    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        term_idx = match.end(1)
        # Check token preceding the terminal punctuation
        preceding = text[last_idx:term_idx].split()
        if preceding:
            last_word = preceding[-1].rstrip('.!?\"\'”’)]').lower()
            # If word is an abbreviation or a single letter initial (e.g. "J. Smith"), skip split
            if last_word in _KNOWN_ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                continue
        splits.append(match.end())

    sentences: List[str] = []
    start = 0
    for split_idx in splits:
        sent = text[start:split_idx].strip()
        if sent:
            sentences.append(sent)
        start = split_idx

    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)

    return sentences


def split_oversized_sentence(sentence: str, max_size: int) -> List[str]:
    """Split an oversized sentence exceeding max_size into word-bounded segments.

    If a single word is larger than max_size, it is hard-sliced at character boundary.

    Args:
        sentence: Long sentence string.
        max_size: Maximum allowable character length per segment.

    Returns:
        List of segments, each guaranteed <= max_size.
    """
    if len(sentence) <= max_size:
        return [sentence]

    words = sentence.split(" ")
    segments: List[str] = []
    current_words: List[str] = []
    current_len = 0

    for word in words:
        # Handle unbreakable single token larger than max_size
        if len(word) > max_size:
            if current_words:
                segments.append(" ".join(current_words))
                current_words = []
                current_len = 0
            for i in range(0, len(word), max_size):
                segments.append(word[i:i + max_size])
            continue

        projected = current_len + (1 if current_words else 0) + len(word)
        if projected <= max_size:
            current_words.append(word)
            current_len = projected
        else:
            if current_words:
                segments.append(" ".join(current_words))
            current_words = [word]
            current_len = len(word)

    if current_words:
        segments.append(" ".join(current_words))

    return [s for s in segments if s.strip()]


def compute_overlap_prefix(prev_text: str, overlap: int) -> str:
    """Extract an overlap prefix from the tail of the previous chunk bounded by word boundary.

    Args:
        prev_text: Text of the chunk just emitted.
        overlap: Maximum target character count for overlap.

    Returns:
        Cleaned substring from tail of previous text to prepend to the next chunk.
    """
    if overlap <= 0 or not prev_text.strip():
        return ""

    prev_text = prev_text.strip()
    if len(prev_text) <= overlap:
        return prev_text

    tail = prev_text[-overlap:].lstrip()
    first_space = tail.find(" ")
    if first_space != -1 and first_space < len(tail) - 1:
        tail = tail[first_space + 1:].strip()

    return tail


class Chunker:
    """Paragraph-first, sentence-aware text chunking service."""

    def __init__(self, default_config: Optional[ChunkingConfig] = None) -> None:
        self.default_config = default_config or ChunkingConfig()

    def chunk_text(
        self,
        text: str,
        document_id: str,
        page_number: int,
        config: Optional[ChunkingConfig] = None,
        start_chunk_index: int = 0,
    ) -> List[Chunk]:
        """Chunk a text stream belonging to a specific page into bounded Chunk objects.

        Chunking process:
        1. Text is cleaned via TextCleaner (conservative normalization).
        2. Cleaned text is segmented into paragraphs (preferred semantic boundary).
        3. Paragraphs > max_size are broken into sentences.
        4. Sentences > max_size are segmented at word/character boundaries.
        5. Units are accumulated until target_size is reached or adding next unit exceeds max_size.
        6. When a chunk is emitted, controlled overlap is applied for subsequent chunk.
        7. Chunk IDs are deterministically generated with strict page provenance.

        Args:
            text: Raw or partially cleaned text content.
            document_id: Unique identifier for the parent document.
            page_number: 1-based source page number.
            config: Optional ChunkingConfig override.
            start_chunk_index: Starting sequential chunk counter for chunk IDs.

        Returns:
            List of deterministic Chunk objects. Empty list if text is empty/whitespace.
        """
        cfg = config or self.default_config
        cleaned = TextCleaner.clean(text)

        if not cleaned:
            return []

        # Split into paragraphs
        raw_paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        if not raw_paragraphs:
            return []

        # Decompose paragraphs into units that strictly satisfy len(unit) <= cfg.max_size
        # Each entry is a tuple: (unit_text, is_paragraph_start)
        decomposed_units: List[tuple[str, bool]] = []

        for para in raw_paragraphs:
            if len(para) <= cfg.max_size:
                decomposed_units.append((para, True))
            else:
                # Paragraph is larger than max_size -> decompose into sentences
                sentences = split_sentences(para)
                first_in_para = True
                for sent in sentences:
                    if len(sent) <= cfg.max_size:
                        decomposed_units.append((sent, first_in_para))
                        first_in_para = False
                    else:
                        # Sentence is larger than max_size -> word/hard segment
                        segments = split_oversized_sentence(sent, cfg.max_size)
                        for seg in segments:
                            decomposed_units.append((seg, first_in_para))
                            first_in_para = False

        # Accumulate units into chunks
        chunks: List[Chunk] = []
        current_units: List[str] = []
        current_len = 0
        chunk_idx = start_chunk_index

        def emit_chunk(chunk_content: str) -> None:
            nonlocal chunk_idx
            chunk_content = chunk_content.strip()
            if not chunk_content:
                return

            chunk_id = f"{document_id}_p{page_number:03d}_c{chunk_idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    text=chunk_content,
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    metadata={"source_page": page_number},
                )
            )
            chunk_idx += 1

        for unit_text, is_para_start in decomposed_units:
            if not current_units:
                current_units.append(unit_text)
                current_len = len(unit_text)
                continue

            separator = "\n\n" if is_para_start else " "
            projected_len = current_len + len(separator) + len(unit_text)

            # If adding unit stays within max_size and chunk hasn't exceeded target_size
            if projected_len <= cfg.max_size and (current_len < cfg.target_size or projected_len <= cfg.target_size):
                current_units.append(separator + unit_text)
                current_len = projected_len
            else:
                # Finalize and emit current chunk
                emitted_text = "".join(current_units)
                emit_chunk(emitted_text)

                # Reset accumulator
                current_units = []
                current_len = 0

                # Compute available headroom for overlap in the new chunk
                available_headroom = cfg.max_size - len(separator) - len(unit_text)
                effective_overlap = min(cfg.overlap, available_headroom)

                overlap_prefix = ""
                if effective_overlap >= 20:
                    overlap_prefix = compute_overlap_prefix(emitted_text, effective_overlap)

                if (
                    overlap_prefix
                    and overlap_prefix != unit_text
                    and not unit_text.startswith(overlap_prefix)
                    and len(overlap_prefix) + len(separator) + len(unit_text) <= cfg.max_size
                ):
                    current_units.append(overlap_prefix)
                    current_units.append(separator + unit_text)
                    current_len = len(overlap_prefix) + len(separator) + len(unit_text)
                else:
                    current_units.append(unit_text)
                    current_len = len(unit_text)


        if current_units:
            final_text = "".join(current_units)
            emit_chunk(final_text)

        return chunks

    def chunk_page(
        self,
        page: Page,
        document_id: str,
        config: Optional[ChunkingConfig] = None,
        start_chunk_index: int = 0,
    ) -> List[Chunk]:
        """Convenience method to chunk directly from a Phase 1 Page model.

        Preserves page provenance and does not mutate page.text.

        Args:
            page: Phase 1 Page instance.
            document_id: Parent document identifier.
            config: Optional ChunkingConfig override.
            start_chunk_index: Starting sequential chunk counter.

        Returns:
            List of generated Chunk objects for this page.
        """
        return self.chunk_text(
            text=page.text,
            document_id=document_id,
            page_number=page.page_number,
            config=config,
            start_chunk_index=start_chunk_index,
        )
