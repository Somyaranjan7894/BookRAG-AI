"""Comprehensive unit and integration tests for Phase 2 text cleaning and intelligent chunking."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.schemas.chunk import Chunk, ChunkingConfig
from app.schemas.document import Document, Page
from app.services.text.chunker import (
    Chunker,
    compute_overlap_prefix,
    split_oversized_sentence,
    split_sentences,
)
from app.services.text.cleaner import TextCleaner
from app.services.text.processor import TextProcessingService


# ==============================================================================
# A. Basic Cleaning Tests
# ==============================================================================

def test_clean_line_endings_crlf_cr_lf() -> None:
    """Requirement A: Normalize Windows CRLF and legacy CR line endings to LF."""
    raw = "Line one.\r\nLine two.\rLine three.\nLine four."
    cleaned = TextCleaner.clean(raw)
    assert "\r" not in cleaned
    assert "Line one. Line two. Line three. Line four." == cleaned


def test_clean_repeated_whitespace() -> None:
    """Requirement A: Collapse multiple horizontal spaces and tabs to single spaces."""
    raw = "Word1    Word2\t\tWord3   Word4."
    cleaned = TextCleaner.clean(raw)
    assert cleaned == "Word1 Word2 Word3 Word4."


def test_clean_repeated_blank_lines() -> None:
    """Requirement A: Collapse 3+ consecutive newlines to standard double newline (paragraph break)."""
    raw = "Paragraph one.\n\n\n\n\nParagraph two.\n\n\nParagraph three."
    cleaned = TextCleaner.clean(raw)
    assert "\n\n\n" not in cleaned
    assert cleaned == "Paragraph one.\n\nParagraph two.\n\nParagraph three."


def test_clean_leading_trailing_whitespace() -> None:
    """Requirement A: Strip extraneous leading and trailing whitespace from the document."""
    raw = "   \n\n   The main text content.   \n\n   "
    cleaned = TextCleaner.clean(raw)
    assert cleaned == "The main text content."


# ==============================================================================
# B. Safe Dehyphenation Tests
# ==============================================================================

def test_safe_dehyphenation_line_breaks() -> None:
    """Requirement B: Rejoin words broken across line wraps with a hyphen."""
    raw = "This is an artificial intel-\nligence system with trans-\nformers."
    cleaned = TextCleaner.clean(raw)
    assert "intelligence" in cleaned
    assert "transformers" in cleaned
    assert "intel-" not in cleaned


def test_safe_dehyphenation_preserves_inline_compound_words() -> None:
    """Requirement B: Normal compound words with hyphens must remain untouched."""
    raw = "This is a state-of-the-art, high-performance, domain-specific application."
    cleaned = TextCleaner.clean(raw)
    assert "state-of-the-art" in cleaned
    assert "high-performance" in cleaned
    assert "domain-specific" in cleaned


def test_safe_dehyphenation_preserves_cross_line_compound_names() -> None:
    """Requirement B: Hyphenated compound proper nouns split across lines retain hyphen."""
    raw = "The algorithm was authored by Smith-\nJones and Taylor-\nBrown."
    cleaned = TextCleaner.clean(raw)
    assert "Smith-Jones" in cleaned
    assert "Taylor-Brown" in cleaned


# ==============================================================================
# C. Paragraph Handling Tests
# ==============================================================================

def test_paragraph_preservation_small_paragraphs_stay_together() -> None:
    """Requirement C: Multiple small paragraphs fit inside target_size and remain together."""
    p1 = "First brief paragraph introducing the chapter."
    p2 = "Second brief paragraph elaborating on the core concept."
    p3 = "Third brief paragraph concluding the section."
    raw = f"{p1}\n\n{p2}\n\n{p3}"

    chunker = Chunker(default_config=ChunkingConfig(target_size=1200, max_size=1600))
    chunks = chunker.chunk_text(raw, document_id="doc_1", page_number=1)

    assert len(chunks) == 1
    assert p1 in chunks[0].text
    assert p2 in chunks[0].text
    assert p3 in chunks[0].text
    assert "\n\n" in chunks[0].text  # Paragraph separation is preserved


def test_multiple_paragraphs_split_when_exceeding_target_size() -> None:
    """Requirement C: Distinct paragraphs split across chunks when combined length exceeds target_size."""
    p1 = "Paragraph A with lots of content. " * 15  # ~525 chars
    p2 = "Paragraph B with lots of content. " * 15  # ~525 chars
    p3 = "Paragraph C with lots of content. " * 15  # ~525 chars
    raw = f"{p1}\n\n{p2}\n\n{p3}"

    chunker = Chunker(default_config=ChunkingConfig(target_size=800, max_size=1100, overlap=100))
    chunks = chunker.chunk_text(raw, document_id="doc_1", page_number=1)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.char_count <= 1100


# ==============================================================================
# D. Chunk Size Configuration Tests
# ==============================================================================

def test_chunking_config_validation() -> None:
    """Requirement D: Validation prevents contradictory or negative configuration values."""
    with pytest.raises(ValueError):
        # target_size cannot exceed max_size
        ChunkingConfig(target_size=2000, max_size=1500)

    with pytest.raises(ValueError):
        # overlap cannot be >= target_size
        ChunkingConfig(target_size=500, max_size=1000, overlap=500)

    with pytest.raises(ValueError):
        # target_size must be > 0
        ChunkingConfig(target_size=0, max_size=1000)


def test_chunking_config_overrides_behavior() -> None:
    """Requirement D: Smaller target_size produces more chunks for the same input text."""
    text = "Detailed scientific explanation. " * 60  # ~1980 chars

    large_config = ChunkingConfig(target_size=1200, max_size=1600, overlap=100)
    small_config = ChunkingConfig(target_size=400, max_size=600, overlap=50)

    chunker = Chunker()
    large_chunks = chunker.chunk_text(text, document_id="doc_1", page_number=1, config=large_config)
    small_chunks = chunker.chunk_text(text, document_id="doc_1", page_number=1, config=small_config)

    assert len(small_chunks) > len(large_chunks)
    for c in small_chunks:
        assert c.char_count <= 600


# ==============================================================================
# E. Long Paragraph & Sentence-Aware Splitting Tests
# ==============================================================================

def test_long_paragraph_split_by_sentences() -> None:
    """Requirement E: Paragraph larger than max_size is split cleanly at sentence boundaries."""
    sent1 = "The first sentence describes the initial conditions of the experiment."
    sent2 = "The second sentence explains the methodological apparatus in extensive detail."
    sent3 = "The third sentence records all quantitative observations made during the trial."
    sent4 = "The fourth sentence highlights unexpected anomalies discovered during analysis."
    long_para = f"{sent1} {sent2} {sent3} {sent4}"

    # Force split with small max_size
    config = ChunkingConfig(target_size=120, max_size=160, overlap=20)
    chunker = Chunker(default_config=config)
    chunks = chunker.chunk_text(long_para, document_id="doc_1", page_number=1)

    assert len(chunks) > 1
    # Check that chunks do not break in the middle of words
    for c in chunks:
        assert not c.text.startswith(" ")
        assert not c.text.endswith(" ")
        assert c.char_count <= 160


def test_abbreviations_not_split_as_sentences() -> None:
    """Requirement E: Common abbreviations and numbers do not trigger false sentence breaks."""
    text = "Dr. Smith met with Prof. Jones at 10 a.m. in the U.S. They tested model v1.0. Results were 99.8% positive."
    sentences = split_sentences(text)

    assert len(sentences) == 3
    assert sentences[0] == "Dr. Smith met with Prof. Jones at 10 a.m. in the U.S."
    assert sentences[1] == "They tested model v1.0."
    assert sentences[2] == "Results were 99.8% positive."


# ==============================================================================
# F. Very Long Sentence Tests
# ==============================================================================

def test_very_long_sentence_hard_boundary_fallback() -> None:
    """Requirement F: A massive sentence with no punctuation splits at word/hard boundary without hanging."""
    words = ["word" + str(i) for i in range(250)]  # ~1750 chars single sentence without period
    long_sent = " ".join(words)

    config = ChunkingConfig(target_size=400, max_size=500, overlap=50)
    chunker = Chunker(default_config=config)
    chunks = chunker.chunk_text(long_sent, document_id="doc_1", page_number=1)

    assert len(chunks) >= 3
    for c in chunks:
        assert c.char_count <= 500
        assert len(c.text.strip()) > 0


def test_oversized_unbreakable_token_slice() -> None:
    """Requirement F: Unbreakable continuous string larger than max_size is hard-sliced safely."""
    unbreakable = "A" * 1500
    segments = split_oversized_sentence(unbreakable, max_size=400)

    assert len(segments) == 4
    for seg in segments:
        assert len(seg) <= 400


# ==============================================================================
# G. Overlap Tests
# ==============================================================================

def test_overlap_applied_and_bounded() -> None:
    """Requirement G: Adjacent chunks on the same page contain bounded overlap from previous chunk."""
    p1 = "Paragraph 1 is very informative and sets up the whole thesis. " * 8
    p2 = "Paragraph 2 follows up with secondary arguments and proofs. " * 8
    raw = f"{p1}\n\n{p2}"

    config = ChunkingConfig(target_size=400, max_size=650, overlap=100)
    chunker = Chunker(default_config=config)
    chunks = chunker.chunk_text(raw, document_id="doc_1", page_number=1)

    assert len(chunks) >= 2
    # Verify that chunk 2 has text appearing at the end of chunk 1
    c1_tail = chunks[0].text[-50:].strip()
    assert c1_tail in chunks[1].text



def test_no_overlap_across_different_pages() -> None:
    """Requirement G & Page Boundaries: Overlap is never leaked across different pages."""
    page1 = Page(
        page_number=1,
        text="Page one concluding sentence with unique keyword AlphaOmega.",
        char_count=58,
        word_count=8,
        has_text=True,
    )
    page2 = Page(
        page_number=2,
        text="Page two opening sentence discussing a fresh topic.",
        char_count=52,
        word_count=8,
        has_text=True,
    )
    doc = Document(
        document_id="doc_test",
        filename="test.pdf",
        source_path="data/test.pdf",
        page_count=2,
        total_characters=110,
        total_words=16,
        pages=[page1, page2],
    )

    processor = TextProcessingService()
    chunks = processor.process_document(doc)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert "AlphaOmega" not in chunks[1].text  # Page 1 text MUST NOT leak into Page 2


# ==============================================================================
# H. Provenance & Deterministic ID Tests
# ==============================================================================

def test_provenance_strictly_preserved() -> None:
    """Requirement H: document_id, page_number, and chunk indices are accurately preserved."""
    text = "Sample book content for testing page provenance preservation."
    chunker = Chunker()
    chunks = chunker.chunk_text(text, document_id="book_999", page_number=42, start_chunk_index=7)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.document_id == "book_999"
    assert c.page_number == 42
    assert c.chunk_index == 7
    assert c.chunk_id == "book_999_p042_c0007"
    assert c.metadata["source_page"] == 42


def test_chunk_ids_unique_across_document() -> None:
    """Requirement H: All chunk IDs across a multi-page document are guaranteed unique."""
    doc = Document(
        document_id="unique_doc",
        filename="doc.pdf",
        source_path="/path/doc.pdf",
        page_count=3,
        total_characters=3000,
        total_words=400,
        pages=[
            Page(page_number=1, text="Page 1 text. " * 30, char_count=390, word_count=90, has_text=True),
            Page(page_number=2, text="Page 2 text. " * 30, char_count=390, word_count=90, has_text=True),
            Page(page_number=3, text="Page 3 text. " * 30, char_count=390, word_count=90, has_text=True),
        ],
    )
    processor = TextProcessingService(default_config=ChunkingConfig(target_size=200, max_size=300, overlap=30))
    chunks = processor.process_document(doc)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates


# ==============================================================================
# I. Edge Cases Tests
# ==============================================================================

def test_empty_and_whitespace_page_produces_no_chunks() -> None:
    """Requirement I: Empty or whitespace-only pages produce zero chunks (no empty chunks)."""
    chunker = Chunker()
    assert chunker.chunk_text("", document_id="doc_1", page_number=1) == []
    assert chunker.chunk_text("   \n\t  \n  ", document_id="doc_1", page_number=1) == []


def test_clean_preserves_numbers_code_and_punctuation() -> None:
    """Requirement I: Numbers, equations, symbols, and punctuation remain intact."""
    raw = "Formula: x = (-b ± √(b² - 4ac)) / (2a); Accuracy: 99.4%; Year: 2026."
    cleaned = TextCleaner.clean(raw)
    assert "x = (-b ± √(b² - 4ac)) / (2a)" in cleaned
    assert "99.4%" in cleaned
    assert "2026" in cleaned


def test_unicode_and_multilingual_text() -> None:
    """Requirement I: Unicode characters and accented words are fully preserved."""
    raw = "Schöne Grüße aus München! Déjà vu dans le café. α-helix and β-sheet."
    cleaned = TextCleaner.clean(raw)
    assert "Schöne Grüße aus München!" in cleaned
    assert "Déjà vu dans le café." in cleaned
    assert "α-helix and β-sheet." in cleaned


def test_structured_bullet_lists_preserved() -> None:
    """Requirement I: Bullet lists and enumerated outlines retain structural breaks."""
    raw = "Key features:\n* Item 1\n* Item 2\n* Item 3"
    cleaned = TextCleaner.clean(raw)
    assert "* Item 1\n* Item 2\n* Item 3" in cleaned


# ==============================================================================
# J. Determinism & Non-Destructive Invariance Tests
# ==============================================================================

def test_chunking_is_strictly_deterministic() -> None:
    """Requirement J: Repeated processing of identical inputs produces identical chunks and IDs."""
    text = "Deterministic testing text. " * 50
    config = ChunkingConfig(target_size=300, max_size=400, overlap=50)

    chunker = Chunker()
    run1 = chunker.chunk_text(text, document_id="doc_det", page_number=1, config=config)
    run2 = chunker.chunk_text(text, document_id="doc_det", page_number=1, config=config)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.char_count == c2.char_count
        assert c1.word_count == c2.word_count


def test_raw_page_text_is_not_destructively_modified() -> None:
    """Requirement J: Document processing never alters the original Page.text attributes."""
    raw_content = "Raw  page   text with   excessive  spaces.\r\nNext line."
    page = Page(
        page_number=1,
        text=raw_content,
        char_count=len(raw_content),
        word_count=len(raw_content.split()),
        has_text=True,
    )
    doc = Document(
        document_id="immutability_test",
        filename="immut.pdf",
        source_path="/path/immut.pdf",
        page_count=1,
        total_characters=len(raw_content),
        total_words=len(raw_content.split()),
        pages=[page],
    )

    processor = TextProcessingService()
    chunks = processor.process_document(doc)

    assert len(chunks) == 1
    # Verify Page.text is completely untouched!
    assert doc.pages[0].text == raw_content


# ==============================================================================
# Development API Endpoint Tests
# ==============================================================================

def test_api_clean_endpoint(client: TestClient) -> None:
    """Development API endpoint POST /api/v1/chunks/clean normalizes text."""
    response = client.post(
        "/api/v1/chunks/clean",
        json={"text": "   Text with   spaces  and intel-\nligence.   "},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "cleaned_text" in data
    assert data["cleaned_text"] == "Text with spaces and intelligence."


def test_api_chunk_page_endpoint(client: TestClient) -> None:
    """Development API endpoint POST /api/v1/chunks/chunk-page produces structured chunks."""
    response = client.post(
        "/api/v1/chunks/chunk-page",
        json={
            "document_id": "api_doc_1",
            "page_number": 3,
            "text": "First paragraph. " * 30 + "\n\n" + "Second paragraph. " * 30,
            "config": {"target_size": 300, "max_size": 400, "overlap": 40},
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["document_id"] == "api_doc_1"
    assert data["page_number"] == 3
    assert data["total_chunks"] > 1
    assert data["chunks"][0]["chunk_id"].startswith("api_doc_1_p003_c")
