"""Deterministic text cleaning service for BookRAG AI.

Performs conservative text normalization, whitespace cleanup, and safe dehyphenation
while preserving punctuation, numbers, code-like structures, and authorial meaning.
"""

import re
from typing import List

from app.core.logging import get_logger

logger = get_logger(__name__)

# Pattern to repair word-level hyphenation broken across a line break.
# e.g., "intel-\nligence" -> "intelligence"
# Requires at least 2 letters before the hyphen and 2 lowercase letters following it.
_LINEBREAK_HYPHEN_LOWERCASE = re.compile(
    r"(\b[a-zA-Z]{2,})-(?:[^\S\n]*\n[^\S\n]*)([a-z]{2,})\b"
)

# Pattern to preserve hyphenated compound words split across lines.
# e.g., "Smith-\nJones" -> "Smith-Jones"
_LINEBREAK_HYPHEN_COMPOUND = re.compile(
    r"(\b[a-zA-Z]{2,})-(?:[^\S\n]*\n[^\S\n]*)([A-Z][a-zA-Z]+)\b"
)

# Detect list items, headers, code, or block quotes that should maintain line breaks.
_STRUCTURED_LINE_PREFIXES = re.compile(
    r"^(?:\s*[-*•–—]\s+|\s*\d+[.)]\s+|\s*[>#]\s+|\s{4,}|\t)"
)


class TextCleaner:
    """Conservative, deterministic text cleaner."""

    @staticmethod
    def normalize_line_endings(text: str) -> str:
        """Convert Windows (CRLF) and legacy Mac (CR) line endings to standard Unix (LF)."""
        if not text:
            return ""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def repair_hyphenation(text: str) -> str:
        """Repair words broken across line wraps while preserving genuine compound hyphens.

        Examples:
            "intel-\\nligence" -> "intelligence"
            "state-of-the-art" -> "state-of-the-art" (unchanged)
            "Smith-\\nJones" -> "Smith-Jones"
        """
        if not text:
            return ""
        # First repair mid-word hyphens continuing in lowercase
        repaired = _LINEBREAK_HYPHEN_LOWERCASE.sub(r"\1\2", text)
        # Preserve compound hyphens split across line break
        repaired = _LINEBREAK_HYPHEN_COMPOUND.sub(r"\1-\2", repaired)
        return repaired

    @staticmethod
    def normalize_horizontal_whitespace(text: str) -> str:
        """Collapse multiple horizontal spaces and tabs into a single space per line."""
        if not text:
            return ""
        # Replace non-breaking spaces or tabs with regular spaces
        cleaned = text.replace("\u00a0", " ").replace("\t", " ")
        # Collapse multiple horizontal whitespace characters, preserving newlines
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        return cleaned

    @classmethod
    def clean(cls, text: str) -> str:
        """Apply conservative, deterministic cleaning pipeline to raw text.

        Pipeline stages:
        1. Line ending normalization (CRLF/CR -> LF)
        2. Safe line-break dehyphenation
        3. Horizontal whitespace collapse per line
        4. Soft line wrap joining within paragraphs (preserving structured list/code lines)
        5. Excessive blank lines normalization (3+ newlines -> 2 newlines)
        6. Strip leading and trailing whitespace

        Args:
            text: Raw extracted page text.

        Returns:
            Cleaned, readable, paragraph-structured text.
        """
        if not text or not text.strip():
            return ""

        # Step 1: Normalize line endings
        normalized = cls.normalize_line_endings(text)

        # Step 2: Repair line-break hyphenation
        dehyphenated = cls.repair_hyphenation(normalized)

        # Step 3: Normalize horizontal whitespace
        ws_cleaned = cls.normalize_horizontal_whitespace(dehyphenated)

        # Step 4: Split into paragraphs by 2 or more newlines
        raw_paragraphs = re.split(r"\n{2,}", ws_cleaned)
        cleaned_paragraphs: List[str] = []

        for para in raw_paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue

            lines = [line.strip() for line in para_stripped.split("\n")]
            lines = [line for line in lines if line]

            if not lines:
                continue

            # Join lines within a paragraph:
            # If any line looks like structured text (bullet, number, code, markdown),
            # preserve original line breaks; otherwise join soft line breaks with space.
            has_structured_lines = any(_STRUCTURED_LINE_PREFIXES.match(line) for line in lines)

            if has_structured_lines:
                cleaned_para = "\n".join(lines)
            else:
                cleaned_para = " ".join(lines)

            # Ensure internal multi-spaces are collapsed
            cleaned_para = re.sub(r" +", " ", cleaned_para).strip()
            if cleaned_para:
                cleaned_paragraphs.append(cleaned_para)

        # Step 5: Join paragraphs with double newline
        result = "\n\n".join(cleaned_paragraphs)

        return result.strip()
