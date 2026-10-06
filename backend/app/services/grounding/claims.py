"""Deterministic claim decomposition for BookRAG AI Phase 9.

Decomposes generated answers into discrete sentence-level claim units for individual
NLI groundedness verification against retrieved book evidence.

ARCHITECTURAL LIMITATIONS:
Sentence splitting produces sentence-level propositions as a deterministic heuristic.
It does NOT produce true semantic atomic claims (e.g., compound sentences containing
multiple independent factual clauses are evaluated as a single unit). This modular
design allows a more advanced syntactic or dependency-based atomic claim decomposer
to be introduced in future iterations without changing the downstream NLI pipeline.
"""

import re
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.grounding import ExtractedClaim

logger = get_logger(__name__)

# Heuristic sentence boundary split: punctuation followed by space or line break
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


# Common words indicating verbal predicates
FINITE_VERBS_REGEX = re.compile(
    r"\b(is|are|was|were|has|have|had|do|does|did|can|could|will|would|should|may|might|"
    r"calculates|computes|optimizes|encodes|relies|measures|evaluates|counteracts|mitigates|"
    r"concatenates|discovers|propagates|exploits|operates|classifies)\b",
    re.IGNORECASE,
)


def classify_claim_form(text: str) -> str:
    """Classify the syntactic and structural form of a claim text.

    Distinguishes:
    - 'numerical': numbers, percentages, constants, values
    - 'list': comma-separated or coordinated item list without finite verbs
    - 'noun_phrase': multi-word noun phrase without finite verbs
    - 'short_span': very short span (<= 5 words)
    - 'declarative': full sentence with subject and predicate
    """
    s = text.strip()
    words = s.split()

    if not words:
        return "short_span"

    # Numerical form
    if (
        re.search(r"^(?:a\s+constant\s+)?-?\d+(?:\.\d+)?%?$", s, re.IGNORECASE)
        or re.search(r"^\d+(?:\.\d+)?\s*(?:percent|%|parameters|layers|units)?$", s, re.IGNORECASE)
        or re.search(r"^a\s+constant\s+\d+$", s, re.IGNORECASE)
    ):
        return "numerical"

    # List form
    if ("," in s or " and " in s or " or " in s) and len(words) <= 12:
        if not FINITE_VERBS_REGEX.search(s):
            return "list"

    # Noun phrase or short span
    has_finite_verb = bool(FINITE_VERBS_REGEX.search(s))
    if not has_finite_verb:
        if len(words) <= 5:
            return "short_span"
        return "noun_phrase"

    return "declarative"


def contextualize_fragment(question: Optional[str], fragment: str) -> str:
    """Transform a non-sentential fragment into a declarative proposition using question context.

    Ensures DeBERTa NLI evaluates a complete semantic proposition rather than an isolated phrase.
    """
    if not question or not isinstance(question, str) or not question.strip():
        return fragment

    if classify_claim_form(fragment) == "declarative":
        return fragment

    clean_frag = fragment.strip().rstrip(".!?")
    q = question.strip().rstrip("?. ")

    # Case 1: "What is/are/was/were [the] X?" -> "The X is/are {fragment}."
    m_copula = re.match(r"^(?:what|which)\s+(is|are|was|were)\s+(?:the\s+)?(.+)", q, re.IGNORECASE)
    if m_copula:
        copula = m_copula.group(1).lower()
        subj = m_copula.group(2).strip()
        # Adjust copula if fragment is a plural list
        if ("," in clean_frag or " and " in clean_frag) and copula == "is":
            copula = "are"
        return f"The {subj} {copula} {clean_frag}."

    # Case 2: "What [Subject] [Verb]...?" -> "{Subject} {Verb} ... is {fragment}."
    m_what = re.match(r"^what\s+(.+)", q, re.IGNORECASE)
    if m_what:
        rest = m_what.group(1).strip()
        return f"Regarding what {rest}, it is {clean_frag}."

    # Case 3: "Why does/did/do [Subject] [Verb]...?" -> "[Subject] [Verb] ... {fragment}."
    m_why = re.match(r"^why\s+(?:does|did|do|is|are)\s+(.+)", q, re.IGNORECASE)
    if m_why:
        rest = m_why.group(1).strip()
        connector = "" if clean_frag.lower().startswith(("to ", "because", "by ", "in order to")) else "because "
        return f"{rest} {connector}{clean_frag}."

    # Case 4: "How does/do/did [Subject] [Verb]...?" -> "[Subject] [Verb] ... by/with {fragment}."
    m_how = re.match(r"^how\s+(?:does|do|did|can)\s+(.+)", q, re.IGNORECASE)
    if m_how:
        rest = m_how.group(1).strip()
        connector = "" if clean_frag.lower().startswith(("by ", "through ", "with ", "using ", "via ")) else "by "
        return f"{rest} {connector}{clean_frag}."

    return f"Regarding {q}, the answer is {clean_frag}."


class ClaimDecomposer:
    """Deterministic sentence-level claim decomposer."""

    def __init__(self, min_claim_length: Optional[int] = None) -> None:
        """Initialize the ClaimDecomposer.

        Args:
            min_claim_length: Minimum character length for a fragment to be treated as a claim.
                              Defaults to settings.GROUNDING_MIN_CLAIM_LENGTH.
        """
        self.min_claim_length = (
            min_claim_length
            if min_claim_length is not None
            else settings.GROUNDING_MIN_CLAIM_LENGTH
        )

    def decompose(self, text: Optional[str]) -> List[ExtractedClaim]:
        """Decompose text into normalized, indexed sentence claims.

        Args:
            text: Input string (e.g., generated answer from FLAN-T5).

        Returns:
            List of ExtractedClaim objects with deterministic 0-based indices and claim_form.
        """
        if text is None or not isinstance(text, str):
            return []

        cleaned = text.strip()
        if not cleaned:
            return []

        # Split on sentence boundaries and newlines
        raw_fragments = SENTENCE_SPLIT_REGEX.split(cleaned)

        claims: List[ExtractedClaim] = []
        current_index = 0

        for fragment in raw_fragments:
            # Normalize whitespace: collapse internal tabs/spaces/newlines into single space
            norm = " ".join(fragment.strip().split())

            # Discard empty fragments and fragments shorter than minimum length
            if not norm or len(norm) < self.min_claim_length:
                continue

            form = classify_claim_form(norm)
            claims.append(
                ExtractedClaim(
                    claim_index=current_index,
                    claim_text=norm,
                    claim_form=form,
                )
            )
            current_index += 1

        logger.debug(
            "Decomposed text into %d claim(s) (min_length=%d).",
            len(claims),
            self.min_claim_length,
        )
        return claims
