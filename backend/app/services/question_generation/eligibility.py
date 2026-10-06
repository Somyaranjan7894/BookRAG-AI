"""Document-structure eligibility for Question Generation.

Decides whether a chunk is suitable source material for generating study questions.

Design principles (validated against a real 420-page textbook, see
``docs/QUESTION_GENERATION_QUALITY_REPORT.md``):

* Section headings are only recognised at the *start* of a chunk (the "lead"),
  optionally preceded by a page number / running header. Earlier versions matched
  keywords anywhere in the chunk, which rejected technical text such as
  "An index (e.g., an integer) is assigned to each node" as INDEX and accepted
  real back-of-book index pages as technical content.
* Content-shape detectors recognise structures that carry no heading in a given
  chunk: bibliography entries, back-of-book index entries, publishing metadata,
  acknowledgement name lists and extraction garbage.
* ``classify_chunks`` adds document-order context so that continuation chunks of
  a preface / acknowledgements / references section inherit the section label.

No book-specific strings are used; all signals are generic publishing structure.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.schemas.chunk import Chunk


@dataclass
class EligibilityResult:
    eligible: bool
    category: str
    score: Optional[float] = None
    reason: Optional[str] = None


# Categories
FRONT_MATTER = "FRONT_MATTER"
TITLE_PAGE = "TITLE_PAGE"
COPYRIGHT = "COPYRIGHT"
TABLE_OF_CONTENTS = "TABLE_OF_CONTENTS"
PREFACE = "PREFACE"
FOREWORD = "FOREWORD"
ACKNOWLEDGEMENTS = "ACKNOWLEDGEMENTS"
AUTHOR_BIO = "AUTHOR_BIO"
ABSTRACT = "ABSTRACT"
INTRODUCTION = "INTRODUCTION"
TECHNICAL_CONTENT = "TECHNICAL_CONTENT"
EXAMPLE = "EXAMPLE"
CASE_STUDY = "CASE_STUDY"
EXERCISE = "EXERCISE"
CONCLUSION = "CONCLUSION"
REFERENCES = "REFERENCES"
BIBLIOGRAPHY = "BIBLIOGRAPHY"
INDEX = "INDEX"
APPENDIX = "APPENDIX"
METADATA = "METADATA"
NAME_LIST = "NAME_LIST"
LOW_TEXT_QUALITY = "LOW_TEXT_QUALITY"
LOW_TECHNICAL_DENSITY = "LOW_TECHNICAL_DENSITY"
UNKNOWN = "UNKNOWN"

# Minimum technical density for an INTRODUCTION chunk to remain eligible.
INTRODUCTION_MIN_DENSITY = 0.18

# ---------------------------------------------------------------------------
# Lead (heading) detection
# ---------------------------------------------------------------------------
# Optional leading page number (arabic or roman) before a heading, e.g. "xvi PREFACE".
_LEAD_PREFIX = r"^[\s\W]*(?:(?:\d{1,4}|[ivxlcdm]{1,8})\s+)?"
_LEAD_CHARS = 120

# (category, uppercase keyword regex, titlecase keyword regex)
_HEADING_RULES = [
    (TABLE_OF_CONTENTS, r"(?:TABLE\s+OF\s+)?CONTENTS", r"(?:Table\s+of\s+)?Contents"),
    (COPYRIGHT, r"COPYRIGHT", r"Copyright"),
    (PREFACE, r"PREFACE", r"Preface"),
    (FOREWORD, r"FOREWORD", r"Foreword"),
    (ACKNOWLEDGEMENTS, r"ACKNOWLEDG(?:E)?MENTS?", r"Acknowledg(?:e)?ments?"),
    (AUTHOR_BIO, r"ABOUT\s+THE\s+AUTHORS?", r"About\s+the\s+Authors?"),
    (ABSTRACT, r"ABSTRACT", r"Abstract"),
    (REFERENCES, r"REFERENCES", r"References"),
    (BIBLIOGRAPHY, r"BIBLIOGRAPHY", r"Bibliography"),
    (INDEX, r"INDEX", r"Index"),
    (APPENDIX, r"APPENDIX", r"Appendix"),
    (EXERCISE, r"EXERCISES?", r"Exercises?"),
    (CASE_STUDY, r"CASE\s+STUDY", r"Case\s+Study"),
    (EXAMPLE, r"EXAMPLES?", r"Example\s+\d"),
    (CONCLUSION, r"(?:CONCLUSIONS?|SUMMARY)", r"(?:Conclusions?|Summary)"),
    (INTRODUCTION, r"INTRODUCTION", r"Introduction"),
]

_COMPILED_HEADINGS = [
    (
        cat,
        # Uppercase heading word at chunk start (case-sensitive).
        re.compile(_LEAD_PREFIX + r"(?:CHAPTER\s+[\dIVXL]+\s+)?" + upper + r"\b"),
        # Titlecase heading must be followed by a newline or heading punctuation.
        re.compile(_LEAD_PREFIX + r"(?:\d+(?:\.\d+)*\.?\s+)?" + title + r"\s*(?:\n|[:.\u2014\-]\s|$)"),
    )
    for cat, upper, title in _HEADING_RULES
]

# Start of a new chapter / numbered section ends inherited front/back-matter context.
_SECTION_START_RE = re.compile(
    r"^[\s\W]*(?:\d{1,4}\s+)?(?:CHAPTER|Chapter)\s+[\dIVXL]+\b|^[\s\W]*\d+(?:\.\d+)+\s+[A-Z]"
)

_STRONG_EXCLUSIONS = {
    TITLE_PAGE, COPYRIGHT, TABLE_OF_CONTENTS, PREFACE, FOREWORD, ACKNOWLEDGEMENTS,
    AUTHOR_BIO, ABSTRACT, REFERENCES, BIBLIOGRAPHY, INDEX,
}

_CATEGORY_REASON = {
    TITLE_PAGE: "title_page_content",
    COPYRIGHT: "copyright_content",
    TABLE_OF_CONTENTS: "toc_content",
    PREFACE: "preface_content",
    FOREWORD: "foreword_content",
    ACKNOWLEDGEMENTS: "acknowledgements_content",
    AUTHOR_BIO: "author_bio_content",
    ABSTRACT: "abstract_content",
    REFERENCES: "references_content",
    BIBLIOGRAPHY: "bibliography_content",
    INDEX: "index_content",
    METADATA: "publishing_metadata",
    NAME_LIST: "name_list_content",
    LOW_TEXT_QUALITY: "low_text_quality",
    LOW_TECHNICAL_DENSITY: "low_technical_density",
}

# ---------------------------------------------------------------------------
# Content-shape detectors
# ---------------------------------------------------------------------------
_PUBLISHING_MARKERS = re.compile(
    r"\b(?:ISBN(?:-1[03])?|Product\s+Details|Hardcover|Paperback|Publisher|"
    r"Library\s+of\s+Congress|All\s+rights\s+reserved|Printed\s+in|"
    r"Dimensions\s*\(in\s+inches\)|Shipping\s+Weight|Customer\s+Reviews|Editorial\s+Reviews|"
    r"Sales\s+Rank|Book\s+Info|Book\s+News|Cataloging[- ]in[- ]Publication|"
    r"First\s+published|Reprinted|ASIN|Average\s+Customer\s+Review)\b",
    re.IGNORECASE,
)
_CITE_AUTHOR = re.compile(r"\b[A-Z][A-Za-z'\-]+,\s(?:[A-Z]\.\s?){1,3}")
_CITE_YEAR = re.compile(r"\((?:19|20)\d{2}[a-z]?\)\.")
_CITE_PAGES = re.compile(r"\bpp\.\s*\d+")
_CITE_VOLUME = re.compile(r"\b\d+\(\d+\),\s*\d+")
_INDEX_REF = re.compile(r",\s*\d{1,4}(?:\s*[-\u2013]\s*\d{1,4})?n?(?![\d.])")
_NAME_SEGMENT = re.compile(r"^(?:and\s+)?[A-Z][a-z'\-]+(?:\s+[A-Z]\.)?(?:\s+(?:van|von|de|da|del|le)?\s*[A-Z][A-Za-z'\-]+){1,2}$")

# Simple stop-word list for technical density estimation
_STOP_WORDS = {
    "the", "and", "or", "but", "if", "else", "when", "where", "how",
    "what", "why", "who", "which", "is", "are", "was", "were", "to",
    "of", "in", "for", "on", "by", "with", "a", "an", "as", "that",
    "this", "these", "those",
}


def _technical_density(text: str) -> float:
    """Return a rough technical-content density score between 0 and 1.

    The score is the proportion of *content* words that are longer than 5 characters
    and are not in a generic stop-word list.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    content_words = [w for w in words if w not in _STOP_WORDS]
    technical_words = [w for w in content_words if len(w) > 5]
    return len(technical_words) / max(len(content_words), 1)


def citation_score(text: str) -> int:
    """Count bibliography-entry signals (author initials, '(1995).', 'pp. 12', '6(2), 192')."""
    return (
        len(_CITE_AUTHOR.findall(text))
        + len(_CITE_YEAR.findall(text))
        + len(_CITE_PAGES.findall(text))
        + len(_CITE_VOLUME.findall(text))
    )


def index_ref_count(text: str) -> int:
    """Count back-of-book index page references such as 'Entropy, 55-57, 282'."""
    return len(_INDEX_REF.findall(text))


def name_list_count(text: str) -> int:
    """Count comma-separated segments that look like personal names."""
    segments = [s.strip() for s in re.split(r"[,;]", text)]
    return sum(1 for s in segments if _NAME_SEGMENT.match(s))


def publishing_marker_count(text: str) -> int:
    """Count distinct publishing / retail metadata markers."""
    return len({m.lower() for m in _PUBLISHING_MARKERS.findall(text)})


def _alpha_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    alpha = [t for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z'\-]{2,}[.,;:]?", t)]
    return len(alpha) / len(tokens)


def _has_section_heading(chunk: Chunk) -> Optional[str]:
    """Return an explicit section heading from chunk metadata when the ingestion pipeline provides one."""
    if isinstance(chunk.metadata, dict):
        sec = chunk.metadata.get("section") or chunk.metadata.get("heading")
        if sec:
            return str(sec).strip().lower()
    return None


def detect_lead_heading(text: str) -> Optional[str]:
    """Detect a section heading at the start of the chunk text. Returns the category or None."""
    lead = text[:_LEAD_CHARS]
    for cat, upper_re, title_re in _COMPILED_HEADINGS:
        if upper_re.search(lead) or title_re.search(lead):
            return cat
    return None


def _category_from_metadata_heading(heading: str) -> Optional[str]:
    h = heading.lower()
    mapping = [
        ("title page", TITLE_PAGE), ("copyright", COPYRIGHT), ("table of contents", TABLE_OF_CONTENTS),
        ("contents", TABLE_OF_CONTENTS), ("preface", PREFACE), ("foreword", FOREWORD),
        ("acknowledg", ACKNOWLEDGEMENTS), ("about the author", AUTHOR_BIO), ("abstract", ABSTRACT),
        ("references", REFERENCES), ("bibliography", BIBLIOGRAPHY), ("index", INDEX),
        ("appendix", APPENDIX), ("exercise", EXERCISE), ("case study", CASE_STUDY),
        ("example", EXAMPLE), ("conclusion", CONCLUSION), ("summary", CONCLUSION),
        ("introduction", INTRODUCTION),
    ]
    for key, cat in mapping:
        if h == key or h.startswith(key):
            return cat
    return None


def _reject(cat: str, score: Optional[float] = None) -> EligibilityResult:
    return EligibilityResult(False, cat, score=score, reason=_CATEGORY_REASON.get(cat, cat.lower()))


def is_eligible(chunk: Chunk) -> EligibilityResult:
    """Classify a chunk and decide whether it is eligible for question generation.

    Policy:
    * Strongly exclude front matter, title pages, copyright, TOC, preface, foreword,
      acknowledgements, author bio, abstract, references, bibliography, index,
      publishing metadata, name lists and garbled extraction output.
    * Usually allow technical content, examples, case studies, exercises, conclusions,
      appendix.
    * Introduction is allowed only when it contains sufficient technical density.
    """
    text = (chunk.text or "").strip()
    density = _technical_density(text)
    words = text.split()

    # 1. Explicit metadata heading supplied by the ingestion pipeline
    heading = _has_section_heading(chunk)
    category = _category_from_metadata_heading(heading) if heading else None

    # 2. Heading at the very start of the chunk
    if category is None:
        category = detect_lead_heading(text)

    if category in _STRONG_EXCLUSIONS:
        return _reject(category, density)

    # 3. Content-shape detectors (independent of headings)
    if publishing_marker_count(text) >= 2:
        return _reject(METADATA, density)
    n_words = max(len(words), 1)
    cites = citation_score(text)
    if cites >= 8 or (cites >= 5 and cites * 100.0 / n_words >= 4.0):
        return _reject(REFERENCES, density)
    idx_refs = index_ref_count(text)
    if idx_refs >= 10 and idx_refs * 100.0 / n_words >= 8.0:
        return _reject(INDEX, density)
    if name_list_count(text) >= 8:
        return _reject(NAME_LIST, density)
    # Short but legitimate technical passages should still qualify; we only reject
    # obviously low-value text or malformed extraction output.
    if len(words) < 6 and not re.search(r"\b[A-Za-z]{4,}\b", text):
        return _reject(LOW_TEXT_QUALITY, density)
    if len(words) >= 6 and len(words) < 12 and _alpha_ratio(text) < 0.35 and len({w for w in words if len(w) > 3}) < 5:
        return _reject(LOW_TEXT_QUALITY, density)
    if len(words) >= 12 and _alpha_ratio(text) < 0.45:
        return _reject(LOW_TEXT_QUALITY, density)

    if category is None:
        category = TECHNICAL_CONTENT

    if category == INTRODUCTION:
        if density < INTRODUCTION_MIN_DENSITY:
            return EligibilityResult(False, INTRODUCTION, score=density, reason="low_technical_density")
        return EligibilityResult(True, INTRODUCTION, score=density, reason="eligible_introduction")
    return EligibilityResult(True, category, score=density, reason="eligible")


# Sections whose continuation chunks (without their own heading) inherit the label.
_CARRY_SECTIONS = {PREFACE, FOREWORD, ACKNOWLEDGEMENTS, AUTHOR_BIO, ABSTRACT, TABLE_OF_CONTENTS, COPYRIGHT}
_CARRY_MAX_PAGES = 2
_BACK_MATTER_SECTIONS = {REFERENCES, BIBLIOGRAPHY, INDEX}


def classify_chunks(chunks: Sequence[Chunk]) -> List[EligibilityResult]:
    """Classify chunks in document order using ``is_eligible`` plus section context.

    ``is_eligible`` remains the per-chunk source of truth. Context only *adds*
    rejections for continuation chunks:

    * Front-matter sections (preface, acknowledgements, ...) carry forward for at most
      ``_CARRY_MAX_PAGES`` pages until a chapter / numbered-section start is seen.
    * Back-matter sections (references, index) carry forward while the chunk still shows
      weaker evidence of the same structure (hysteresis).

    Results are returned in the same order as the input sequence.
    """
    order = sorted(range(len(chunks)), key=lambda i: (chunks[i].page_number, chunks[i].chunk_index or 0))
    results: List[Optional[EligibilityResult]] = [None] * len(chunks)

    active: Optional[str] = None
    active_page = 0
    for i in order:
        chunk = chunks[i]
        res = is_eligible(chunk)
        text = (chunk.text or "").strip()
        own_heading = detect_lead_heading(text) or _has_section_heading(chunk)

        if res.eligible and active is not None and not own_heading and not _SECTION_START_RE.search(text[:_LEAD_CHARS]):
            if active in _CARRY_SECTIONS and chunk.page_number - active_page <= _CARRY_MAX_PAGES:
                res = EligibilityResult(False, active, score=res.score, reason=_CATEGORY_REASON[active] + "_continuation")
            elif active in (REFERENCES, BIBLIOGRAPHY) and citation_score(text) >= 3:
                res = EligibilityResult(False, active, score=res.score, reason=_CATEGORY_REASON[active] + "_continuation")
            elif active == INDEX and index_ref_count(text) >= 5:
                res = EligibilityResult(False, active, score=res.score, reason="index_content_continuation")

        if not res.eligible and (res.category in _CARRY_SECTIONS or res.category in _BACK_MATTER_SECTIONS):
            if res.category != active:
                active_page = chunk.page_number
            active = res.category
        elif res.eligible:
            active = None
        results[i] = res

    return [r for r in results if r is not None]
