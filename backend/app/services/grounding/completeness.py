"""Answer completeness validation for Phase 21 Query-Type-Aware Generation.

Performs generic, deterministic completeness checks on generated answers to verify
that all requested components (such as both sides of a comparison or all parts of
a multi-part question) are addressed before accepting the answer.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.schemas.query_plan import QueryPlan, QueryType

logger = get_logger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "how", "why", "where", "when", "can", "could", "will", "would",
    "terms", "terms of", "between", "difference", "differences", "compare",
}


class CompletenessCheckResult(BaseModel):
    """Result of an answer completeness check."""

    is_complete: bool = Field(description="Whether all requested components are addressed in the answer.")
    missing_components: List[str] = Field(default_factory=list, description="List of unaddressed question components.")
    reason: Optional[str] = Field(default=None, description="Deterministic reason explaining any completeness failure.")

    @property
    def missing_aspects(self) -> List[str]:
        return self.missing_components


class AnswerCompletenessChecker:
    """Verifies that generated answers thoroughly cover all requested query components."""

    def check_completeness(self, query_plan: Optional[QueryPlan], answer: Optional[str]) -> CompletenessCheckResult:
        """Evaluate whether the generated answer contains all required components.

        Args:
            query_plan: Structured QueryPlan containing query_type, comparison_aspects, and sub_questions.
            answer: Raw generated answer text from the generation service.

        Returns:
            CompletenessCheckResult with pass/fail status and missing components.
        """
        if not answer or not isinstance(answer, str) or not answer.strip():
            return CompletenessCheckResult(
                is_complete=False,
                missing_components=["full_answer"],
                reason="Generated answer is empty or whitespace-only.",
            )

        if query_plan is None:
            return CompletenessCheckResult(is_complete=True)

        lower_answer = answer.lower()

        # 1. Comparison Completeness Check
        if query_plan.query_type == QueryType.COMPARISON or query_plan.comparison_aspects:
            aspects = query_plan.comparison_aspects
            if len(aspects) >= 2:
                missing: List[str] = []
                aspect_tokens_map = {
                    asp: [
                        t.lower().strip()
                        for t in re.findall(r"\b[a-zA-Z0-9\+\-\*]+\b", asp)
                        if t.lower().strip() not in STOPWORDS
                    ]
                    for asp in aspects
                }
                for aspect in aspects:
                    clean_asp = aspect.lower().strip()
                    if clean_asp in lower_answer:
                        continue

                    tokens = aspect_tokens_map[aspect]
                    other_tokens = set()
                    for other_asp, other_toks in aspect_tokens_map.items():
                        if other_asp != aspect:
                            other_tokens.update(other_toks)

                    unique_toks = [t for t in tokens if t not in other_tokens and len(t) >= 1]
                    if unique_toks:
                        found = any(tok in lower_answer for tok in unique_toks)
                    else:
                        found = any(tok in lower_answer for tok in tokens if len(tok) > 2)

                    if not found:
                        missing.append(aspect)

                if missing:
                    logger.info("Completeness check failed for comparison: missing %s", missing)
                    missing_str = ", ".join(f"'{m}'" for m in missing)
                    return CompletenessCheckResult(
                        is_complete=False,
                        missing_components=missing,
                        reason=f"Generated answer does not address comparison aspect(s): {', '.join(missing)}; missing coverage of {missing_str}.",
                    )

        # 2. Multi-Part Question Completeness Check
        if query_plan.query_type in (QueryType.MULTI_PART, QueryType.MULTI_PAGE) or query_plan.sub_questions:
            sub_qs = query_plan.sub_questions
            if len(sub_qs) >= 2:
                missing_parts: List[str] = []
                for part in sub_qs:
                    part_tokens = [
                        t.lower().strip()
                        for t in re.findall(r"\b[a-zA-Z0-9\+\-\*]+\b", part)
                        if t.lower().strip() not in STOPWORDS and len(t) > 2
                    ]
                    if not part_tokens:
                        continue

                    # Require at least one distinctive keyword from the sub-question
                    found = any(tok in lower_answer for tok in part_tokens)
                    if not found:
                        missing_parts.append(part)

                if missing_parts:
                    logger.info("Completeness check failed for multi-part query: missing %s", missing_parts)
                    return CompletenessCheckResult(
                        is_complete=False,
                        missing_components=missing_parts,
                        reason=f"Generated answer is missing requested question part(s): {', '.join(missing_parts)}.",
                    )

        return CompletenessCheckResult(is_complete=True)
