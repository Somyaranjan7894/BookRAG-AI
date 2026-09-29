"""Deterministic Query Planner for Phase 11 Query Understanding.

Implements rule-based query normalization, classification, entity/term extraction,
constraint detection, and focused retrieval query generation.
No external LLM or heavyweight models are used.
"""

import re
from typing import List, Optional, Set, Tuple

from app.schemas.query_plan import (
    ExpectedAnswerType,
    QueryConstraints,
    QueryPlan,
    QueryType,
)
from app.services.query_understanding.exceptions import InvalidQueryError

# Common stopwords to exclude from standalone entity extraction
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "how", "why", "where", "when", "can", "could", "will", "would",
    "should", "may", "might", "must", "give", "tell", "explain", "describe",
    "compare", "list", "summarize", "define", "name", "show",
}


class QueryPlanner:
    """Deterministic, rule-based query planner producing structured QueryPlan contracts."""

    def normalize(self, query: str) -> str:
        """Safely normalize whitespace and formatting noise without losing semantic tokens.

        Preserves numbers, years, hyphens, and technical symbols like C++, COVID-19, R&D.
        """
        if query is None or not isinstance(query, str):
            raise InvalidQueryError("Query must be a non-empty string.")

        # Replace non-breaking spaces and tabs with standard space
        text = query.replace("\u00a0", " ").replace("\t", " ")

        # Collapse repeated whitespace
        text = re.sub(r"[ \f\v\r\n]+", " ", text).strip()

        if not text:
            raise InvalidQueryError("Query cannot be empty or whitespace-only.")

        return text

    def classify(self, query: str) -> Tuple[QueryType, str]:
        """Classify query intent taxonomy and expected answer type using conservative rules."""
        lower = query.lower().strip()

        # 1. Multi-hop indicators (compositional dependent reasoning)
        if (
            re.search(r"\b(mother|father|founder|inventor|author|creator|advisor|teacher)\s+of\s+the\s+(founder|inventor|author|creator)\b", lower)
            or re.search(r"\bwho\s+was\s+the\s+.*\s+of\s+the\s+(founder|author|inventor)\b", lower)
            or re.search(r"\bwhat\s+happened\s+in\s+the\s+year\s+that\b", lower)
            or re.search(r"\bwhich\s+company\s+acquired\s+the\s+firm\s+that\b", lower)
        ):
            return QueryType.MULTI_HOP, ExpectedAnswerType.EXPLANATION.value

        # 2. Comparison queries
        if (
            re.search(r"\bcompare\b", lower)
            or re.search(r"\b(difference|differences)\s+between\b", lower)
            or re.search(r"\b(vs\.?|versus)\b", lower)
            or re.search(r"\bhow\s+does\s+.*\s+(differ\s+from|compare\s+to)\b", lower)
        ):
            return QueryType.COMPARISON, ExpectedAnswerType.COMPARISON.value

        # 3. Summary queries
        if (
            re.search(r"^summarize\b", lower)
            or re.search(r"\b(summary|overview|synopsis)\s+(of|for)\b", lower)
            or re.search(r"^give\s+a\s+summary\b", lower)
        ):
            return QueryType.SUMMARY, ExpectedAnswerType.SUMMARY.value

        # 4. Procedural queries
        if (
            re.search(r"^how\s+(do|can|does|to|would)\s+(you|one|we)?\s*(perform|implement|calculate|compute|train|build|create|derive|evaluate|run|execute|use)\b", lower)
            or re.search(r"^how\s+to\s+", lower)
            or re.search(r"\b(steps?|procedure|algorithm|process)\s+(to|for|of)\b", lower)
        ):
            return QueryType.PROCEDURAL, ExpectedAnswerType.PROCEDURE.value

        # 5. Causal queries
        if (
            re.search(r"^why\b", lower)
            or re.search(r"\b(causes?|reasons?)\s+(of|for|behind)\b", lower)
            or re.search(r"^what\s+(caused|causes)\b", lower)
            or re.search(r"^how\s+did\s+.*\s+(cause|lead\s+to|result\s+in)\b", lower)
        ):
            return QueryType.CAUSAL, ExpectedAnswerType.EXPLANATION.value

        # 6. List queries
        if (
            re.search(r"^list\b", lower)
            or re.search(r"\b(name|list)\s+(the|all|some)\b", lower)
            or re.search(r"^what\s+are\s+(the\s+)?(main|primary|key|different|various|major)\s+(types|components|methods|examples|features|elements|approaches|factors)\b", lower)
        ):
            return QueryType.LIST, ExpectedAnswerType.LIST.value

        # 7. Definition queries
        if (
            re.search(r"^what\s+(is|are|was|were)\s+(a|an|the)?\s*[\w\s\+\-\*]+(?:\?|\.|$)", lower)
            and not re.search(r"\b(causes?|reasons?|difference|steps?|procedure|chapter|page)\b", lower)
        ) or re.search(r"^define\b", lower) or re.search(r"^what\s+does\s+.*\s+mean\b", lower):
            return QueryType.DEFINITION, ExpectedAnswerType.DEFINITION.value

        # 8. Location queries
        if re.search(r"^where\b", lower) or re.search(r"\bwhich\s+(chapter|page|section)\b", lower):
            return QueryType.LOCATION, ExpectedAnswerType.LOCATION.value

        # 9. Factual queries
        if re.search(r"^who\b", lower):
            return QueryType.FACTUAL, ExpectedAnswerType.PERSON_ENTITY.value

        if re.search(r"^when\b", lower) or re.search(r"\b(what|in\s+which)\s+year\b", lower):
            return QueryType.FACTUAL, ExpectedAnswerType.DATE_YEAR.value

        if re.search(r"^how\s+(many|much)\b", lower):
            return QueryType.FACTUAL, ExpectedAnswerType.NUMBER.value

        if re.search(r"^(which|what|did|does|is|was|were|has|have)\b", lower):
            return QueryType.FACTUAL, ExpectedAnswerType.EXPLANATION.value

        # 10. Unknown / Ambiguous fallback
        return QueryType.UNKNOWN, ExpectedAnswerType.EXPLANATION.value

    def extract_constraints(self, query: str) -> QueryConstraints:
        """Extract explicit constraints (chapter, year, page, quoted phrases, named entities).

        Constraints are ONLY added when explicitly stated in query text.
        """
        chapter: Optional[int] = None
        year: Optional[int] = None
        page_number: Optional[int] = None
        page_range: Optional[Tuple[int, int]] = None
        quoted_phrases: List[str] = []
        named_entities: List[str] = []

        # 1. Quoted phrases ("..." or '...')
        for m in re.finditer(r'["\']([^"\']{2,})["\']', query):
            quoted = m.group(1).strip()
            if quoted:
                quoted_phrases.append(quoted)

        # 2. Explicit chapter constraint: "Chapter 7", "Ch. 4", "chapter 12"
        chap_match = re.search(r"\b(?:chapter|ch\.?)\s+(\d+)\b", query, re.IGNORECASE)
        if chap_match:
            try:
                chapter = int(chap_match.group(1))
            except ValueError:
                pass

        # 3. Explicit page range or single page: "pages 10-15", "page 12"
        page_range_match = re.search(r"\bpages?\s+(\d+)\s*(?:-|to)\s*(\d+)\b", query, re.IGNORECASE)
        if page_range_match:
            try:
                p1, p2 = int(page_range_match.group(1)), int(page_range_match.group(2))
                page_range = (min(p1, p2), max(p1, p2))
            except ValueError:
                pass
        else:
            page_single_match = re.search(r"\bpages?\s+(\d+)\b", query, re.IGNORECASE)
            if page_single_match:
                try:
                    page_number = int(page_single_match.group(1))
                except ValueError:
                    pass

        # 4. Explicit calendar year: "in 2020", "during 1998", "since 2015", or standalone 4-digit year
        year_match = re.search(r"\b(?:in|during|year|since|before|after|by)?\s*\b(1[89]\d\d|20\d\d)\b", query, re.IGNORECASE)
        if year_match:
            try:
                year = int(year_match.group(1))
            except ValueError:
                pass

        # 5. Explicit named entities (multi-word capitalized proper names)
        # Avoid leading question words at the start of sentence
        trimmed_query = re.sub(r"^(Who|What|Where|When|Why|How|Which|Compare|Summarize|List|Define)\b", "", query, flags=re.IGNORECASE)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", trimmed_query):
            ent = m.group(1).strip()
            if ent.lower() not in STOPWORDS and ent not in named_entities:
                named_entities.append(ent)

        return QueryConstraints(
            chapter=chapter,
            year=year,
            page_number=page_number,
            page_range=page_range,
            quoted_phrases=quoted_phrases,
            named_entities=named_entities,
        )

    def extract_entities(self, query: str) -> List[str]:
        """Extract important entities, technical terms, years, and quoted terms.

        Conservative deterministic heuristic; not an external ML NER model.
        """
        entities: List[str] = []

        # 1. Quoted terms
        for m in re.finditer(r'["\']([^"\']{2,})["\']', query):
            quoted = m.group(1).strip()
            if quoted and quoted not in entities:
                entities.append(quoted)

        # 2. Technical symbols and terms: C++, COVID-19, R&D, A*
        for m in re.finditer(r"\b(?:C\+\+|COVID-19|R&D|A\*|[A-Z]{2,}(?:-[A-Z0-9]+)?)\b", query):
            tok = m.group(0).strip()
            if tok and tok not in entities:
                entities.append(tok)

        # 3. Capitalized proper nouns and entity names
        # Exclude common question starters
        words = query.split()
        for idx, word in enumerate(words):
            clean_w = re.sub(r"[^\w\+\-\*]", "", word)
            if not clean_w:
                continue
            if idx == 0 and clean_w.lower() in STOPWORDS:
                continue
            if clean_w[0].isupper() and clean_w.lower() not in STOPWORDS:
                if clean_w not in entities:
                    entities.append(clean_w)

        # 4. Multi-word phrases like "India and China", "Neural Networks"
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
            phrase = m.group(1).strip()
            if phrase.lower() not in STOPWORDS and phrase not in entities:
                entities.append(phrase)

        # 5. Distinct 4-digit years
        for m in re.finditer(r"\b(1[89]\d\d|20\d\d)\b", query):
            yr = m.group(1)
            if yr not in entities:
                entities.append(yr)

        return entities

    def generate_retrieval_queries(
        self,
        normalized_query: str,
        query_type: QueryType,
        entities: List[str],
        constraints: QueryConstraints,
    ) -> List[str]:
        """Generate 1 to 3 targeted, non-redundant retrieval queries.

        Rules:
        - Normally 1 query (normalized query).
        - Multi-part or comparison queries generate focused variations.
        - Maximum 3 queries.
        - No redundant queries.
        - Preserve important entities and constraints.
        """
        # Default single query
        default_query = normalized_query.rstrip("?.!")

        # 1. Comparison Queries: Compare A and B [in Y]
        if query_type == QueryType.COMPARISON:
            comp_match = re.search(
                r"(?:compare\s+(?:the\s+)?(?:[a-zA-Z\s]+\s+of\s+)?|difference\s+between\s+)(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\s+in\s+|\s+during\s+|\?|\.|$)",
                normalized_query,
                re.IGNORECASE,
            )
            if comp_match:
                sub_a = comp_match.group(1).strip().rstrip("?.!")
                sub_b = comp_match.group(2).strip().rstrip("?.!")

                # Extract aspect (e.g. population) if present
                aspect_match = re.search(
                    r"compare\s+(?:the\s+)?([a-zA-Z]+)\s+of\b",
                    normalized_query,
                    re.IGNORECASE,
                )
                aspect = aspect_match.group(1) if aspect_match else ""

                year_suffix = f" {constraints.year}" if constraints.year else ""
                aspect_suffix = f" {aspect}" if aspect else ""

                q1 = f"{sub_a}{aspect_suffix}{year_suffix}".strip()
                q2 = f"{sub_b}{aspect_suffix}{year_suffix}".strip()

                if q1 and q2 and q1.lower() != q2.lower():
                    return [q1, q2]

        # 2. Dual-aspect Causal/Consequence: causes and consequences of X
        if query_type in (QueryType.CAUSAL, QueryType.LIST, QueryType.FACTUAL):
            cause_eff_match = re.search(
                r"(?:causes?\s+and\s+(?:consequences?|effects?|impacts?)|(?:consequences?|effects?|impacts?)\s+and\s+causes?)\s+(?:of\s+)?(.+?)(?:\?|\.|$)",
                normalized_query,
                re.IGNORECASE,
            )
            if cause_eff_match:
                target = cause_eff_match.group(1).strip().rstrip("?.!")
                if target:
                    q1 = f"causes of {target}"
                    q2 = f"consequences of {target}"
                    return [q1, q2]

        # Default fallback: single controlled retrieval query
        return [default_query]

    def plan(self, query: str) -> QueryPlan:
        """Analyze query and construct complete QueryPlan contract."""
        normalized = self.normalize(query)
        q_type, expected_ans = self.classify(normalized)
        constraints = self.extract_constraints(normalized)
        entities = self.extract_entities(normalized)
        retrieval_queries = self.generate_retrieval_queries(
            normalized_query=normalized,
            query_type=q_type,
            entities=entities,
            constraints=constraints,
        )

        # Enforce max 3 retrieval queries and deduplicate
        deduped_retrieval_queries: List[str] = []
        for rq in retrieval_queries:
            clean_rq = rq.strip()
            if clean_rq and clean_rq not in deduped_retrieval_queries:
                deduped_retrieval_queries.append(clean_rq)
            if len(deduped_retrieval_queries) >= 3:
                break

        if not deduped_retrieval_queries:
            deduped_retrieval_queries = [normalized]

        requires_multi_evidence = (
            len(deduped_retrieval_queries) > 1
            or q_type in (QueryType.COMPARISON, QueryType.LIST, QueryType.MULTI_HOP, QueryType.SUMMARY)
        )

        return QueryPlan(
            original_query=query,
            normalized_query=normalized,
            query_type=q_type,
            entities=entities,
            constraints=constraints,
            retrieval_queries=deduped_retrieval_queries,
            expected_answer_type=expected_ans,
            requires_multiple_evidence=requires_multi_evidence,
        )
