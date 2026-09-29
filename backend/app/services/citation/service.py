"""Citation and Provenance Mapping Service for BookRAG AI Phase 10.

Implements the core principle:
"The generator generates the answer. The system assigns citations from verified evidence."

Responsibilities:
1. Inspect Phase 9 NLI grounding validation results.
2. Filter for verified supporting evidence (never assign citations to unsupported claims).
3. Validate document isolation (evidence must match query document; reject cross-document leakage).
4. Deduplicate evidence chunks across claims using deterministic key (document_id, chunk_id).
5. Generate deterministic, response-local citation IDs (cite_1, cite_2, ...).
6. Build many-to-many claim citation references.
7. Preserve conflict and contradiction diagnostics (supporting vs. contradicting evidence).
8. Return structured, presentation-independent citation result.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from app.core.logging import get_logger
from app.schemas.citation import Citation, CitationMappingResult, ClaimCitationRef
from app.schemas.grounding import ClaimEvidenceProvenance, ClaimResult, GroundingReport
from app.services.citation.exceptions import (
    DocumentIsolationError,
    InvalidCitationInputError,
)

logger = get_logger(__name__)


class CitationService:
    """Service for constructing deterministic citations and provenance mappings from Phase 9 grounding results."""

    def build_citations(
        self,
        grounding_result: Union[GroundingReport, List[ClaimResult]],
        expected_document_id: Optional[str] = None,
    ) -> CitationMappingResult:
        """Map verified grounding evidence into deterministic, deduplicated citations.

        Args:
            grounding_result: Either a full GroundingReport or a list of ClaimResult items.
            expected_document_id: Optional document ID that all evidence must belong to.

        Returns:
            CitationMappingResult with unique top-level citations and claim citation references.

        Raises:
            InvalidCitationInputError: If grounding_result is None or malformed.
            DocumentIsolationError: If cross-document evidence or document mismatch is detected.
        """
        if grounding_result is None:
            raise InvalidCitationInputError("grounding_result cannot be None.")

        # Extract list of ClaimResult
        if isinstance(grounding_result, GroundingReport):
            claim_results = grounding_result.claim_results
        elif isinstance(grounding_result, list):
            claim_results = grounding_result
        else:
            raise InvalidCitationInputError(
                f"Expected GroundingReport or List[ClaimResult], got {type(grounding_result).__name__}."
            )

        if not claim_results:
            return CitationMappingResult(
                citations=[],
                claim_citations={},
                contradicting_claim_citations={},
            )

        # 1. First Pass: Validate Document Isolation across all evidence
        self._validate_document_isolation(claim_results, expected_document_id)

        # 2. Second Pass: Build deterministic deduplicated citations
        # Mapping from (document_id, chunk_id) to Citation
        seen_chunks: Dict[Tuple[str, str], Citation] = {}
        citations_list: List[Citation] = []
        claim_citations: Dict[int, List[ClaimCitationRef]] = {}
        contradicting_claim_citations: Dict[int, List[ClaimCitationRef]] = {}

        for claim in claim_results:
            claim_index = claim.claim_index
            status = getattr(claim, "status", getattr(claim, "grounding_status", "unsupported"))

            supporting_refs: List[ClaimCitationRef] = []
            contradicting_refs: List[ClaimCitationRef] = []

            # Rule 6: Unsupported claims receive NO authoritative citations
            if status == "unsupported":
                claim.citations = []
                claim.contradicting_citations = []
                claim_citations[claim_index] = []
                contradicting_claim_citations[claim_index] = []
                continue

            # Process verified supporting evidence for entailed or conflicted claims
            if status in ("entailed", "conflicted"):
                supporting_evidences = self._gather_supporting_evidence(claim)
                for evidence in supporting_evidences:
                    citation = self._get_or_create_citation(evidence, seen_chunks, citations_list)
                    ref = ClaimCitationRef(
                        citation_id=citation.citation_id,
                        document_id=citation.document_id,
                        chunk_id=citation.chunk_id,
                        page_number=citation.page_number,
                        chunk_index=citation.chunk_index,
                        source_text=citation.source_text,
                        relation="supports",
                    )
                    # Deduplicate refs within the same claim
                    if not any(r.citation_id == ref.citation_id for r in supporting_refs):
                        supporting_refs.append(ref)

            # Process contradicting evidence for conflicted or contradicted claims
            if status in ("contradicted", "conflicted"):
                contradicting_evidences = getattr(claim, "contradicting_evidence", []) or []
                for evidence in contradicting_evidences:
                    citation = self._get_or_create_citation(evidence, seen_chunks, citations_list)
                    ref = ClaimCitationRef(
                        citation_id=citation.citation_id,
                        document_id=citation.document_id,
                        chunk_id=citation.chunk_id,
                        page_number=citation.page_number,
                        chunk_index=citation.chunk_index,
                        source_text=citation.source_text,
                        relation="contradicts",
                    )
                    # Deduplicate refs within the same claim
                    if not any(r.citation_id == ref.citation_id for r in contradicting_refs):
                        contradicting_refs.append(ref)

            # Assign to claim object and dictionaries
            claim.citations = supporting_refs
            claim.contradicting_citations = contradicting_refs
            claim_citations[claim_index] = supporting_refs
            contradicting_claim_citations[claim_index] = contradicting_refs

        logger.info(
            "Built %d unique citations across %d claims.",
            len(citations_list),
            len(claim_results),
        )

        return CitationMappingResult(
            citations=citations_list,
            claim_citations=claim_citations,
            contradicting_claim_citations=contradicting_claim_citations,
        )

    def _gather_supporting_evidence(self, claim: ClaimResult) -> List[ClaimEvidenceProvenance]:
        """Extract supporting evidence chunks for a claim."""
        evidences: List[ClaimEvidenceProvenance] = []
        if hasattr(claim, "supporting_evidences") and claim.supporting_evidences:
            evidences.extend(claim.supporting_evidences)
        elif hasattr(claim, "supporting_evidence") and claim.supporting_evidence is not None:
            evidences.append(claim.supporting_evidence)
        return evidences

    def _get_or_create_citation(
        self,
        evidence: ClaimEvidenceProvenance,
        seen_chunks: Dict[Tuple[str, str], Citation],
        citations_list: List[Citation],
    ) -> Citation:
        """Retrieve existing citation for chunk or construct a new deterministic one."""
        key = (evidence.document_id, evidence.chunk_id)
        if key in seen_chunks:
            return seen_chunks[key]

        # Generate deterministic response-local ID: cite_1, cite_2, ...
        citation_id = f"cite_{len(citations_list) + 1}"

        citation = Citation(
            citation_id=citation_id,
            document_id=evidence.document_id,
            chunk_id=evidence.chunk_id,
            page_number=evidence.page_number,
            chunk_index=evidence.chunk_index,
            source_text=evidence.source_text,
            similarity_score=evidence.similarity_score,
            reranker_score=evidence.reranker_score,
            evidence_rank=evidence.rank,
        )

        seen_chunks[key] = citation
        citations_list.append(citation)
        return citation

    def _validate_document_isolation(
        self,
        claim_results: List[ClaimResult],
        expected_document_id: Optional[str] = None,
    ) -> None:
        """Ensure all supporting and contradicting evidence strictly respects document isolation."""
        observed_doc_ids: Set[str] = set()

        for claim in claim_results:
            status = getattr(claim, "status", getattr(claim, "grounding_status", "unsupported"))
            if status == "unsupported":
                continue

            # Gather all evidence items relevant for this claim
            all_evidences: List[ClaimEvidenceProvenance] = []
            if status in ("entailed", "conflicted"):
                all_evidences.extend(self._gather_supporting_evidence(claim))
            if status in ("contradicted", "conflicted"):
                all_evidences.extend(getattr(claim, "contradicting_evidence", []) or [])

            for ev in all_evidences:
                doc_id = ev.document_id.strip() if ev.document_id else ""
                if not doc_id:
                    continue

                if expected_document_id is not None and doc_id != expected_document_id.strip():
                    raise DocumentIsolationError(
                        f"Cross-document evidence detected. Evidence chunk '{ev.chunk_id}' has document_id "
                        f"'{doc_id}' which does not match expected document_id '{expected_document_id}'."
                    )
                observed_doc_ids.add(doc_id)

        # If no expected_document_id was supplied, all citations in the response must still belong to the same document
        if expected_document_id is None and len(observed_doc_ids) > 1:
            raise DocumentIsolationError(
                f"Inconsistent document IDs detected across citations: {sorted(list(observed_doc_ids))}. "
                "All citations in a response must belong to the single target document."
            )
