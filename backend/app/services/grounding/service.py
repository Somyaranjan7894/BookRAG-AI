"""Claim-level NLI Groundedness Validation Service for BookRAG AI Phase 9.

Evaluates whether individual claims decomposed from generated answers are strictly
supported by retrieved book evidence passages using Natural Language Inference (NLI).

CRITICAL DISTINCTION:
This service verifies:
"Is the generated answer supported by the retrieved evidence?"
It does NOT determine whether the book itself is objectively correct in the real world.
The groundedness_score is a system evaluation metric, NOT a factual truth probability.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.grounding import (
    ClaimEvidenceProvenance,
    ClaimResult,
    ExtractedClaim,
    GroundingReport,
)
from app.services.grounding.claims import ClaimDecomposer
from app.services.grounding.exceptions import (
    InvalidGroundingConfigError,
    InvalidGroundingInputError,
)
from app.services.grounding.model import NLIModel, NLIScores

logger = get_logger(__name__)


def _extract_evidence_item(evidence_obj: Any, rank: int) -> ClaimEvidenceProvenance:
    """Extract standard ClaimEvidenceProvenance from various evidence formats.

    Supports GenerationEvidenceItem, SearchResult, Chunk, or dict representations.
    """
    if hasattr(evidence_obj, "chunk_id"):
        chunk_id = str(evidence_obj.chunk_id)
        document_id = str(getattr(evidence_obj, "document_id", "unknown"))
        page_number = int(getattr(evidence_obj, "page_number", 1))
        chunk_index = int(getattr(evidence_obj, "chunk_index", 0))
        source_text = getattr(evidence_obj, "source_text", None) or getattr(evidence_obj, "text", "")
        sim_score = getattr(evidence_obj, "similarity_score", None)
        rerank_score = getattr(evidence_obj, "reranker_score", None)
        item_rank = getattr(evidence_obj, "rank", rank)
    elif isinstance(evidence_obj, dict):
        chunk_id = str(evidence_obj.get("chunk_id", "unknown"))
        document_id = str(evidence_obj.get("document_id", "unknown"))
        page_number = int(evidence_obj.get("page_number", 1))
        chunk_index = int(evidence_obj.get("chunk_index", 0))
        source_text = str(evidence_obj.get("source_text") or evidence_obj.get("text") or "")
        sim_score = evidence_obj.get("similarity_score")
        rerank_score = evidence_obj.get("reranker_score")
        item_rank = int(evidence_obj.get("rank", rank))
    else:
        raise InvalidGroundingInputError(f"Unsupported evidence item type: {type(evidence_obj)}")

    return ClaimEvidenceProvenance(
        chunk_id=chunk_id,
        document_id=document_id,
        page_number=page_number,
        chunk_index=chunk_index,
        source_text=source_text,
        rank=item_rank,
        similarity_score=float(sim_score) if sim_score is not None else None,
        reranker_score=float(rerank_score) if rerank_score is not None else None,
    )


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "how", "why", "where", "when", "can", "could", "will", "would",
    "regarding", "answer", "according", "terms", "question",
}


def _is_evidence_relevant_to_claim(evidence_text: str, claim_text: str) -> bool:
    """Determine whether an evidence chunk is semantically relevant to a claim.

    An evidence chunk can only contradict a claim if it shares topic/entity vocabulary.
    Out-of-domain or orthogonal passages (e.g. CNNs vs NLI) that share zero non-stopword
    substantive terms cannot logically contradict the claim.
    """
    import re
    ev_lower = evidence_text.lower()
    claim_tokens = [
        t.lower().strip()
        for t in re.findall(r"\b[a-zA-Z0-9\+\-\*]+\b", claim_text)
        if t.lower().strip() not in STOPWORDS and len(t) > 2
    ]
    if not claim_tokens:
        return True
    return any(tok in ev_lower for tok in claim_tokens)


def _check_evidence_contains_fragment(evidence_text: str, fragment_text: str) -> bool:
    """Verify whether a short answer fragment is supported by lexical containment or alignment."""
    import re
    ev_lower = evidence_text.lower()
    frag_lower = fragment_text.lower().strip().rstrip(".!?")

    # Direct substring match
    if frag_lower in ev_lower:
        return True

    # Constituent token containment (for lists or multi-word entities)
    frag_tokens = [
        t.strip()
        for t in re.findall(r"\b[a-zA-Z0-9\+\-\*]+\b", frag_lower)
        if t.strip() not in STOPWORDS and len(t) > 1
    ]
    if not frag_tokens:
        return False

    matched = sum(1 for tok in frag_tokens if tok in ev_lower)
    return (matched / len(frag_tokens)) >= 0.75


class GroundingService:
    """Validates generated claims against retrieved book evidence using CrossEncoder NLI."""

    def __init__(
        self,
        model: Optional[NLIModel] = None,
        decomposer: Optional[ClaimDecomposer] = None,
        entailment_threshold: Optional[float] = None,
        contradiction_threshold: Optional[float] = None,
        top_k_evidence: Optional[int] = None,
    ) -> None:
        """Initialize GroundingService.

        Args:
            model: Optional NLIModel wrapper instance (defaults to cached singleton).
            decomposer: Optional ClaimDecomposer instance.
            entailment_threshold: Minimum entailment score to consider a claim supported.
            contradiction_threshold: Minimum contradiction score to flag a contradiction.
            top_k_evidence: Maximum number of evidence chunks to evaluate per claim.
        """
        self._model = model
        self._decomposer = decomposer
        self.entailment_threshold = (
            entailment_threshold
            if entailment_threshold is not None
            else settings.GROUNDING_ENTAILMENT_THRESHOLD
        )
        self.contradiction_threshold = (
            contradiction_threshold
            if contradiction_threshold is not None
            else settings.GROUNDING_CONTRADICTION_THRESHOLD
        )
        self.top_k_evidence = (
            top_k_evidence
            if top_k_evidence is not None
            else settings.GROUNDING_TOP_K_EVIDENCE
        )

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate threshold parameters."""
        if not (0.0 <= self.entailment_threshold <= 1.0):
            raise InvalidGroundingConfigError(
                f"entailment_threshold must be between 0.0 and 1.0 (got {self.entailment_threshold})."
            )
        if not (0.0 <= self.contradiction_threshold <= 1.0):
            raise InvalidGroundingConfigError(
                f"contradiction_threshold must be between 0.0 and 1.0 (got {self.contradiction_threshold})."
            )
        if self.top_k_evidence <= 0:
            raise InvalidGroundingConfigError(
                f"top_k_evidence must be positive (got {self.top_k_evidence})."
            )

    @property
    def model(self) -> NLIModel:
        """Lazy load or return the injected NLIModel instance."""
        if self._model is None:
            self._model = NLIModel.get_instance()
        return self._model

    @property
    def decomposer(self) -> ClaimDecomposer:
        """Lazy load or return the injected ClaimDecomposer."""
        if self._decomposer is None:
            self._decomposer = ClaimDecomposer()
        return self._decomposer

    def validate_claims(
        self,
        claims: Sequence[ExtractedClaim],
        evidence: Sequence[Any],
        entailment_threshold: Optional[float] = None,
        contradiction_threshold: Optional[float] = None,
        top_k_evidence: Optional[int] = None,
        query: Optional[str] = None,
    ) -> GroundingReport:
        """Validate decomposed claims against retrieved evidence passages.

        NLI Semantics:
        premise    = evidence chunk text
        hypothesis = generated claim text (or contextualized declarative proposition for fragments)
        pair       = (premise, hypothesis)

        Claim Classification:
        - If strongest entailment >= entailment_threshold and contradiction < contradiction_threshold:
          status = "entailed"
        - If strongest contradiction >= contradiction_threshold and entailment < entailment_threshold:
          status = "contradicted"
        - If strongest entailment >= entailment_threshold and strongest contradiction >= contradiction_threshold:
          status = "conflicted"
        - Otherwise (neither threshold satisfied):
          status = "unsupported"

        Args:
            claims: Sequence of ExtractedClaim objects to validate.
            evidence: Sequence of evidence items.
            entailment_threshold: Optional threshold override.
            contradiction_threshold: Optional threshold override.
            top_k_evidence: Optional evidence chunk count override.
            query: Optional original question query used to contextualize non-sentential fragments.

        Returns:
            GroundingReport with claim-level validation outcomes and overall groundedness score.
        """
        eff_ent_thresh = (
            entailment_threshold if entailment_threshold is not None else self.entailment_threshold
        )
        eff_contra_thresh = (
            contradiction_threshold if contradiction_threshold is not None else self.contradiction_threshold
        )
        eff_top_k = (
            top_k_evidence if top_k_evidence is not None else self.top_k_evidence
        )

        # Edge case: No substantive claims provided
        if not claims:
            logger.info("Grounding validation called with zero claims.")
            return GroundingReport(
                groundedness_score=0.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                conflicted_claims=0,
                overall_status="empty",
                claim_results=[],
                reason="No substantive claims found to evaluate.",
            )

        # Standardize evidence chunks up to top_k
        selected_evidence: List[ClaimEvidenceProvenance] = [
            _extract_evidence_item(item, rank=idx + 1)
            for idx, item in enumerate(evidence[:eff_top_k])
        ]

        # Edge case: No evidence available
        if not selected_evidence:
            logger.info("Grounding validation called with empty evidence.")
            claim_results = [
                ClaimResult(
                    claim_index=c.claim_index,
                    claim_text=c.claim_text,
                    status="unsupported",
                    entailment_score=0.0,
                    contradiction_score=0.0,
                    neutral_score=1.0,
                    supporting_evidence=None,
                    contradicting_evidence=[],
                )
                for c in claims
            ]
            return GroundingReport(
                groundedness_score=0.0,
                total_claims=len(claims),
                supported_claims=0,
                unsupported_claims=len(claims),
                contradicted_claims=0,
                conflicted_claims=0,
                overall_status="unsupported",
                claim_results=claim_results,
                reason="No evidence passages available for validation.",
            )

        from app.services.grounding.claims import contextualize_fragment

        # Construct batch pairs: (premise, hypothesis) = (evidence_text, claim_text)
        pairs: List[Tuple[str, str]] = []
        pair_mapping: List[Tuple[int, int]] = []  # (claim_idx, evidence_idx)

        # Contextualized pairs for non-sentential fragments
        ctx_pairs: List[Tuple[str, str]] = []
        ctx_pair_mapping: List[Tuple[int, int]] = []

        for c_idx, claim in enumerate(claims):
            c_form = getattr(claim, "claim_form", "declarative")
            is_fragment = c_form in ("noun_phrase", "list", "numerical", "short_span")
            ctx_text = contextualize_fragment(query, claim.claim_text) if (is_fragment and query) else None

            for e_idx, ev in enumerate(selected_evidence):
                pairs.append((ev.source_text, claim.claim_text))
                pair_mapping.append((c_idx, e_idx))
                if ctx_text and ctx_text != claim.claim_text:
                    ctx_pairs.append((ev.source_text, ctx_text))
                    ctx_pair_mapping.append((c_idx, e_idx))

        # Batched NLI inference across all pairs
        logger.debug(
            "Executing batched NLI inference for %d base pairs and %d contextualized pairs...",
            len(pairs),
            len(ctx_pairs),
        )
        scores_list = self.model.predict(pairs)
        ctx_scores_list = self.model.predict(ctx_pairs) if ctx_pairs else []

        # Map predictions: (claim_idx, evidence_idx) -> NLIScores
        raw_score_map: Dict[Tuple[int, int], NLIScores] = {}
        for (c_idx, e_idx), sc in zip(pair_mapping, scores_list):
            raw_score_map[(c_idx, e_idx)] = sc

        ctx_score_map: Dict[Tuple[int, int], NLIScores] = {}
        for (c_idx, e_idx), sc in zip(ctx_pair_mapping, ctx_scores_list):
            ctx_score_map[(c_idx, e_idx)] = sc

        # Organize merged predictions per claim
        claim_evaluations: Dict[int, List[Tuple[ClaimEvidenceProvenance, NLIScores]]] = {
            i: [] for i in range(len(claims))
        }

        for c_idx, claim in enumerate(claims):
            c_form = getattr(claim, "claim_form", "declarative")
            is_fragment = c_form in ("noun_phrase", "list", "numerical", "short_span")

            for e_idx, ev_item in enumerate(selected_evidence):
                sc_raw = raw_score_map[(c_idx, e_idx)]
                sc_ctx = ctx_score_map.get((c_idx, e_idx))

                if sc_ctx is not None:
                    # Contextualized proposition removes non-sentential NLI failure
                    merged_ent = max(sc_raw.entailment, sc_ctx.entailment)
                    merged_contra = min(sc_raw.contradiction, sc_ctx.contradiction)
                    merged_neutral = min(sc_raw.neutral, sc_ctx.neutral)
                else:
                    merged_ent = sc_raw.entailment
                    merged_contra = sc_raw.contradiction
                    merged_neutral = sc_raw.neutral

                # Fragment containment verification: a fragment must be lexically verifiable in the evidence
                if is_fragment:
                    frag_supported = _check_evidence_contains_fragment(ev_item.source_text, claim.claim_text)
                    if not frag_supported and merged_ent >= eff_ent_thresh:
                        # If fragment is not present in this chunk, cannot claim entailment from this chunk
                        merged_ent = min(merged_ent, 0.20)

                claim_evaluations[c_idx].append(
                    (
                        ev_item,
                        NLIScores(
                            entailment=merged_ent,
                            contradiction=merged_contra,
                            neutral=merged_neutral,
                        ),
                    )
                )

        # Evaluate each claim against its candidate evidence chunks
        claim_results: List[ClaimResult] = []
        supported_count = 0
        unsupported_count = 0
        contradicted_count = 0
        conflicted_count = 0

        for c_idx, claim in enumerate(claims):
            evals = claim_evaluations[c_idx]

            # Find strongest entailment
            best_ent_score = -1.0
            best_ent_ev: Optional[ClaimEvidenceProvenance] = None
            best_ent_neutral = 0.0

            # Find strongest contradiction
            best_contra_score = -1.0
            contra_evidence_list: List[ClaimEvidenceProvenance] = []
            supporting_evidence_list: List[ClaimEvidenceProvenance] = []

            for ev_item, sc in evals:
                if sc.entailment > best_ent_score:
                    best_ent_score = sc.entailment
                    best_ent_ev = ev_item.model_copy(update={"nli_score": round(sc.entailment, 4)})
                    best_ent_neutral = sc.neutral

                if sc.entailment >= eff_ent_thresh:
                    supporting_evidence_list.append(
                        ev_item.model_copy(update={"nli_score": round(sc.entailment, 4)})
                    )

                # Relevance-Gated Contradiction:
                # An evidence chunk can only contradict a claim if it is semantically relevant to the claim topic.
                # Disjoint passages that share no content with the claim cannot logically contradict it.
                is_relevant = _is_evidence_relevant_to_claim(ev_item.source_text, claim.claim_text)
                if is_relevant:
                    if sc.contradiction > best_contra_score:
                        best_contra_score = sc.contradiction

                    if sc.contradiction >= eff_contra_thresh:
                        contra_evidence_list.append(
                            ev_item.model_copy(update={"nli_score": round(sc.contradiction, 4)})
                        )

            # Determine claim status based on thresholds
            has_entailment = best_ent_score >= eff_ent_thresh
            has_contradiction = best_contra_score >= eff_contra_thresh

            if has_entailment and not has_contradiction:
                status = "entailed"
                supported_count += 1
            elif has_contradiction and not has_entailment:
                status = "contradicted"
                contradicted_count += 1
            elif has_entailment and has_contradiction:
                status = "conflicted"
                conflicted_count += 1
            else:
                status = "unsupported"
                unsupported_count += 1

            # Only retain supporting evidence if entailment was meaningful, or best overall
            supporting_evidence = (
                best_ent_ev if best_ent_score >= eff_ent_thresh else best_ent_ev
            )

            claim_results.append(
                ClaimResult(
                    claim_index=claim.claim_index,
                    claim_text=claim.claim_text,
                    status=status,
                    entailment_score=round(max(best_ent_score, 0.0), 4),
                    contradiction_score=round(max(best_contra_score, 0.0), 4),
                    neutral_score=round(best_ent_neutral, 4),
                    supporting_evidence=supporting_evidence,
                    supporting_evidences=supporting_evidence_list if has_entailment else [],
                    contradicting_evidence=contra_evidence_list,
                )
            )

        total_claims = len(claims)
        groundedness_score = round(supported_count / total_claims, 4) if total_claims > 0 else 0.0

        # Determine overall status
        if contradicted_count > 0:
            overall_status = "contradicted"
            reason = f"Generated answer contains {contradicted_count} claim(s) contradicted by retrieved book evidence."
        elif conflicted_count > 0:
            overall_status = "conflicted"
            reason = f"Generated answer contains {conflicted_count} claim(s) with conflicting book evidence."
        elif unsupported_count > 0:
            overall_status = "unsupported"
            reason = f"Generated answer contains {unsupported_count} claim(s) not sufficiently supported by retrieved book evidence."
        elif supported_count == total_claims:
            overall_status = "grounded"
            reason = "All substantive claims are sufficiently supported by retrieved book evidence."
        else:
            overall_status = "unsupported"
            reason = "Generated answer is not fully supported by retrieved book evidence."

        return GroundingReport(
            groundedness_score=groundedness_score,
            total_claims=total_claims,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            contradicted_claims=contradicted_count,
            conflicted_claims=conflicted_count,
            overall_status=overall_status,
            claim_results=claim_results,
            reason=reason,
        )
