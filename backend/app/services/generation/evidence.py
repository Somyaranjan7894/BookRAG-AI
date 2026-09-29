"""Evidence builder and context budgeting mechanism for BookRAG AI Phase 8.

Constructs grounded prompts with explicit passage boundaries, enforces model token limits,
and preserves complete chunk provenance.
"""

from typing import Any, List, Optional, Sequence, Tuple
from transformers import PreTrainedTokenizer

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.generation import GenerationEvidenceItem
from app.services.generation.exceptions import ContextBudgetExceededError, InvalidGenerationEvidenceError

logger = get_logger(__name__)

PROMPT_TEMPLATE = (
    "Answer the question using only the provided context.\n\n"
    "Context:\n"
    "{context}\n\n"
    "Question:\n"
    "{question}\n\n"
    "Answer:"
)


class BuiltPrompt:
    """Encapsulates an assembled grounded prompt and its included evidence provenance."""

    def __init__(
        self,
        prompt: str,
        included_evidence: List[GenerationEvidenceItem],
        total_tokens: int,
    ) -> None:
        self.prompt = prompt
        self.included_evidence = included_evidence
        self.total_tokens = total_tokens


class EvidenceBuilder:
    """Formats evidence passages with clear boundaries and budgets context tokens."""

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_input_tokens: Optional[int] = None,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        """Initialize EvidenceBuilder with tokenizer and token budget.

        Args:
            tokenizer: PreTrainedTokenizer used to measure exact prompt token counts.
            max_input_tokens: Total token ceiling for prompt + context (default: 2048).
            prompt_template: Grounded template string with {context} and {question} placeholders.
        """
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens or settings.GENERATION_MAX_INPUT_TOKENS
        self.prompt_template = prompt_template

    def count_tokens(self, text: str) -> int:
        """Calculate token count using tokenizer if available, else approximate via word ratio."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Conservative approximation: 1 token ≈ 4 characters or ~0.75 words
        return max(1, int(len(text) / 3.5))

    def build_prompt(
        self,
        question: str,
        evidence: Sequence[Any],
    ) -> BuiltPrompt:
        """Assemble a grounded prompt adhering strictly to the context token budget.

        Algorithm:
        1. Calculate base template overhead with empty context and the user question.
        2. Verify base prompt fits within max_input_tokens.
        3. Iteratively add high-ranked evidence chunks formatted with '[Page X] <text>'.
        4. When a chunk exceeds remaining budget:
           - If no evidence is included yet, truncate the top chunk safely to fit.
           - If higher-ranked evidence is already included, stop adding further chunks.
        5. Return BuiltPrompt containing the finalized prompt and full provenance items.

        Args:
            question: Cleaned natural language question.
            evidence: Sequence of SearchResult or dict objects in descending relevance order.

        Returns:
            BuiltPrompt with the assembled string, included evidence items, and token usage.

        Raises:
            ContextBudgetExceededError: If question alone exceeds max_input_tokens.
            InvalidGenerationEvidenceError: If evidence contains unparseable objects.
        """
        # 1. Base overhead computation
        empty_prompt = self.prompt_template.format(context="", question=question)
        base_tokens = self.count_tokens(empty_prompt)

        if base_tokens >= self.max_input_tokens:
            raise ContextBudgetExceededError(
                f"Question and prompt overhead ({base_tokens} tokens) exceeds total input budget "
                f"({self.max_input_tokens} tokens)."
            )

        remaining_budget = self.max_input_tokens - base_tokens
        formatted_blocks: List[str] = []
        included_evidence: List[GenerationEvidenceItem] = []

        for rank_idx, chunk in enumerate(evidence, start=1):
            item = self._normalize_evidence_item(chunk, rank=rank_idx)
            clean_text = item.source_text.strip()
            if not clean_text:
                continue

            block_text = f"[Page {item.page_number}] {clean_text}"
            separator_cost = self.count_tokens("\n\n") if formatted_blocks else 0
            block_tokens = self.count_tokens(block_text) + separator_cost

            if block_tokens <= remaining_budget:
                formatted_blocks.append(block_text)
                included_evidence.append(item)
                remaining_budget -= block_tokens
            else:
                # If no chunks fit at all, truncate this top chunk to utilize available budget
                if not formatted_blocks and remaining_budget > 20:
                    truncated_text = self._truncate_text_to_budget(
                        prefix=f"[Page {item.page_number}] ",
                        text=clean_text,
                        token_budget=remaining_budget,
                    )
                    if truncated_text:
                        formatted_blocks.append(truncated_text)
                        # Keep full item metadata with notice of truncation in source_text
                        truncated_item = item.model_copy(update={"source_text": clean_text})
                        included_evidence.append(truncated_item)
                # Stop adding lower-ranked chunks once budget is filled
                break

        context_str = "\n\n".join(formatted_blocks)
        full_prompt = self.prompt_template.format(context=context_str, question=question)
        total_tokens = self.count_tokens(full_prompt)

        return BuiltPrompt(
            prompt=full_prompt,
            included_evidence=included_evidence,
            total_tokens=total_tokens,
        )

    def _truncate_text_to_budget(
        self,
        prefix: str,
        text: str,
        token_budget: int,
    ) -> str:
        """Carefully truncate a long text passage so prefix + text fits within token_budget."""
        prefix_tokens = self.count_tokens(prefix)
        allowed_text_tokens = max(5, token_budget - prefix_tokens)

        if self.tokenizer is not None:
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            truncated_tokens = text_tokens[:allowed_text_tokens]
            decoded_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True).strip()
            return f"{prefix}{decoded_text}..."

        # Character fallback
        char_limit = int(allowed_text_tokens * 3.5)
        return f"{prefix}{text[:char_limit]}..."

    @staticmethod
    def _normalize_evidence_item(chunk: Any, rank: int) -> GenerationEvidenceItem:
        """Convert various chunk input types into a standardized GenerationEvidenceItem."""
        try:
            if isinstance(chunk, dict):
                return GenerationEvidenceItem(
                    rank=rank,
                    chunk_id=str(chunk.get("chunk_id", f"chunk_{rank}")),
                    document_id=str(chunk.get("document_id", "unknown")),
                    page_number=int(chunk.get("page_number", 1)),
                    chunk_index=int(chunk.get("chunk_index", 0)),
                    source_text=str(chunk.get("text", "") or chunk.get("source_text", "")),
                    similarity_score=float(chunk["similarity_score"]) if chunk.get("similarity_score") is not None else None,
                    reranker_score=float(chunk["reranker_score"]) if chunk.get("reranker_score") is not None else None,
                )
            # SearchResult or object
            return GenerationEvidenceItem(
                rank=rank,
                chunk_id=str(getattr(chunk, "chunk_id", f"chunk_{rank}")),
                document_id=str(getattr(chunk, "document_id", "unknown")),
                page_number=int(getattr(chunk, "page_number", 1)),
                chunk_index=int(getattr(chunk, "chunk_index", 0)),
                source_text=str(getattr(chunk, "text", "") or getattr(chunk, "source_text", "")),
                similarity_score=float(chunk.similarity_score) if getattr(chunk, "similarity_score", None) is not None else None,
                reranker_score=float(chunk.reranker_score) if getattr(chunk, "reranker_score", None) is not None else None,
            )
        except Exception as exc:
            raise InvalidGenerationEvidenceError(
                f"Failed to normalize evidence chunk at rank {rank}: {exc}",
                details=str(exc),
            ) from exc
