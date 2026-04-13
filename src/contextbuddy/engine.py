from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .budget import BudgetEnforcer
from .chunking import Chunker
from .embedder import LocalHashEmbedder
from .entities import EntityExtractor
from .pricing import OPENAI_GPT4O_MINI
from .scoring import SemanticScorer
from .tokenizer import HeuristicTokenizer
from .types import CostEstimate, Embedder, ModelPricing, Tokenizer


@dataclass(frozen=True)
class ContextEngineConfig:
    max_context_tokens: int = 4000
    min_relevance: float = 0.15
    dev_mode: bool = False
    rich_output: bool = True
    pricing: ModelPricing = field(default_factory=lambda: OPENAI_GPT4O_MINI)
    include_entities_section: bool = True
    chunk_min_chars: int = 40


@dataclass
class ContextReport:
    original_prompt_tokens: int
    final_prompt_tokens: int
    original_context_tokens: int
    final_context_tokens: int
    reduction_pct: float
    estimated_savings: float
    kept_chunks: int
    total_chunks: int
    entities: List[str]
    selected_indices: List[int]


def _default_cost(tokens_in: int, tokens_out: int, pricing: ModelPricing) -> CostEstimate:
    inc = (tokens_in / 1000.0) * pricing.input_per_1k
    outc = (tokens_out / 1000.0) * pricing.output_per_1k
    return CostEstimate(input_cost=inc, output_cost=outc)


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    low = text.lower()
    for n in needles:
        if not n:
            continue
        if n.lower() in low:
            return True
    return False


class ContextEngine:
    def __init__(
        self,
        config: ContextEngineConfig = ContextEngineConfig(),
        *,
        embedder: Optional[Embedder] = None,
        tokenizer: Optional[Tokenizer] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ):
        self.config = config
        self.embedder = embedder or LocalHashEmbedder()
        self.tokenizer = tokenizer or HeuristicTokenizer()
        self.entity_extractor = entity_extractor or EntityExtractor()

        self._chunker = Chunker(min_chars=config.chunk_min_chars)
        self._scorer = SemanticScorer(embedder=self.embedder)
        self._budget = BudgetEnforcer(tokenizer=self.tokenizer)

        self.last_report: Optional[ContextReport] = None

    def build_prompt(
        self,
        *,
        user_prompt: str,
        context: Union[str, Sequence[str]],
    ) -> Tuple[str, ContextReport]:
        chunks = self._chunker.chunk(context)
        total_chunks = len(chunks)

        # Original token estimate (context only)
        original_context_text = "\n\n".join(chunks)
        original_context_tokens = self._budget.count_tokens(original_context_text) if chunks else 0
        original_prompt_tokens = (
            self._budget.count_tokens(f"Context:\n{original_context_text}\n\nUser:\n{user_prompt}\n")
            if chunks
            else self._budget.count_tokens(user_prompt)
        )

        if not chunks:
            final_prompt = user_prompt
            report = ContextReport(
                original_prompt_tokens=original_prompt_tokens,
                final_prompt_tokens=original_prompt_tokens,
                original_context_tokens=original_context_tokens,
                final_context_tokens=0,
                reduction_pct=0.0,
                estimated_savings=0.0,
                kept_chunks=0,
                total_chunks=0,
                entities=[],
                selected_indices=[],
            )
            return final_prompt, report

        scores = self._scorer.score(query=user_prompt, chunks=chunks)

        # Initial pruning by relevance threshold
        keep_by_score = [s >= self.config.min_relevance for s in scores]

        # Entities from prompt + top chunks (by score)
        ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
        top_text = "\n\n".join(chunks[i] for i in ranked[: min(8, len(ranked))])
        entities = self.entity_extractor.extract(user_prompt + "\n" + top_text)

        # Keep list guardrail: any chunk containing an entity survives pruning
        keep_by_entity = [_contains_any(ch, entities) for ch in chunks]
        keep_mask = [a or b for a, b in zip(keep_by_score, keep_by_entity)]

        # If pruning kills everything, keep the top chunk.
        if not any(keep_mask):
            keep_mask[ranked[0]] = True

        pruned_chunks = [ch for ch, keep in zip(chunks, keep_mask) if keep]
        pruned_scores = [s for s, keep in zip(scores, keep_mask) if keep]
        pruned_keep_mask = [keep for keep in keep_mask if keep]  # all True
        pruned_indices = [i for i, keep in enumerate(keep_mask) if keep]

        # Enforce token budget on pruned set (preserve entity chunks first)
        entity_keep_mask = [_contains_any(ch, entities) for ch in pruned_chunks]
        selected_chunks, selected_local_indices = self._budget.enforce(
            chunks=pruned_chunks,
            scores=pruned_scores,
            keep_mask=entity_keep_mask,
            max_context_tokens=self.config.max_context_tokens,
        )
        selected_indices = [pruned_indices[i] for i in selected_local_indices]

        final_context = "\n\n".join(selected_chunks).strip()
        final_context_tokens = self._budget.count_tokens(final_context) if final_context else 0

        # Compose final prompt
        parts: List[str] = []
        if self.config.include_entities_section and entities:
            parts.append("KeyEntities:\n- " + "\n- ".join(entities))
        if final_context:
            parts.append("Context:\n" + final_context)
        parts.append("User:\n" + user_prompt.strip())
        final_prompt = "\n\n".join(parts).strip() + "\n"
        final_prompt_tokens = self._budget.count_tokens(final_prompt) if final_prompt else 0

        reduction_pct = 0.0
        if original_prompt_tokens > 0:
            reduction_pct = max(
                0.0, (original_prompt_tokens - final_prompt_tokens) / original_prompt_tokens * 100.0
            )

        cost_before = _default_cost(original_prompt_tokens, 0, self.config.pricing).total_cost
        cost_after = _default_cost(final_prompt_tokens, 0, self.config.pricing).total_cost
        estimated_savings = max(0.0, cost_before - cost_after)

        report = ContextReport(
            original_prompt_tokens=original_prompt_tokens,
            final_prompt_tokens=final_prompt_tokens,
            original_context_tokens=original_context_tokens,
            final_context_tokens=final_context_tokens,
            reduction_pct=reduction_pct,
            estimated_savings=estimated_savings,
            kept_chunks=len(selected_chunks),
            total_chunks=total_chunks,
            entities=entities,
            selected_indices=selected_indices,
        )
        return final_prompt, report

    def _emit_report(self, report: ContextReport) -> None:
        if not self.config.dev_mode:
            return
        if self.config.rich_output:
            from .telemetry import print_report
            print_report(report)
        else:
            before = report.original_prompt_tokens
            after = report.final_prompt_tokens
            import sys
            sys.stderr.write(
                f"[ContextBuddy] {before} \u2192 {after} tokens "
                f"({report.reduction_pct:.1f}% reduction). "
                f"Est. savings: ${report.estimated_savings:.4f}\n"
            )

    def run(
        self,
        *,
        user_prompt: str,
        context: Union[str, Sequence[str]],
        llm_call: Callable[[str], Any],
    ) -> Any:
        final_prompt, report = self.build_prompt(user_prompt=user_prompt, context=context)
        self.last_report = report
        self._emit_report(report)
        return llm_call(final_prompt)

    async def arun(
        self,
        *,
        user_prompt: str,
        context: Union[str, Sequence[str]],
        llm_call: Callable[[str], Awaitable[Any]],
    ) -> Any:
        final_prompt, report = self.build_prompt(user_prompt=user_prompt, context=context)
        self.last_report = report
        self._emit_report(report)
        return await llm_call(final_prompt)

