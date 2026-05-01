from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union

from .budget import BudgetEnforcer
from .chunking import SmartChunker
from .embedder import LocalHashEmbedder
from .entities import EntityExtractor
from .hybrid_scorer import HybridScorer
from .pricing import OPENAI_GPT4O_MINI
from .scoring import SemanticScorer
from .tokenizer import HeuristicTokenizer
from .types import CostEstimate, Embedder, ModelPricing, Tokenizer


@dataclass(frozen=True)
class ContextEngineConfig:
    max_context_tokens: int = 4000
    min_relevance: float = 0.15
    conservative_mode: bool = False
    dev_mode: bool = False
    rich_output: bool = True
    pricing: ModelPricing = field(default_factory=lambda: OPENAI_GPT4O_MINI)
    include_entities_section: bool = True
    # Playbook defaults (applied by ContextEngine). Chunker defaults stay
    # backwards-compatible for direct Chunker() use.
    chunk_min_chars: int = 100
    chunk_merge_under_chars: int = 200


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

@dataclass(frozen=True)
class CompressionEvent:
    """
    Streaming event for compression UX.

    This is intentionally lightweight: UIs can render the stage and the latest
    report numbers without needing internal chunk/scores details.
    """

    stage: str  # start | chunked | scored | selected | done
    message: str = ""
    report: Optional[ContextReport] = None
    prompt: Optional[str] = None


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


def _is_keep_worthy_context_entity(e: str) -> bool:
    """
    Context-derived entities are a safety net, but some are noisy in real PDFs:
    - header emails
    - DOI/URL fragments
    - random decimals that look like versions

    This filter keeps the playbook guarantee for IDs/dates/etc. while avoiding
    common hijacks.
    """
    s = (e or "").strip()
    if not s:
        return False
    low = s.lower()
    if "@" in s:
        return False
    if low.startswith("http://") or low.startswith("https://"):
        return False
    # Drop bare decimals like "2016.2573832" (common in DOIs/citations).
    if s.replace(".", "", 1).isdigit() and "." in s:
        return False
    return True


class ContextEngine:
    def __init__(
        self,
        config: ContextEngineConfig = ContextEngineConfig(),
        *,
        embedder: Optional[Embedder] = None,
        tokenizer: Optional[Tokenizer] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        scorer: Optional[object] = None,
    ):
        # Playbook: conservative mode keeps more chunks (lower risk of misses).
        if config.conservative_mode and config.min_relevance > 0.05:
            config = replace(config, min_relevance=0.05)
        self.config = config
        self.embedder = embedder or LocalHashEmbedder()
        self.tokenizer = tokenizer or HeuristicTokenizer()
        self.entity_extractor = entity_extractor or EntityExtractor()

        self._chunker = SmartChunker(
            min_chars=config.chunk_min_chars,
            merge_under_chars=config.chunk_merge_under_chars,
        )
        if scorer is not None:
            self._scorer = scorer
        elif embedder is not None:
            self._scorer = SemanticScorer(embedder=self.embedder)
        else:
            self._scorer = HybridScorer()
        self._budget = BudgetEnforcer(tokenizer=self.tokenizer)

        self.last_report: Optional[ContextReport] = None

    def _compress(
        self,
        *,
        user_prompt: str,
        context: Union[str, Sequence[str]],
    ) -> Tuple[str, ContextReport]:
        if isinstance(context, str):
            chunks = self._chunker.chunk(context, doc_type="auto")
        else:
            chunks = list(context)
        total_chunks = len(chunks)

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

        keep_by_score = [s >= self.config.min_relevance for s in scores]

        ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
        top_text = "\n\n".join(chunks[i] for i in ranked[: min(8, len(ranked))])
        prompt_entities = self.entity_extractor.extract(user_prompt)
        context_entities = self.entity_extractor.extract(top_text)
        seen_e: set[str] = set()
        entities: List[str] = []
        for e in [*prompt_entities, *context_entities]:
            if e and e not in seen_e:
                seen_e.add(e)
                entities.append(e)

        entities = [e for e in entities if (e in prompt_entities) or _is_keep_worthy_context_entity(e)]

        keep_entities = list(prompt_entities)
        keep_entities.extend([e for e in context_entities if _is_keep_worthy_context_entity(e)])

        entity_chunks = {i for i, ch in enumerate(chunks) if _contains_any(ch, keep_entities)} if keep_entities else set()
        entity_context: set[int] = set()
        for i in entity_chunks:
            entity_context.add(max(0, i - 1))
            entity_context.add(min(len(chunks) - 1, i + 1))
        keep_by_entity = [i in (entity_chunks | entity_context) for i in range(len(chunks))]
        keep_mask = [a or b for a, b in zip(keep_by_score, keep_by_entity)]

        if not any(keep_mask):
            keep_mask[ranked[0]] = True

        pruned_chunks = [ch for ch, keep in zip(chunks, keep_mask) if keep]
        pruned_scores = [s for s, keep in zip(scores, keep_mask) if keep]
        pruned_keep_mask = [kb for kb, keep in zip(keep_by_entity, keep_mask) if keep]
        pruned_indices = [i for i, keep in enumerate(keep_mask) if keep]

        entity_keep_mask = pruned_keep_mask
        selected_chunks, selected_local_indices = self._budget.enforce(
            chunks=pruned_chunks,
            scores=pruned_scores,
            keep_mask=entity_keep_mask,
            max_context_tokens=self.config.max_context_tokens,
        )
        selected_indices = [pruned_indices[i] for i in selected_local_indices]

        final_context = "\n\n".join(selected_chunks).strip()
        final_context_tokens = self._budget.count_tokens(final_context) if final_context else 0

        if final_context_tokens > original_context_tokens:
            raise RuntimeError(
                "ContextBuddy compressed context is larger than input context "
                f"({final_context_tokens} > {original_context_tokens} tokens). "
                "This is a bug — please file an issue."
            )

        parts: List[str] = []
        if self.config.include_entities_section and entities:
            parts.append("KeyEntities:\n- " + "\n- ".join(entities))
        if final_context:
            parts.append("Context:\n" + final_context)
        parts.append("User:\n" + user_prompt.strip())
        final_prompt = "\n\n".join(parts).strip() + "\n"
        final_prompt_tokens = self._budget.count_tokens(final_prompt) if final_prompt else 0

        if original_prompt_tokens > 0 and not final_prompt.strip():
            raise RuntimeError(
                "ContextBuddy produced empty output for non-empty input. "
                "This is a bug — please file an issue with the input that triggered it."
            )

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

    def build_prompt(
        self,
        *,
        user_prompt: str,
        context: Union[str, Sequence[str]],
    ) -> Tuple[str, ContextReport]:
        return self._compress(user_prompt=user_prompt, context=context)

    def build_prompt_stream(
        self,
        *,
        user_prompt: str,
        context: Union[str, Sequence[str]],
    ) -> Iterator[CompressionEvent]:
        yield CompressionEvent(stage="start", message="Starting compression")

        if isinstance(context, str):
            chunks = self._chunker.chunk(context, doc_type="auto")
        else:
            chunks = list(context)
        yield CompressionEvent(stage="chunked", message=f"Chunked into {len(chunks)} chunks")

        yield CompressionEvent(stage="scoring", message="Scoring chunks for relevance")

        final_prompt, report = self._compress(user_prompt=user_prompt, context=context)

        yield CompressionEvent(stage="scored", message="Scored and selected chunks under budget", report=report)
        yield CompressionEvent(stage="done", message="Compression complete", report=report, prompt=final_prompt)

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
        stream: bool = False,
    ) -> Any:
        final_prompt, report = self.build_prompt(user_prompt=user_prompt, context=context)
        self.last_report = report
        self._emit_report(report)

        if stream:
            return self._stream_sync(final_prompt, llm_call)
        return llm_call(final_prompt)

    def _stream_sync(self, prompt: str, llm_call: Callable[[str], Any]) -> Generator[Any, None, None]:
        response = llm_call(prompt)
        if hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
            yield from response
        else:
            yield response

    async def arun(
        self,
        *,
        user_prompt: str,
        context: Union[str, Sequence[str]],
        llm_call: Callable[[str], Awaitable[Any]],
        stream: bool = False,
    ) -> Any:
        """Async-compatible entry point for LLM calls with context compression.

        Note: compression (chunking, scoring, budget enforcement) runs synchronously
        inside this coroutine. For high-concurrency async workloads, wrap with
        ``asyncio.to_thread(engine.build_prompt, ...)``. True async compression
        is targeted for v0.4.0.
        """
        final_prompt, report = self.build_prompt(user_prompt=user_prompt, context=context)
        self.last_report = report
        self._emit_report(report)

        if stream:
            return self._stream_async(final_prompt, llm_call)
        return await llm_call(final_prompt)

    async def _stream_async(self, prompt: str, llm_call: Callable[[str], Awaitable[Any]]) -> AsyncIterator[Any]:
        response = await llm_call(prompt)
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                yield chunk
        elif hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
            for chunk in response:
                yield chunk
        else:
            yield response

