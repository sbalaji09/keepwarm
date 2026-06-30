"""Compactors — naive vs cost-aware. Gate 0 spike.

Mechanics only; real summarization is out of scope here. The deterministic
`fake_summarize` is enough to prove the layout/cost-accounting story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from keepwarm.context import is_summary_block
from keepwarm.contracts import (
    Block,
    CostModel,
    RenderedPrompt,
    default_token_estimator,
)


@dataclass
class CompactionCandidate:
    start_index: int          # inclusive — first block of the batch
    end_index: int            # exclusive — one past the last block of the batch
    saved_tokens: int         # tokens removed (batch_tokens - summary_tokens), recurring
    destroyed_tokens: int     # tokens after `end_index` that lose cache from this rewrite


@dataclass
class CompactionDecision:
    fired: bool
    candidate: CompactionCandidate | None
    new_prompt: RenderedPrompt
    reason: str


class Compactor(Protocol):
    def compact(self, prompt: RenderedPrompt) -> CompactionDecision: ...


# ---------- deterministic fake summarizer ----------


def fake_summarize(batch: list[Block]) -> Block:
    """Build a deterministic summary block. Real LLM summarization comes later."""
    content = {"summary": f"summary of {len(batch)} blocks", "replaces": len(batch)}
    return Block(
        zone="tail",
        role="system",
        content=content,
        token_count=default_token_estimator(content),
    )


def _eligible_indices(prompt: RenderedPrompt) -> list[int]:
    """Indices of blocks the compactor is allowed to touch.

    Stable blocks are off-limits (that's the cached prefix). Existing summary
    blocks are off-limits (never rewrite a summary).
    """
    return [
        i for i, b in enumerate(prompt)
        if b.zone != "stable" and not is_summary_block(b)
    ]


# ---------- naive: oldest-first, rewrites the middle, no gate ----------


class NaiveOldestFirstCompactor:
    """Strawman. Always fires when there is enough eligible content.

    Picks the *oldest* eligible blocks → modification lands at a low index →
    everything after it loses cache on the next prefill. Used to demonstrate
    the trap the cost-aware variant avoids.
    """

    def __init__(self, cost_model: CostModel, batch_size: int = 4) -> None:
        self.cost_model = cost_model
        self.batch_size = batch_size

    def compact(self, prompt: RenderedPrompt) -> CompactionDecision:
        eligible = _eligible_indices(prompt)
        if len(eligible) < self.batch_size:
            return CompactionDecision(
                fired=False, candidate=None, new_prompt=list(prompt),
                reason=f"not enough eligible blocks ({len(eligible)} < {self.batch_size})",
            )
        target = eligible[: self.batch_size]
        start, end = target[0], target[-1] + 1
        batch = list(prompt[start:end])
        summary = fake_summarize(batch)
        new_prompt = list(prompt[:start]) + [summary] + list(prompt[end:])
        saved = sum(b.token_count for b in batch) - summary.token_count
        destroyed = sum(b.token_count for b in prompt[end:])
        return CompactionDecision(
            fired=True,
            candidate=CompactionCandidate(start, end, saved, destroyed),
            new_prompt=new_prompt,
            reason=f"naive oldest-first: replaced blocks [{start}:{end}]",
        )


# ---------- cost-aware: newest-first, prefix-preserving, gated ----------


class CostAwareNewestFirstCompactor:
    """The Gate 0 candidate.

    - Picks the *newest* eligible batch so the cached prefix stays byte-identical.
    - Fires only when prefill saved exceeds prefill destroyed under the cost model.
    - Never rewrites an existing summary block.
    """

    def __init__(self, cost_model: CostModel, batch_size: int = 4) -> None:
        self.cost_model = cost_model
        self.batch_size = batch_size

    def compact(self, prompt: RenderedPrompt) -> CompactionDecision:
        eligible = _eligible_indices(prompt)
        if len(eligible) < self.batch_size:
            return CompactionDecision(
                fired=False, candidate=None, new_prompt=list(prompt),
                reason=f"not enough eligible blocks ({len(eligible)} < {self.batch_size})",
            )
        target = eligible[-self.batch_size:]
        start, end = target[0], target[-1] + 1
        batch = list(prompt[start:end])
        summary = fake_summarize(batch)
        saved_tokens = sum(b.token_count for b in batch) - summary.token_count
        destroyed_tokens = sum(b.token_count for b in prompt[end:])

        # Dollarize via the cost model. `saved` is recurring uncached-input we no
        # longer have to send; `destroyed` is one-shot cached-prefix loss on the
        # next prefill (those blocks fall out of cache because the prefix changed).
        saved_dollars = self.cost_model.price(
            cached_tokens=0, uncached_tokens=max(saved_tokens, 0)
        )
        destroyed_dollars = self.cost_model.price(
            cached_tokens=destroyed_tokens, uncached_tokens=0
        )
        candidate = CompactionCandidate(start, end, saved_tokens, destroyed_tokens)

        if saved_dollars <= destroyed_dollars:
            return CompactionDecision(
                fired=False, candidate=candidate, new_prompt=list(prompt),
                reason=(
                    f"refused: saved ${saved_dollars:.2f} "
                    f"<= destroyed ${destroyed_dollars:.2f} "
                    f"(saved_tokens={saved_tokens}, destroyed_tokens={destroyed_tokens})"
                ),
            )

        new_prompt = list(prompt[:start]) + [summary] + list(prompt[end:])
        return CompactionDecision(
            fired=True,
            candidate=candidate,
            new_prompt=new_prompt,
            reason=(
                f"fired: saved ${saved_dollars:.2f} > destroyed ${destroyed_dollars:.2f}"
            ),
        )
