"""Compactors — naive vs cost-aware. Gate 0 spike.

Mechanics only; real summarization is out of scope here. The deterministic
`fake_summarize` is enough to prove the layout/cost-accounting story.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from keepwarm.context import is_active_tools_block, is_summary_block
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
    saved_tokens: int         # batch_tokens - summary_tokens, recurring per future call
    destroyed_tokens: int     # tokens after end_index whose cache is lost on next call
    batch_tokens: int = 0
    summary_tokens: int = 0
    saved_dollars: float = 0.0
    destroyed_dollars: float = 0.0


@dataclass
class CompactionDecision:
    fired: bool
    candidate: CompactionCandidate | None
    new_prompt: RenderedPrompt
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


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
    """Indices of blocks the compactor may touch.

    Default policy: only `tail` blocks are eligible, excluding existing summary
    blocks and the `active_tools` constraint block. The stable prefix and the
    volatile/memory zone are off-limits — touching them busts the cached
    prefix or destroys mid-run memory updates.
    """
    return [
        i for i, b in enumerate(prompt)
        if b.zone == "tail"
        and not is_summary_block(b)
        and not is_active_tools_block(b)
    ]


def _empty_decision(prompt: RenderedPrompt, reason: str) -> CompactionDecision:
    return CompactionDecision(
        fired=False, candidate=None, new_prompt=list(prompt), reason=reason,
        metrics={"eligible_count": len(_eligible_indices(prompt))},
    )


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
            return _empty_decision(
                prompt,
                f"not enough eligible blocks ({len(eligible)} < {self.batch_size})",
            )
        target = eligible[: self.batch_size]
        start, end = target[0], target[-1] + 1
        batch = list(prompt[start:end])
        summary = fake_summarize(batch)
        new_prompt = list(prompt[:start]) + [summary] + list(prompt[end:])
        batch_tokens = sum(b.token_count for b in batch)
        saved_tokens = batch_tokens - summary.token_count
        destroyed_tokens = sum(b.token_count for b in prompt[end:])
        saved_dollars = self.cost_model.price(
            cached_tokens=0, uncached_tokens=max(saved_tokens, 0)
        )
        destroyed_dollars = self.cost_model.price(
            cached_tokens=destroyed_tokens, uncached_tokens=0
        )
        candidate = CompactionCandidate(
            start, end, saved_tokens, destroyed_tokens,
            batch_tokens=batch_tokens, summary_tokens=summary.token_count,
            saved_dollars=saved_dollars, destroyed_dollars=destroyed_dollars,
        )
        return CompactionDecision(
            fired=True, candidate=candidate, new_prompt=new_prompt,
            reason=f"naive oldest-first: replaced blocks [{start}:{end}]",
            metrics={
                "strategy": "naive_oldest_first",
                "prefix_preserved_blocks": start,
                "prefix_preserved_tokens": sum(b.token_count for b in prompt[:start]),
            },
        )


# ---------- cost-aware: newest-first, prefix-preserving, gated ----------


class CostAwareNewestFirstCompactor:
    """The Gate 0 candidate.

    - Picks the *newest* eligible batch so the cached prefix stays byte-identical.
    - Fires only when prefill saved exceeds prefill destroyed under the cost model.
    - Refuses summaries that aren't meaningfully smaller than the batch
      (`min_saving_ratio`, default 0.25).
    - Never rewrites an existing summary block or the active_tools constraint.
    """

    def __init__(
        self,
        cost_model: CostModel,
        batch_size: int = 4,
        min_saving_ratio: float = 0.25,
    ) -> None:
        self.cost_model = cost_model
        self.batch_size = batch_size
        self.min_saving_ratio = min_saving_ratio

    def compact(self, prompt: RenderedPrompt) -> CompactionDecision:
        eligible = _eligible_indices(prompt)
        if len(eligible) < self.batch_size:
            return _empty_decision(
                prompt,
                f"not enough eligible blocks ({len(eligible)} < {self.batch_size})",
            )
        target = eligible[-self.batch_size:]
        start, end = target[0], target[-1] + 1
        batch = list(prompt[start:end])
        summary = fake_summarize(batch)
        batch_tokens = sum(b.token_count for b in batch)
        saved_tokens = batch_tokens - summary.token_count
        destroyed_tokens = sum(b.token_count for b in prompt[end:])

        saved_dollars = self.cost_model.price(
            cached_tokens=0, uncached_tokens=max(saved_tokens, 0)
        )
        destroyed_dollars = self.cost_model.price(
            cached_tokens=destroyed_tokens, uncached_tokens=0
        )
        candidate = CompactionCandidate(
            start, end, saved_tokens, destroyed_tokens,
            batch_tokens=batch_tokens, summary_tokens=summary.token_count,
            saved_dollars=saved_dollars, destroyed_dollars=destroyed_dollars,
        )
        base_metrics = {
            "strategy": "cost_aware_newest_first",
            "prefix_preserved_blocks": start,
            "prefix_preserved_tokens": sum(b.token_count for b in prompt[:start]),
            "saving_ratio": (saved_tokens / batch_tokens) if batch_tokens else 0.0,
        }

        # Quality gate: refuse when the summary isn't meaningfully smaller than
        # the batch (rounding/encoding noise, mostly).
        if batch_tokens > 0 and (saved_tokens / batch_tokens) < self.min_saving_ratio:
            return CompactionDecision(
                fired=False, candidate=candidate, new_prompt=list(prompt),
                reason=(
                    f"refused: saving ratio "
                    f"{saved_tokens / batch_tokens:.2f} "
                    f"< min {self.min_saving_ratio:.2f}"
                ),
                metrics=base_metrics,
            )

        # Cost gate: refuse when destroyed prefill cost would exceed savings.
        if saved_dollars <= destroyed_dollars:
            return CompactionDecision(
                fired=False, candidate=candidate, new_prompt=list(prompt),
                reason=(
                    f"refused: saved ${saved_dollars:.6f} "
                    f"<= destroyed ${destroyed_dollars:.6f} "
                    f"(saved_tokens={saved_tokens}, destroyed_tokens={destroyed_tokens})"
                ),
                metrics=base_metrics,
            )

        new_prompt = list(prompt[:start]) + [summary] + list(prompt[end:])
        return CompactionDecision(
            fired=True, candidate=candidate, new_prompt=new_prompt,
            reason=(
                f"fired: saved ${saved_dollars:.6f} > destroyed ${destroyed_dollars:.6f}"
            ),
            metrics=base_metrics,
        )
