"""Realistic Gate 0 — baseline vs naive vs cost-aware on long-agent prompts.

This is the experiment that decides whether cost-aware compaction is worth
building into the full Lane 1 context layer. We assert the mechanical
properties (cost-aware preserves a longer prefix) and the economic property
(cost-aware has lower estimated next-call prefill cost) under a realistic
provider-shaped cost model.
"""

from __future__ import annotations

from dataclasses import dataclass

from keepwarm.compaction import (
    CompactionDecision,
    CostAwareNewestFirstCompactor,
    NaiveOldestFirstCompactor,
)
from keepwarm.contracts import CostModel, RenderedPrompt
from keepwarm.cost_models import AnthropicLikeCostModel, OpenAILikeCostModel

from tests.fixtures.realistic_prompts import (
    build_long_agent_prompt,
    large_fixture,
    small_fixture,
)


# ---------- metrics ----------


@dataclass
class StrategyReport:
    name: str
    fired: bool
    prompt_tokens: int
    first_changed_index: int
    prefix_preserved_tokens: int
    destroyed_cache_tokens: int
    next_call_prefill_cost: float
    recurring_saved_tokens: int
    recurring_saved_cost: float


def _first_divergence(original: RenderedPrompt, new: RenderedPrompt) -> int:
    for i in range(min(len(original), len(new))):
        if original[i].stable_hash != new[i].stable_hash:
            return i
    if len(original) == len(new):
        return len(original)
    return min(len(original), len(new))


def _next_call_prefill_cost(
    new_prompt: RenderedPrompt, original: RenderedPrompt, cm: CostModel
) -> float:
    """On the next call, blocks up to the first divergence stay cached.
    Everything from the divergence point onward must be re-prefilled.
    """
    div = _first_divergence(original, new_prompt)
    cached = sum(b.token_count for b in new_prompt[:div])
    uncached = sum(b.token_count for b in new_prompt[div:])
    return cm.price(cached_tokens=cached, uncached_tokens=uncached)


def _baseline_report(prompt: RenderedPrompt, cm: CostModel) -> StrategyReport:
    total = sum(b.token_count for b in prompt)
    return StrategyReport(
        name="no_compaction",
        fired=False,
        prompt_tokens=total,
        first_changed_index=len(prompt),
        prefix_preserved_tokens=total,
        destroyed_cache_tokens=0,
        next_call_prefill_cost=cm.price(cached_tokens=total, uncached_tokens=0),
        recurring_saved_tokens=0,
        recurring_saved_cost=0.0,
    )


def _strategy_report(
    name: str,
    original: RenderedPrompt,
    decision: CompactionDecision,
    cm: CostModel,
) -> StrategyReport:
    new_prompt = decision.new_prompt
    div = _first_divergence(original, new_prompt)
    prefix_tokens = sum(b.token_count for b in new_prompt[:div])
    destroyed = decision.candidate.destroyed_tokens if decision.candidate else 0
    saved_tokens = decision.candidate.saved_tokens if decision.candidate else 0
    saved_cost = cm.price(cached_tokens=0, uncached_tokens=max(saved_tokens, 0))
    return StrategyReport(
        name=name,
        fired=decision.fired,
        prompt_tokens=sum(b.token_count for b in new_prompt),
        first_changed_index=div,
        prefix_preserved_tokens=prefix_tokens,
        destroyed_cache_tokens=destroyed,
        next_call_prefill_cost=_next_call_prefill_cost(new_prompt, original, cm),
        recurring_saved_tokens=saved_tokens if decision.fired else 0,
        recurring_saved_cost=saved_cost if decision.fired else 0.0,
    )


def _compare(prompt: RenderedPrompt, cm: CostModel, batch_size: int = 6) -> dict[str, StrategyReport]:
    baseline = _baseline_report(prompt, cm)
    naive = NaiveOldestFirstCompactor(cm, batch_size=batch_size).compact(prompt)
    cost_aware = CostAwareNewestFirstCompactor(cm, batch_size=batch_size).compact(prompt)
    return {
        "baseline": baseline,
        "naive": _strategy_report("naive_oldest_first", prompt, naive, cm),
        "cost_aware": _strategy_report("cost_aware_newest_first", prompt, cost_aware, cm),
    }


# ---------- tests ----------


def test_realistic_anthropic_cost_aware_beats_naive():
    cm = AnthropicLikeCostModel()
    prompt = large_fixture()
    reports = _compare(prompt, cm, batch_size=8)

    naive = reports["naive"]
    ca = reports["cost_aware"]

    assert naive.fired
    assert ca.fired

    # Mechanical: cost-aware diverges later than naive.
    assert ca.first_changed_index > naive.first_changed_index, (
        f"naive_div={naive.first_changed_index} ca_div={ca.first_changed_index}"
    )

    # Mechanical: cost-aware preserves materially more prefix.
    assert ca.prefix_preserved_tokens > naive.prefix_preserved_tokens

    # Economic: cost-aware next-call prefill cost is lower.
    assert ca.next_call_prefill_cost < naive.next_call_prefill_cost, (
        f"naive=${naive.next_call_prefill_cost:.6f} "
        f"cost_aware=${ca.next_call_prefill_cost:.6f}"
    )


def test_realistic_openai_cost_aware_beats_naive():
    cm = OpenAILikeCostModel()
    prompt = small_fixture()
    reports = _compare(prompt, cm, batch_size=6)

    naive = reports["naive"]
    ca = reports["cost_aware"]

    assert naive.fired and ca.fired
    assert ca.first_changed_index > naive.first_changed_index
    assert ca.prefix_preserved_tokens > naive.prefix_preserved_tokens
    assert ca.next_call_prefill_cost < naive.next_call_prefill_cost


def test_cost_aware_skips_volatile_memory_block():
    """Memory must never be compacted; it lives in volatile and is off-limits."""
    cm = AnthropicLikeCostModel()
    prompt = build_long_agent_prompt(
        seed=3, num_tools=15, num_tail_steps=40, with_memory=True, with_active_tools=False
    )
    original_memory = [b.stable_hash for b in prompt if b.zone == "volatile"]
    decision = CostAwareNewestFirstCompactor(cm, batch_size=6).compact(prompt)
    assert decision.fired
    new_memory = [b.stable_hash for b in decision.new_prompt if b.zone == "volatile"]
    assert new_memory == original_memory


def test_cost_aware_skips_active_tools_constraint():
    """The tail-side active_tools constraint must survive compaction unchanged."""
    cm = AnthropicLikeCostModel()
    prompt = build_long_agent_prompt(
        seed=4, num_tools=12, num_tail_steps=40, with_memory=False, with_active_tools=True
    )
    from keepwarm.context import is_active_tools_block
    before = [b.stable_hash for b in prompt if is_active_tools_block(b)]
    assert len(before) == 1
    decision = CostAwareNewestFirstCompactor(cm, batch_size=6).compact(prompt)
    assert decision.fired
    after = [b.stable_hash for b in decision.new_prompt if is_active_tools_block(b)]
    assert after == before


def test_decision_metrics_are_populated():
    cm = AnthropicLikeCostModel()
    prompt = small_fixture()
    decision = CostAwareNewestFirstCompactor(cm, batch_size=6).compact(prompt)
    assert decision.fired
    m = decision.metrics
    assert m["strategy"] == "cost_aware_newest_first"
    assert m["prefix_preserved_blocks"] >= 1
    assert m["prefix_preserved_tokens"] > 0
    assert 0.0 < m["saving_ratio"] <= 1.0
    c = decision.candidate
    assert c.batch_tokens > 0
    assert c.summary_tokens > 0
    assert c.saved_dollars > 0
    assert c.destroyed_dollars >= 0
