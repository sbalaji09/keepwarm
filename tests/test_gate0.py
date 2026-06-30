"""Gate 0 fixture — the go/no-go demo for cost-aware compaction.

Shows on a long rendered prompt:
1. Naive oldest-first changes an early block (busts the cached prefix).
2. Cost-aware newest-first keeps the entire prefix byte-identical.
3. Under a simple cost model, the next-call prefill cost for cost-aware is
   strictly lower than for naive.
"""

from keepwarm.compaction import (
    CostAwareNewestFirstCompactor,
    NaiveOldestFirstCompactor,
)
from keepwarm.contracts import Block, FlatCostModel, RenderedPrompt


def _build_long_prompt() -> RenderedPrompt:
    blocks: RenderedPrompt = [
        Block(zone="stable", role="system", content="SYS+TOOLS", token_count=2_000)
    ]
    for i in range(8):
        blocks.append(
            Block(zone="tail", role="user", content=f"turn-{i}", token_count=500)
        )
    return blocks


def _first_divergence(original: RenderedPrompt, new: RenderedPrompt) -> int:
    for i in range(min(len(original), len(new))):
        if original[i].stable_hash != new[i].stable_hash:
            return i
    return min(len(original), len(new))


def _next_call_prefill_cost(
    new_prompt: RenderedPrompt, original: RenderedPrompt, cm: FlatCostModel
) -> float:
    """On the next call, blocks up to first-divergence are cache hits; the rest is uncached prefill."""
    div = _first_divergence(original, new_prompt)
    cached = sum(b.token_count for b in new_prompt[:div])
    uncached = sum(b.token_count for b in new_prompt[div:])
    return cm.price(cached_tokens=cached, uncached_tokens=uncached)


def test_gate0_cost_aware_beats_naive():
    cm = FlatCostModel()
    prompt = _build_long_prompt()
    naive = NaiveOldestFirstCompactor(cm, batch_size=3).compact(prompt)
    ca = CostAwareNewestFirstCompactor(cm, batch_size=3).compact(prompt)

    assert naive.fired and ca.fired

    # Naive divergence sits near the front; cost-aware divergence sits near the back.
    naive_div = _first_divergence(prompt, naive.new_prompt)
    ca_div = _first_divergence(prompt, ca.new_prompt)
    assert naive_div < ca_div, (
        f"expected cost-aware to diverge later than naive, "
        f"got naive_div={naive_div}, ca_div={ca_div}"
    )

    # Cost-aware preserves the entire pre-batch prefix byte-identically.
    for i in range(ca.candidate.start_index):
        assert ca.new_prompt[i].stable_hash == prompt[i].stable_hash

    naive_cost = _next_call_prefill_cost(naive.new_prompt, prompt, cm)
    ca_cost = _next_call_prefill_cost(ca.new_prompt, prompt, cm)
    assert ca_cost < naive_cost, (
        f"cost-aware ({ca_cost:.2f}) should beat naive ({naive_cost:.2f})"
    )
