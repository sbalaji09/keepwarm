from keepwarm.compaction import (
    CostAwareNewestFirstCompactor,
    NaiveOldestFirstCompactor,
)
from keepwarm.context import is_summary_block
from keepwarm.contracts import Block, FlatCostModel


def _stable(text="SYS", tokens=200):
    return Block(zone="stable", role="system", content=text, token_count=tokens)


def _tail(text, tokens=100):
    return Block(zone="tail", role="user", content=text, token_count=tokens)


def test_naive_rewrites_earlier_blocks():
    prompt = [_stable()] + [_tail(f"turn-{i}") for i in range(4)]
    naive = NaiveOldestFirstCompactor(FlatCostModel(), batch_size=2)
    dec = naive.compact(prompt)
    assert dec.fired
    # the earliest tail block in the new prompt is now the summary
    assert is_summary_block(dec.new_prompt[1])
    assert dec.new_prompt[1].stable_hash != prompt[1].stable_hash


def test_cost_aware_preserves_prefix_byte_identically():
    prompt = [_stable()] + [_tail(f"turn-{i}", tokens=500) for i in range(4)]
    ca = CostAwareNewestFirstCompactor(FlatCostModel(), batch_size=2)
    dec = ca.compact(prompt)
    assert dec.fired
    # everything before the modification point is byte-identical
    for i in range(dec.candidate.start_index):
        assert dec.new_prompt[i].stable_hash == prompt[i].stable_hash


def test_cost_aware_fires_when_saved_exceeds_destroyed():
    # batch lives at the very end → destroyed_tokens == 0 → fires
    prompt = [_stable()] + [_tail(f"turn-{i}", tokens=1000) for i in range(3)]
    ca = CostAwareNewestFirstCompactor(FlatCostModel(), batch_size=2)
    dec = ca.compact(prompt)
    assert dec.fired
    assert dec.candidate.destroyed_tokens == 0


def test_cost_aware_refuses_when_destroyed_exceeds_saved():
    # An existing large summary block sits after the eligible batch. Compacting
    # the batch would invalidate the cache on that big trailing block, and the
    # math should refuse.
    existing_summary = Block(
        zone="tail", role="system",
        content={"summary": "old", "replaces": 3},
        token_count=50_000,
    )
    prompt = [
        _stable(),
        _tail("turn-0", tokens=10),
        _tail("turn-1", tokens=10),
        existing_summary,
    ]
    ca = CostAwareNewestFirstCompactor(FlatCostModel(), batch_size=2)
    dec = ca.compact(prompt)
    assert not dec.fired
    assert "refused" in dec.reason
    # original prompt is returned unchanged
    assert [b.stable_hash for b in dec.new_prompt] == [b.stable_hash for b in prompt]


def test_existing_summary_block_is_not_rewritten():
    summary = Block(
        zone="tail", role="system",
        content={"summary": "old", "replaces": 2},
        token_count=20,
    )
    prompt = [
        _stable(),
        summary,
        _tail("turn-0", tokens=1000),
        _tail("turn-1", tokens=1000),
    ]
    ca = CostAwareNewestFirstCompactor(FlatCostModel(), batch_size=2)
    dec = ca.compact(prompt)
    assert dec.fired
    surviving = [
        b for b in dec.new_prompt
        if is_summary_block(b) and b.content.get("summary") == "old"
    ]
    assert len(surviving) == 1


def test_not_enough_eligible_blocks_returns_unfired():
    prompt = [_stable()]
    ca = CostAwareNewestFirstCompactor(FlatCostModel(), batch_size=2)
    dec = ca.compact(prompt)
    assert not dec.fired
    assert dec.candidate is None
