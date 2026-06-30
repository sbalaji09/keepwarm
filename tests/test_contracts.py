from keepwarm.contracts import (
    Block,
    CostModel,
    FlatCostModel,
    default_token_estimator,
    stable_content_hash,
)


def test_block_autocomputes_stable_hash():
    b = Block(zone="stable", role="system", content="hello", token_count=2)
    assert b.stable_hash == stable_content_hash("hello")


def test_block_preserves_explicit_hash():
    b = Block(
        zone="tail", role="user", content="hi", token_count=1, stable_hash="abc"
    )
    assert b.stable_hash == "abc"


def test_hash_is_canonical_for_dicts():
    h1 = stable_content_hash({"a": 1, "b": 2})
    h2 = stable_content_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_token_estimator_deterministic():
    assert default_token_estimator("a" * 8) == default_token_estimator("b" * 8)
    assert default_token_estimator("") == 1  # floored at 1


def test_flat_cost_model_satisfies_protocol():
    cm = FlatCostModel()
    assert isinstance(cm, CostModel)


def test_flat_cost_model_pricing():
    cm = FlatCostModel(input_price_per_token=1.0, cached_discount_factor=0.1)
    # 100 cached @ 0.1 + 10 uncached @ 1.0 = 10 + 10 = 20
    assert cm.price(cached_tokens=100, uncached_tokens=10) == 20.0
