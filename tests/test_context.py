import pytest

from keepwarm.context import (
    Context,
    is_active_tools_block,
    is_memory_block,
    is_tools_block,
)


def test_zone_ordering_stable_volatile_tail():
    ctx = Context()
    ctx.tail.user("hi")          # added first, must still render last
    ctx.volatile.memory("mem")
    ctx.stable.instructions("sys")
    rendered = ctx.render()
    zones = [b.zone for b in rendered]
    assert zones == ["stable", "volatile", "tail"]


def test_stable_frozen_after_render():
    ctx = Context()
    ctx.stable.instructions("sys")
    ctx.render()
    with pytest.raises(RuntimeError):
        ctx.stable.instructions("more")


def test_explicit_freeze_stable():
    ctx = Context()
    ctx.stable.instructions("sys")
    ctx.freeze_stable()
    with pytest.raises(RuntimeError):
        ctx.stable.tools([{"name": "x"}])


def test_tools_canonicalization_independent_of_input_order():
    a = Context()
    a.stable.tools([{"name": "b", "x": 1, "y": 2}, {"name": "a"}])
    b = Context()
    b.stable.tools([{"name": "a"}, {"y": 2, "x": 1, "name": "b"}])
    ta = next(x for x in a.render() if is_tools_block(x))
    tb = next(x for x in b.render() if is_tools_block(x))
    assert ta.stable_hash == tb.stable_hash


def test_set_active_tools_does_not_mutate_tools_block():
    ctx = Context()
    ctx.stable.tools([{"name": "edit"}, {"name": "read"}, {"name": "bash"}])
    before_hash = next(b for b in ctx.render() if is_tools_block(b)).stable_hash
    ctx.set_active_tools(["read"])
    after = ctx.render()
    after_tools = next(b for b in after if is_tools_block(b))
    assert after_tools.stable_hash == before_hash
    actives = [b for b in after if is_active_tools_block(b)]
    assert len(actives) == 1
    assert actives[0].zone == "tail"
    assert actives[0].content["active_tools"] == ["read"]


def test_set_active_tools_updates_in_place_not_appends():
    ctx = Context()
    ctx.stable.tools([{"name": "edit"}, {"name": "read"}])
    ctx.set_active_tools(["edit"])
    ctx.set_active_tools(["read"])
    rendered = ctx.render()
    actives = [b for b in rendered if is_active_tools_block(b)]
    assert len(actives) == 1
    assert actives[0].content["active_tools"] == ["read"]


def test_memory_renders_in_volatile_zone():
    ctx = Context()
    ctx.volatile.memory("user prefers metric")
    rendered = ctx.render()
    mems = [b for b in rendered if is_memory_block(b)]
    assert len(mems) == 1
    assert mems[0].zone == "volatile"


def test_breakpoint_is_present_and_controllable():
    ctx = Context()
    ctx.stable.instructions("sys")
    ctx.volatile.memory("m")
    ctx.tail.user("hi")
    rendered = ctx.render()
    assert all(hasattr(b, "breakpoint") for b in rendered)
    # default strategy: end-of-stable, end-of-volatile, and final block all marked
    assert rendered[0].breakpoint   # only stable block -> end of stable
    assert rendered[1].breakpoint   # only volatile block -> end of volatile
    assert rendered[-1].breakpoint  # last block always


def test_every_block_has_stable_hash():
    ctx = Context()
    ctx.stable.instructions("sys")
    ctx.tail.user("hi")
    for b in ctx.render():
        assert b.stable_hash and len(b.stable_hash) == 64
