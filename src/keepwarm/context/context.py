"""Context layer MVP — zones, canonicalization, render(), breakpoint hook.

The whole point: make it impossible to express a cache-breaking layout.
- stable zone is frozen after the first render
- tools are canonicalized (sorted keys, sorted by name)
- memory cannot land in stable
- set_active_tools() never mutates the stable tools block; it writes a tail constraint
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from keepwarm.contracts import (
    Block,
    RenderedPrompt,
    TokenEstimator,
    Zone,
    default_token_estimator,
)

# ---------- helpers for canonicalization ----------


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(v) for v in obj]
    return obj


def _canonicalize_tools(tools: list[dict]) -> list[dict]:
    canon = [_canonicalize(t) for t in tools]
    canon.sort(
        key=lambda t: t.get("name") if isinstance(t.get("name"), str)
        else json.dumps(t, sort_keys=True)
    )
    return canon


# ---------- block-type predicates (used by compactor + tests) ----------


def is_tools_block(b: Block) -> bool:
    return isinstance(b.content, dict) and "tools" in b.content


def is_memory_block(b: Block) -> bool:
    return isinstance(b.content, dict) and "memory" in b.content


def is_active_tools_block(b: Block) -> bool:
    return isinstance(b.content, dict) and "active_tools" in b.content


def is_summary_block(b: Block) -> bool:
    return isinstance(b.content, dict) and "summary" in b.content


# ---------- breakpoint strategy hook ----------


class BreakpointStrategy(Protocol):
    def place(self, blocks: list[Block]) -> list[Block]: ...


class DefaultBreakpointStrategy:
    """Mark the end of stable, end of volatile, and the final block.

    Per-provider strategies (Anthropic 4-breakpoint / ~20-block lookback,
    OpenAI auto, vLLM prefix-cache) plug in here later.
    """

    def place(self, blocks: list[Block]) -> list[Block]:
        if not blocks:
            return blocks
        last_stable: int | None = None
        last_volatile: int | None = None
        for i, b in enumerate(blocks):
            b.breakpoint = False
            if b.zone == "stable":
                last_stable = i
            elif b.zone == "volatile":
                last_volatile = i
        if last_stable is not None:
            blocks[last_stable].breakpoint = True
        if last_volatile is not None:
            blocks[last_volatile].breakpoint = True
        blocks[-1].breakpoint = True
        return blocks


# ---------- internal raw-block bookkeeping ----------


@dataclass
class _RawBlock:
    zone: Zone
    role: str
    content: Any
    kind: str


# ---------- zone accessors ----------


class _StableZone:
    def __init__(self, ctx: "Context") -> None:
        self._ctx = ctx

    def instructions(self, text: str) -> None:
        self._ctx._add(zone="stable", role="system", content=text, kind="instructions")

    def tools(self, tools: list[dict]) -> None:
        canon = _canonicalize_tools(list(tools))
        self._ctx._add(
            zone="stable", role="system", content={"tools": canon}, kind="tools"
        )


class _VolatileZone:
    def __init__(self, ctx: "Context") -> None:
        self._ctx = ctx

    def memory(self, content: Any) -> None:
        self._ctx._add(
            zone="volatile", role="system", content={"memory": content}, kind="memory"
        )


class _TailZone:
    def __init__(self, ctx: "Context") -> None:
        self._ctx = ctx

    def user(self, text: str) -> None:
        self._ctx._add(zone="tail", role="user", content=text, kind="user")

    def assistant(self, text: str) -> None:
        self._ctx._add(zone="tail", role="assistant", content=text, kind="assistant")

    def tool_result(self, content: Any) -> None:
        self._ctx._add(
            zone="tail", role="tool", content=content, kind="tool_result"
        )


# ---------- the Context object ----------


class Context:
    """Cache-stable prompt assembler.

    Usage:
        ctx = Context()
        ctx.stable.instructions("system prompt")
        ctx.stable.tools([...])
        ctx.volatile.memory("user prefers metric")
        ctx.tail.user("latest message")
        rendered = ctx.render()
    """

    def __init__(
        self,
        token_estimator: TokenEstimator | None = None,
        breakpoint_strategy: BreakpointStrategy | None = None,
    ) -> None:
        self._estimator: TokenEstimator = token_estimator or default_token_estimator
        self._breakpoints: BreakpointStrategy = (
            breakpoint_strategy or DefaultBreakpointStrategy()
        )
        self._stable: list[_RawBlock] = []
        self._volatile: list[_RawBlock] = []
        self._tail: list[_RawBlock] = []
        self._active_tools: _RawBlock | None = None
        self._stable_frozen: bool = False
        self.stable = _StableZone(self)
        self.volatile = _VolatileZone(self)
        self.tail = _TailZone(self)

    # -- internal: append a raw block to its zone, with validation --
    def _add(self, *, zone: Zone, role: str, content: Any, kind: str) -> None:
        if zone == "stable" and self._stable_frozen:
            raise RuntimeError(
                "stable zone is frozen; cannot add more stable content after render()"
            )
        if zone == "stable" and kind == "memory":
            raise ValueError("memory must live in the volatile zone, not stable")
        rb = _RawBlock(zone=zone, role=role, content=content, kind=kind)
        if zone == "stable":
            self._stable.append(rb)
        elif zone == "volatile":
            self._volatile.append(rb)
        else:
            self._tail.append(rb)

    def freeze_stable(self) -> None:
        self._stable_frozen = True

    def set_active_tools(self, names: list[str]) -> None:
        """Declare which tools are *live* this turn.

        Critically, this never mutates the stable tools block. It writes a
        constraint into the tail so the cached prefix stays byte-identical
        across phase changes (e.g. plan-mode toggles).
        """
        canon = sorted(set(names))
        content = {"active_tools": canon}
        if self._active_tools is None:
            self._active_tools = _RawBlock(
                zone="tail", role="system", content=content, kind="active_tools"
            )
        else:
            self._active_tools.content = content

    def add_compacted_summary(self, summary_text: str, replaces_count: int) -> None:
        """Append a summary block at the tail. The compactor will not rewrite it."""
        self._tail.append(
            _RawBlock(
                zone="tail",
                role="system",
                content={"summary": summary_text, "replaces": replaces_count},
                kind="summary",
            )
        )

    # -- render --
    def render(self) -> RenderedPrompt:
        # freeze stable on first render — adding more stable content after this raises
        self._stable_frozen = True
        blocks: list[Block] = []
        for rb in self._stable:
            blocks.append(self._materialize(rb))
        for rb in self._volatile:
            blocks.append(self._materialize(rb))
        # active-tools constraint sits at the front of the tail
        if self._active_tools is not None:
            blocks.append(self._materialize(self._active_tools))
        for rb in self._tail:
            blocks.append(self._materialize(rb))
        return self._breakpoints.place(blocks)

    def _materialize(self, rb: _RawBlock) -> Block:
        return Block(
            zone=rb.zone,
            role=rb.role,
            content=rb.content,
            token_count=self._estimator(rb.content),
        )
