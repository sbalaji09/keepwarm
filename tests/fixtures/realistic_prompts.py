"""Synthetic but realistically-shaped long-agent prompts.

Shape is modeled on a research/coding agent: a large system prompt, ~20 tool
definitions, an optional volatile memory block, then a long tail of mixed
user turns, assistant "thinking" summaries, and tool results of varied sizes.
Deterministic so Gate 0 comparisons are reproducible.
"""

from __future__ import annotations

import random

from keepwarm.context import Context
from keepwarm.contracts import RenderedPrompt


SYSTEM_PROMPT = (
    "You are an autonomous research/coding agent. Use the provided tools to "
    "explore the repository, read files, run commands, and produce a final "
    "answer. Be careful about destructive actions. Always cite sources by "
    "file path and line number. Prefer dedicated tools over Bash when one fits."
) * 8  # large-ish (~1.2KB) — typical long system prompt scale


def _make_tools(n: int) -> list[dict]:
    """Synthesize n tool definitions in arbitrary key order (canonicalized on add)."""
    out = []
    for i in range(n):
        out.append({
            "name": f"tool_{i:02d}",
            "description": f"Synthetic tool number {i} for fixture coverage. " * 4,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search string"},
                    "limit": {"type": "integer", "default": 10},
                    "options": {
                        "type": "object",
                        "properties": {
                            "deep": {"type": "boolean"},
                            "context": {"type": "integer"},
                        },
                    },
                },
                "required": ["query"],
            },
        })
    return out


def _user_turn(rng: random.Random, i: int) -> str:
    return f"[user turn {i}] " + ("question about repository internals " * rng.randint(2, 12))


def _assistant_reasoning(rng: random.Random, i: int) -> str:
    return f"[assistant reasoning {i}] " + ("considering tool outputs and citing files " * rng.randint(4, 18))


def _tool_result(rng: random.Random, i: int) -> dict:
    return {
        "tool_result": {
            "tool": f"tool_{rng.randint(0, 19):02d}",
            "call_index": i,
            "stdout": ("line of synthetic tool output\n" * rng.randint(5, 40)).strip(),
            "stderr": "",
        }
    }


def build_long_agent_prompt(
    *,
    seed: int = 0,
    num_tools: int = 20,
    num_tail_steps: int = 60,
    with_memory: bool = True,
    with_active_tools: bool = True,
) -> RenderedPrompt:
    """Return a deterministic rendered prompt shaped like a long agent run."""
    rng = random.Random(seed)
    ctx = Context()
    ctx.stable.instructions(SYSTEM_PROMPT)
    ctx.stable.tools(_make_tools(num_tools))
    if with_memory:
        ctx.volatile.memory(
            {"facts": [
                "user prefers metric units",
                "user repo uses pytest",
                "user previously asked about caching",
            ]}
        )
    if with_active_tools:
        ctx.set_active_tools([f"tool_{i:02d}" for i in range(0, num_tools, 2)])

    # interleave user / assistant-reasoning / tool-result so token sizes vary
    for i in range(num_tail_steps):
        kind = i % 3
        if kind == 0:
            ctx.tail.user(_user_turn(rng, i))
        elif kind == 1:
            ctx.tail.assistant(_assistant_reasoning(rng, i))
        else:
            ctx.tail.tool_result(_tool_result(rng, i))
    return ctx.render()


def small_fixture() -> RenderedPrompt:
    """Smaller variant useful for fast assertions."""
    return build_long_agent_prompt(
        seed=1, num_tools=10, num_tail_steps=30, with_memory=True, with_active_tools=True
    )


def large_fixture() -> RenderedPrompt:
    """Closer to a multi-hour run."""
    return build_long_agent_prompt(
        seed=2, num_tools=25, num_tail_steps=90, with_memory=True, with_active_tools=True
    )
