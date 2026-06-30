"""Lane 1 contracts.

These are the only objects shared with Lane 2 (profiler/proxy). The proxy will
consume `RenderedPrompt` and emit `Trace` objects that reference `Block.stable_hash`
to attribute cache divergence — keep those fields stable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

Zone = Literal["stable", "volatile", "tail"]

TokenEstimator = Callable[[Any], int]


def _serialize(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, separators=(",", ":"))


def default_token_estimator(content: Any) -> int:
    """Deterministic byte-based estimator. Real backends should plug in a tokenizer."""
    s = _serialize(content)
    return max(1, (len(s) + 3) // 4)


def stable_content_hash(content: Any) -> str:
    return hashlib.sha256(_serialize(content).encode("utf-8")).hexdigest()


@dataclass
class Block:
    zone: Zone
    role: str
    content: Any
    token_count: int
    breakpoint: bool = False
    stable_hash: str = ""

    def __post_init__(self) -> None:
        if not self.stable_hash:
            self.stable_hash = stable_content_hash(self.content)


RenderedPrompt = list[Block]


@runtime_checkable
class CostModel(Protocol):
    """Contract 1: maps tokens to dollars, with cached/uncached split."""

    def price(
        self, *, cached_tokens: int, uncached_tokens: int, output_tokens: int = 0
    ) -> float: ...
    def cached_discount(self) -> float: ...
    def cache_write_premium(self) -> float: ...
    def block_size(self) -> int: ...
    def max_breakpoints(self) -> int: ...
    def breakpoint_lookback_blocks(self) -> int: ...


@dataclass
class FlatCostModel:
    """Simple per-token cost model used for tests and the Gate 0 spike.

    Parameters are loosely modeled on Anthropic-style economics (cached ~10% of input,
    cache writes ~1.25x). Override in tests to push the compactor decision boundary.
    """

    input_price_per_token: float = 1.0
    output_price_per_token: float = 5.0
    cached_discount_factor: float = 0.1
    cache_write_premium_factor: float = 1.25
    block_size_value: int = 128
    max_breakpoints_value: int = 4
    breakpoint_lookback_blocks_value: int = 20

    def price(
        self, *, cached_tokens: int, uncached_tokens: int, output_tokens: int = 0
    ) -> float:
        return (
            cached_tokens * self.input_price_per_token * self.cached_discount_factor
            + uncached_tokens * self.input_price_per_token
            + output_tokens * self.output_price_per_token
        )

    def cached_discount(self) -> float:
        return self.cached_discount_factor

    def cache_write_premium(self) -> float:
        return self.cache_write_premium_factor

    def block_size(self) -> int:
        return self.block_size_value

    def max_breakpoints(self) -> int:
        return self.max_breakpoints_value

    def breakpoint_lookback_blocks(self) -> int:
        return self.breakpoint_lookback_blocks_value
