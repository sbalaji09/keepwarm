"""Provider-shaped cost models.

These are shaped after public pricing/cache behavior, not pinned to exact dollar
amounts. The point is the *ratios* — cached vs uncached, cache-write premium,
block size, breakpoint count — because those are what drive the compactor's
trigger decision.

Defaults are roughly Anthropic / OpenAI 2025-era; override per backend.
All four implement the `CostModel` protocol from `keepwarm.contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnthropicLikeCostModel:
    """Anthropic-shaped: explicit cache_control, ~4 breakpoints, ~20-block lookback,
    cached reads ~10% of input, cache writes ~25% premium over uncached input.

    Default `input_price_per_mtok` is illustrative — pass real numbers per model.
    """

    input_price_per_mtok: float = 3.0       # USD per million input tokens (uncached)
    output_price_per_mtok: float = 15.0
    cached_read_discount: float = 0.1       # cached read = 10% of input price
    cache_write_premium_factor: float = 1.25
    block_size_value: int = 128
    max_breakpoints_value: int = 4
    breakpoint_lookback_blocks_value: int = 20

    def price(
        self, *, cached_tokens: int, uncached_tokens: int, output_tokens: int = 0
    ) -> float:
        in_rate = self.input_price_per_mtok / 1_000_000
        out_rate = self.output_price_per_mtok / 1_000_000
        return (
            cached_tokens * in_rate * self.cached_read_discount
            + uncached_tokens * in_rate
            + output_tokens * out_rate
        )

    def cached_discount(self) -> float:
        return self.cached_read_discount

    def cache_write_premium(self) -> float:
        return self.cache_write_premium_factor

    def block_size(self) -> int:
        return self.block_size_value

    def max_breakpoints(self) -> int:
        return self.max_breakpoints_value

    def breakpoint_lookback_blocks(self) -> int:
        return self.breakpoint_lookback_blocks_value


@dataclass
class OpenAILikeCostModel:
    """OpenAI-shaped: automatic prefix caching, cached reads ~50% of input,
    no explicit cache-write premium, larger effective block size.
    """

    input_price_per_mtok: float = 2.5
    output_price_per_mtok: float = 10.0
    cached_read_discount: float = 0.5
    cache_write_premium_factor: float = 1.0
    block_size_value: int = 1024
    # OpenAI's API doesn't expose breakpoints; modelled as effectively unbounded.
    max_breakpoints_value: int = 1
    breakpoint_lookback_blocks_value: int = 1

    def price(
        self, *, cached_tokens: int, uncached_tokens: int, output_tokens: int = 0
    ) -> float:
        in_rate = self.input_price_per_mtok / 1_000_000
        out_rate = self.output_price_per_mtok / 1_000_000
        return (
            cached_tokens * in_rate * self.cached_read_discount
            + uncached_tokens * in_rate
            + output_tokens * out_rate
        )

    def cached_discount(self) -> float:
        return self.cached_read_discount

    def cache_write_premium(self) -> float:
        return self.cache_write_premium_factor

    def block_size(self) -> int:
        return self.block_size_value

    def max_breakpoints(self) -> int:
        return self.max_breakpoints_value

    def breakpoint_lookback_blocks(self) -> int:
        return self.breakpoint_lookback_blocks_value
