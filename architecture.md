# Architecture

## Overview

Three components and three contracts. The contracts are the coordination backbone — define them first and the components can be built in parallel by different people (see `TEAM.md`).

```
                 your orchestrator (LangGraph / CrewAI / custom)
                                  │
                                  ▼
        ┌─────────────────────────────────────────────┐
        │              CONTEXT LAYER (B)               │
        │  zones · breakpoints · set_active_tools ·    │
        │  memory zone · render()                      │
        └───────────────┬─────────────────────────────┘
                        │ rendered prompt (Contract 2)
                        ▼
        ┌─────────────────────────────────────────────┐
        │               COMPACTOR (C)                  │
        │  cost model · newest-first · trigger gate    │
        └───────────────┬─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────────────────┐
        │          PROFILER / PROXY (A)                │
        │  capture · first-divergence · cause · report │
        └───────────────┬─────────────────────────────┘
                        │ trace + cache fields (Contract 3)
                        ▼
                  model provider / inference server
```

The Profiler (A) sits on the wire as an OpenAI-compatible proxy, so it also works **standalone** against agents that don't use the Context Layer at all. That's deliberate: A is the v0.1 product and the shared test harness for B and C.

---

## The three contracts (define these in week 0, together)

These are the only interfaces the three workstreams share. Lock them first; everything else is private to a component.

### Contract 1 — Cost Model

Maps tokens to dollars and latency per backend, with the cached/uncached split.

```python
class CostModel(Protocol):
    def price(self, *, cached_tokens: int, uncached_tokens: int,
              output_tokens: int) -> float: ...
    def cached_discount(self) -> float:        # e.g. 0.1 (cached = 10% of input price)
    def cache_write_premium(self) -> float:    # e.g. 1.25 (writes cost more than reads)
    def block_size(self) -> int:               # min cacheable unit, e.g. 128 tokens
    def max_breakpoints(self) -> int           # e.g. Anthropic: 4
    def breakpoint_lookback_blocks(self) -> int # e.g. Anthropic: ~20
```

Owned by C, consumed by A and B.

### Contract 2 — Rendered Prompt

What `Context.render()` emits: ordered blocks tagged by zone, with breakpoint markers. This is what the compactor operates on and what gets serialized to the provider.

```python
@dataclass
class Block:
    zone: Literal["stable", "volatile", "tail"]
    role: str                  # system / user / assistant / tool
    content: str | list        # provider-native content
    token_count: int
    breakpoint: bool           # place a cache breakpoint after this block?
    stable_hash: str           # content hash, for divergence detection

RenderedPrompt = list[Block]
```

Owned by B, consumed by C and A.

### Contract 3 — Trace

What the proxy captures per call: the request, the response, and the provider's reported cache fields (normalized across providers).

```python
@dataclass
class Trace:
    call_index: int
    prefix_family: str         # (model, system+tools hash, cache_key)
    rendered: RenderedPrompt | None   # present if Context Layer used
    reported_cache_read: int   # Anthropic cache_read_input_tokens / OpenAI cached_tokens
    reported_cache_write: int
    reported_uncached: int
    output_tokens: int
    step_label: str | None     # framework node, if adapter present
```

Owned by A, consumed by B and C for verification.

---

## Component A — Profiler / Proxy

- **Capture.** Local OpenAI-compatible proxy; passthrough; logs request/response; normalizes provider cache fields into `Trace`.
- **First-divergence engine.** Bucket calls by `prefix_family`; compute the longest common prefix vs the previous call in the bucket (byte pre-pass → tokenizer-accurate confirm). The divergence index is where caching stops paying.
- **Avoidable vs unavoidable classifier.** New tokens appended at the tail are *unavoidable*. Tokens that were present before and unchanged but fell out of the cached prefix because something earlier diverged are *avoidable* — that's the waste number.
- **Cause attribution.** Diff the two prompts at the divergence point and name the culprit (`tool array reordered`, `timestamp at offset N`, `memory injected at message[k]`, `compaction rewrote messages[a..b]`).
- **Reconciliation.** Compare predicted hit rate against `reported_cache_*`. Divergence between them is itself a finding (provider-side routing scatter, sub-block prompts, TTL eviction).
- **Report.** CLI + static HTML: overall hit rate, achievable hit rate, wasted re-prefill in tokens and dollars, breakdown by step type, ranked causes.

## Component B — Context Layer

- **Zones.** `stable` (front, frozen, cached), `volatile` (own region + breakpoint), `tail` (the only growing part). Validation refuses volatile content in `stable`.
- **Renderer.** Produces `RenderedPrompt`; places breakpoints per backend (e.g. intermediate breakpoints every ~18 blocks to respect Anthropic's ~20-block lookback).
- **Cache-safe primitives.** `set_active_tools()` writes a tail constraint, never mutates the tools array. Memory store scoped to the volatile zone.
- **Adapters.** Thin shims for LangGraph and CrewAI that route their prompt assembly through the Context object and attach `step_label`.

## Component C — Compactor

- **Cost model.** Implements Contract 1 per backend.
- **Trigger.** Fire only when `prefill_saved > prefill_destroyed` under the cost model (not at a fixed token threshold).
- **Strategy.** Compact **newest-first** so low message indices stay byte-identical; summarize into a stable trailing block; never rewrite an existing summary.
- **Benchmark harness.** The go/no-go: run a real long agent with naive vs cost-aware compaction; report prefill-dollars saved. This is `ROADMAP.md` → *Gate 0*.

---

## Data flow, one step

1. Orchestrator asks for the next model call.
2. Context Layer (B) renders the prompt (Contract 2), placing breakpoints.
3. Compactor (C) checks its trigger; compacts newest-first only if it pays.
4. Proxy (A) sends to the provider, captures the response + cache fields (Contract 3).
5. Profiler updates the running report; in CI, asserts no regression.

## Design rules

- **Client-side only.** Everything happens before the request leaves and after the response returns. No backend internals.
- **Correct by construction.** The API should make a cache-breaking layout impossible to express, not merely warn about it.
- **Provider-neutral core, provider-specific edges.** One zone model; per-backend breakpoint/cost strategies behind the cost model interface.
