# Lane 2 — Profiler & Capture: Implementation Spec

**Owns:** `keepwarm/profiler/` + `benchmarks/`  
**Component:** A (Profiler / Proxy)  
**Contract owned:** Trace (Contract 3)  
**Depends on:** Cost Model (Contract 1, owned by Lane 1) — consumed, not authored  

---

## Goals

1. Ship the v0.1 product: a standalone profiler any agent can point at, no other keepwarm components required.
2. Provide the measurement infrastructure Gate 0 needs to validate Lane 1's compactor.
3. Become the built-in assertion that runs in CI to catch cache regressions before they ship.

---

## Directory layout

```
keepwarm/
  profiler/
    __init__.py
    proxy.py          # OpenAI-compatible ASGI proxy
    capture.py        # request/response normalization → Trace
    divergence.py     # first-divergence engine
    classifier.py     # avoidable vs unavoidable, cause attribution
    reconciler.py     # predicted vs reported cache fields
    report.py         # CLI output + static HTML
    contracts.py      # owns Trace (Contract 3); imports CostModel (Contract 1)

benchmarks/
  __init__.py
  record.py           # record a golden trace from a live agent run
  replay.py           # replay a golden trace against the profiler (no live calls)
  gate0.py            # Gate 0 harness: naive vs cost-aware compaction
  fixtures/           # checked-in golden traces for CI
```

---

## Week-by-week deliverables

### Week 0 (together with Lane 1, 1–2 days)
Lock `contracts.py`. Nothing else starts until these compile and both lanes sign off.

- `Trace` dataclass (Contract 3) — see schema below
- `RenderedPrompt` / `Block` import stubs (Contract 2, owned by Lane 1) — mock for now
- `CostModel` protocol import (Contract 1, owned by Lane 1) — mock for now
- Golden-trace format: JSON serialization of `list[Trace]`, schema version field

### Week 1 — proxy + capture + first-divergence engine
By end of week: can profile a real agent running against OpenAI or Anthropic.

**Deliverables:**
- `proxy.py` — runnable, passes all traffic through
- `capture.py` — normalizes provider cache fields into `Trace`
- `divergence.py` — first-divergence engine (byte pre-pass + tokenizer confirm)
- Minimal `report.py` — prints hit rate + first-divergence point to stdout

### Week 2 — report, reconciliation, ship v0.1
v0.1 profiler releases end of this week regardless of Gate 0 outcome.

**Deliverables:**
- `classifier.py` — avoidable vs unavoidable, cause attribution
- `reconciler.py` — compares predicted vs reported cache fields
- `report.py` — full CLI + static HTML report
- `benchmarks/record.py` + `benchmarks/replay.py`
- `benchmarks/fixtures/` — at least one golden trace checked in for CI

### Weeks 3–4 — integration + launch benchmark
- Gate 0: pair with Lane 1, run `benchmarks/gate0.py` against Lane 1's compactor
- Launch benchmark: profile top 5 frameworks (LangGraph, CrewAI, OpenAI Agents SDK, AutoGen, raw loop), report cache hit rates pre/post keepwarm
- CI regression gate: `profiler` runs against `benchmarks/fixtures/` on every PR; asserts hit rate within threshold
- Optional: pick up a framework adapter (LangGraph or CrewAI) from Lane 1 if Gate 0 is Go and Lane 1 needs breathing room

---

## Contract 3 — Trace (owned here)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Trace:
    call_index: int                        # 0-based within a prefix_family
    prefix_family: str                     # hash(model + system_prompt + tools)
    rendered: "RenderedPrompt | None"      # present only if Context Layer used
    reported_cache_read: int               # provider's cache hit tokens
    reported_cache_write: int              # provider's cache write tokens
    reported_uncached: int                 # tokens sent at full price
    output_tokens: int
    step_label: str | None                 # framework node name, if adapter present
    request_messages: list[dict[str, Any]] # raw messages sent (for divergence engine)
    timestamp_utc: float                   # unix timestamp, for TTL analysis
    provider: str                          # "anthropic" | "openai" | "vllm"
    model: str
```

Serialization: `dataclasses.asdict` → JSON, with a top-level `{"schema_version": 1, "traces": [...]}` envelope.

---

## Component A — detailed design

### `proxy.py` — OpenAI-compatible ASGI proxy

The proxy is the entry point. An agent points its `base_url` here; every call passes through, is captured, and is forwarded to the real provider.

```
agent → proxy (localhost:4000) → provider API
                ↓
           capture.py → Trace
```

**Implementation notes:**
- Use `starlette` (lightweight, no fastapi overhead)
- Passthrough: forward all headers except `Authorization` (re-inject from env `KEEPWARM_API_KEY`)
- Stream-safe: if the provider streams, buffer the full response before capturing, then re-stream to the caller — profiling adds one RTT of latency at most
- Provider detection: infer from `base_url` or a `X-Keepwarm-Provider` header
- Session ID: callers may set `X-Keepwarm-Session` to group calls into a prefix family; otherwise derive from the system prompt + tools hash

**Configuration (env vars):**
```
KEEPWARM_UPSTREAM=https://api.anthropic.com   # or openai, or vllm endpoint
KEEPWARM_API_KEY=sk-...                        # forwarded to upstream
KEEPWARM_SESSION_ID=my-agent                  # optional, sets prefix_family label
KEEPWARM_REPORT_DIR=./keepwarm-traces         # where to write Trace JSON
```

### `capture.py` — normalization

Normalizes the provider-specific cache fields into the `Trace` schema.

```python
def normalize_usage(provider: str, usage: dict) -> tuple[int, int, int]:
    """Returns (cache_read, cache_write, uncached)."""
    if provider == "anthropic":
        return (
            usage.get("cache_read_input_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
            usage.get("input_tokens", 0),
        )
    elif provider == "openai":
        details = usage.get("prompt_tokens_details", {})
        cached = details.get("cached_tokens", 0)
        total = usage.get("prompt_tokens", 0)
        return (cached, 0, total - cached)
    elif provider == "vllm":
        # vLLM doesn't expose cache fields; approximate from latency delta
        # log a warning and return zeros — reconciler will flag it
        return (0, 0, usage.get("prompt_tokens", 0))
```

### `divergence.py` — first-divergence engine

Groups traces by `prefix_family`. For each new trace in a family, finds where the prompt diverges from the previous call. This is the number the profiler reports as "where caching stopped paying."

**Algorithm:**
1. **Byte pre-pass.** Serialize both `request_messages` lists to canonical JSON (sorted keys, no whitespace). Find the first byte index that differs. O(n) and fast.
2. **Tokenizer-accurate confirm.** Map the byte index to a token boundary using `tiktoken` (or the Anthropic tokenizer for Anthropic models). The token boundary is what the model actually sees as the divergence point.
3. **Avoidable vs unavoidable.** Tokens appended at the tail (new user turn, new tool result) are *unavoidable*. Tokens that were present in both prompts but fell after the divergence point are *avoidable* waste.

```python
@dataclass
class DivergenceResult:
    byte_index: int
    token_index: int
    avoidable_tokens: int       # tokens re-read unnecessarily
    unavoidable_tokens: int     # genuinely new tokens
    cause: str | None           # filled by classifier.py
```

### `classifier.py` — cause attribution

Takes a `DivergenceResult` and the two raw message lists and names the cause.

**Causes (in detection order):**
1. `timestamp_in_system` — system prompt diverges; content contains an ISO/unix timestamp pattern at or before the divergence byte. Pattern: `\d{4}-\d{2}-\d{2}|\d{10,13}`.
2. `tool_array_reordered` — tool definitions present in both but in different order.
3. `memory_injected` — a message block present in call N+1 that wasn't in call N, inserted before the tail.
4. `compaction_rewrote_history` — messages[0..k] changed content (not just appended).
5. `new_system_content` — system prompt changed but not a timestamp.
6. `unknown` — fallback.

```python
def classify(prev_messages: list, curr_messages: list, div: DivergenceResult) -> str: ...
```

### `reconciler.py` — predicted vs reported

Compares what the divergence engine predicted against what the provider actually reported.

```python
@dataclass
class ReconciliationResult:
    predicted_cache_read: int
    reported_cache_read: int
    delta: int                  # positive = provider cached more than predicted (good)
    findings: list[str]         # e.g. ["sub-block prompt (< 128 tokens)", "TTL eviction suspected"]
```

Common findings to detect:
- `sub_block` — `reported_uncached == total_input_tokens`: prompt is smaller than the provider's minimum cacheable block (128 tokens for Anthropic). Not a keepwarm bug; log as info.
- `ttl_eviction` — large gap between `timestamp_utc` of consecutive traces in the same family and reported_cache_read dropped to 0. Anthropic TTL is ~5 min.
- `routing_scatter` — predicted high hit rate but reported zero repeatedly; likely provider-side routing variance (flag, don't try to fix).

### `report.py` — output

**CLI (default):**
```
keepwarm report — agent: research_agent, 47 calls

  Cache hit rate:        91%   (predicted achievable: 95%)
  Avoidable re-prefill:  178,000 tokens  →  $2.30 / task
  Status:                ✅ prefix stable across all 47 calls

  Top causes of cache misses:
    1. timestamp_in_system   — 3 calls   — 12,000 tokens wasted
    2. tool_array_reordered  — 1 call    —  4,000 tokens wasted

  Reconciliation: predicted 91%, reported 91% — match ✅
```

**Static HTML:** same data, visual timeline of hit/miss per call, color-coded by cause. Single self-contained file (inline CSS/JS, no external deps).

**Usage:**
```bash
keepwarm profile --session my-agent      # start proxy, print live report on exit
keepwarm report ./keepwarm-traces/       # post-hoc report from saved traces
keepwarm ci ./benchmarks/fixtures/       # CI mode: assert hit rate >= threshold, exit 1 on failure
```

---

## `benchmarks/` — Gate 0 harness

Gate 0 is the go/no-go for the entire compactor. Lane 2 runs this with Lane 1 in week 1.

### `gate0.py`

```python
"""
Gate 0: naive vs cost-aware compaction.

Runs a real long agent (or a recorded trace replay) through two paths:
  1. Naive compaction: summarize oldest turns, rewrite middle of prompt.
  2. Cost-aware compaction: Lane 1's compactor (newest-first, trigger gate).

Reports:
  - Prefill tokens (naive) vs prefill tokens (cost-aware)
  - Prefill dollars (naive) vs prefill dollars (cost-aware)
  - Cache hit rate (naive) vs cache hit rate (cost-aware)

Go condition: cost-aware saves >= 30% prefill dollars on a 100+ step agent.
"""
```

**Inputs:** either a live agent run or a `benchmarks/fixtures/*.json` golden trace.  
**Outputs:** a structured JSON result + printed table; CI-assertable.

### `record.py` + `replay.py`

```python
# record.py: run an agent through the proxy and save the trace to fixtures/
keepwarm record --session gate0-agent --out benchmarks/fixtures/gate0.json

# replay.py: replay a saved trace through the divergence engine without live calls
keepwarm replay benchmarks/fixtures/gate0.json
```

Replay is deterministic and fast — the right tool for CI (no API calls, no cost).

---

## CI integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Profiler regression check
  run: |
    pip install keepwarm[profiler]
    keepwarm ci benchmarks/fixtures/ --min-hit-rate 0.80
```

On failure, the exit code is 1 and the output names the cause:
```
❌ cache hit rate regressed: 84% → 9%
   first divergence at token 312 (cause: timestamp_in_system)
   PR introduced: system_prompt += f" request_id={request_id}"
```

---

## Interfaces — what Lane 2 needs from Lane 1

| Need | When | Fallback |
|---|---|---|
| `RenderedPrompt` / `Block` (Contract 2) | Weeks 3–4, for richer step labels | Degrade gracefully: `rendered=None` in Trace, step labels derived from message role only |
| `CostModel` (Contract 1) | Gate 0 (week 1) | Use a hardcoded Anthropic cost model stub; swap in Lane 1's real impl at Gate 0 |

Lane 2 is **fully independent through week 1.** The only hard sync point is Gate 0 mid-week 1 (trace format alignment) and the Gate 0 pairing session itself.

---

## What Lane 1 needs from Lane 2

| Need | When | How to get it without waiting |
|---|---|---|
| Real cache measurement for Gate 0 | End of week 1 | Until proxy is ready: read `usage` from raw provider responses directly in a thin stub; swap in the real profiler at Gate 0 |
| CI assertion for context layer correctness | Weeks 3–4 | Fixtures + `keepwarm ci` command |

---

## Open questions (resolve in week 0)

1. **Tokenizer for divergence.** `tiktoken` for OpenAI models; what for Anthropic? Options: approximate with `tiktoken` cl100k, or call `anthropic.count_tokens`. Decision affects divergence accuracy. Recommendation: approximate with tiktoken for now, add exact path in v0.2.
2. **Streaming.** Buffer-then-re-stream adds latency. Is that acceptable for the v0.1 profiler, or do we need true streaming passthrough? Recommendation: buffer in v0.1 (simpler), ship streaming in v0.2.
3. **Golden trace agent.** Which real agent do we use for Gate 0 and the launch benchmark? Needs to be: > 50 steps, uses tool calls, publicly runnable. Options: a LangGraph research agent, a CrewAI task agent. Decide in week 0 so Lane 2 can record the fixture in week 1.
