# keepwarm

**A drop-in context layer that stops your AI agent from re-reading the same tokens on every step.**

Works with LangGraph, CrewAI, the OpenAI Agents SDK, or your own loop. You keep your orchestrator. `keepwarm` takes over how the prompt is assembled and compacted so your KV cache actually gets reused — and it ships with a profiler that proves it's working.

---

## The problem (in plain terms)

An AI agent doesn't make one call to the model. It makes a *loop* of them: think → call a tool → read the result → think again → call another tool, often 50–200 times to finish one task.

Here's the catch: **every single call re-sends the entire conversation so far.** Your system prompt, all your tool definitions, every previous step — all of it, every time. Step 50 re-sends everything from steps 1 through 49.

Models don't have to *recompute* all that repeated text — there's a feature called **prompt caching** (or KV-cache reuse) that lets them skip it and pick up where they left off. Cached tokens cost up to **10× less** and return much faster. On a real long-running agent this is the difference between a task costing `$0.40` and `$4.00`.

But prompt caching only works on an **exact prefix match.** The cache is reused only up to the *first character that changed* from the last call. Change one byte early in the prompt and everything after it gets re-read at full price.

And here's why this bites everyone: **agent frameworks break that prefix constantly, without telling you.** Real, documented examples:

- A **timestamp** in the system prompt → a new prefix on every single call → 0% cache reuse.
- **Tool definitions re-sorted** between calls (a JSON serializer ordering keys differently) → the whole prefix invalidates.
- A **memory/state update** injected into the middle of the context → everything after it gets re-read.
- **Compaction** (summarizing old turns to save space) rewrites the middle of the prompt → detonates the cache and often costs *more* in re-reading than it saved in length.

Teams usually discover this only when the bill arrives, or when someone spends an afternoon hand-diagnosing why follow-up turns got 8–16× slower. There's no tool that fixes it across frameworks, and the parts that are genuinely hard — keeping the cache alive when the agent has to *change its own context mid-run* — nobody packages at all.

## What "breaks the prefix" looks like

```
Call N:    [ system prompt ][ tools ][ history... ]        ← all cached ✅
Call N+1:  [ system prompt ][ tools ][ history... ][ new ] ← only "new" is re-read ✅  (good!)

Call N+1:  [ system prompt + TIMESTAMP ][ tools ][ history... ]
                              ↑ first divergence here
           everything from this point on is re-read at full price ❌  (bad!)
```

Same information. One is nearly free. The other re-reads tens of thousands of tokens every turn. The only difference is *where the changing bytes live.*

---

## The solution

`keepwarm` is a **context layer** that sits between your agent code and the model. It owns one job: assemble and compact your prompt so the cacheable prefix stays stable, automatically.

You don't adopt a new framework. You keep LangGraph / CrewAI / your own loop. You just build context through `keepwarm` instead of hand-assembling strings, and it makes a cache-breaking layout **impossible to express** — the way an ORM makes SQL injection hard to write by construction instead of nagging you about it.

### 1. Zones, not strings

You declare *what kind* of content each piece is. `keepwarm` handles ordering, serialization, and cache-breakpoint placement so stable content stays at the front and volatile content stays at the tail — where it can't poison the prefix.

```python
from keepwarm import Context

ctx = Context()

# STABLE zone — written once, frozen, lives at the front (cached)
ctx.stable.instructions(SYSTEM_PROMPT)
ctx.stable.tools(ALL_TOOLS)              # canonical order, locked

# VOLATILE zone — changes during the run, sandboxed to its own region
ctx.volatile.memory(user_memory)

# TAIL — the only thing that grows every step
ctx.tail.user(latest_message)

prompt = ctx.render()   # cache-optimal layout + breakpoints, every time
```

Put a timestamp in `stable` and it refuses. Volatile content physically cannot land in the cached prefix. The footgun is gone.

### 2. Cache-safe primitives for the hard cases

The easy stuff above (good ordering) the best harnesses already do. The reason `keepwarm` exists is the part nobody packages: **keeping the prefix alive when the agent changes its own context mid-task.** Three cases:

**Tools that change mid-run.** The naive design swaps tools in and out — and every swap nukes the prefix. `keepwarm` keeps your full tool set always present in a frozen order and expresses "which tools are live right now" as a tail-side instruction. Availability changes; the prefix doesn't.

```python
# does NOT mutate the tools array — writes a constraint to the tail
ctx.set_active_tools(["search", "read"])   # e.g. entering a read-only plan phase
```

**Memory that updates mid-run.** Writes go into a dedicated memory zone with its own breakpoint, so updating a fact costs you the re-read of *only* the memory zone — not the whole conversation.

**Compaction that doesn't detonate the cache.** This is the anchor. Normal compaction rewrites the middle to save tokens and pays for it in re-reading. `keepwarm`'s compactor optimizes the *real* objective — total cost, not token count:

- compacts **newest-first**, so the cached prefix stays byte-identical and only the tail diverges,
- summarizes into a **stable trailing block** instead of rewriting history,
- and only fires when the prefill it would **save** exceeds the prefill it would **destroy**.

```python
ctx.compact()   # cost-aware: preserves the prefix, triggers only when it pays
```

### 3. The profiler — proof it's working

Measurement ships *inside* the layer as a built-in assertion. It watches the model's real cache fields (`cache_read_input_tokens`, `cached_tokens`) and tells you the truth — including if your own code reached around the API and broke something.

```
keepwarm report — agent: research_agent, 47 calls

  Cache hit rate:        91%   (was 31% before keepwarm)
  Re-prefill saved:      178,000 tokens  →  $2.30 / task
  Status:                ✅ prefix stable across all 47 calls
```

---

## Why this and not the alternatives

| Approach | What it does | Why it's not enough |
|---|---|---|
| **Inference servers** (vLLM, SGLang) | Reuse cache *if* the prefix is stable | They can't fix a prefix your framework already broke before the request arrived |
| **Observability** (Langfuse, LangSmith) | Show token counts and latency | They tell you *that* you spent tokens, not that they were avoidable or why |
| **Harness built-ins** (Claude Code, Codex) | Cache-aware prompt assembly | Locked inside *their* harness — useless if you're on LangGraph, CrewAI, or your own loop |
| **A whole new framework** | Owns orchestration *and* context | Huge adoption tax — rewriting your agent to switch orchestrators |
| **keepwarm** | Owns *only* context assembly + compaction, plugs into your existing orchestrator | — |

The seam: every framework has an orchestration story and a weak-to-nonexistent **cache-stable-context** story. `keepwarm` is that missing layer — and it doesn't ask you to leave the framework you already use.

`keepwarm` is not competing with your inference server. It makes its caching actually hit.

---

## Status & scope

- **v0.1 — the profiler.** Drop-in proxy + report. Measures real cache-hit rate, finds the first-divergence point, names the cause (`tool array reordered`, `timestamp at offset 312`, `compaction rewrote middle`), and prices the waste. Deterministic, no model calls. This is the "prove the problem is real on *your* agent" tool.
- **v0.2 — the context layer.** Zones, cache-safe tool/memory primitives, and the cost-aware compactor. The fix.

The honest bet: the layout/zoning is defensible-but-absorbable, but the **cache-aware compactor is both hard and unclaimed** — that's the anchor. If you only try one thing first, prototype the compactor against a real long-running agent and measure prefill-dollars saved versus a naive compactor. That single number decides whether this is worth building.

## License

MIT (core). Hosted profiles, CI regression gates, and fleet dashboards are the paid layer.
