# Lane 1 — Context & Compaction

Foundation for the context layer (B) and compactor (C) described in `architecture.md`.
The profiler/proxy (Lane 2, component A) is intentionally absent here — it will
consume `RenderedPrompt` from `keepwarm.contracts`.

## Layout

```
src/keepwarm/
  contracts.py            # Contract 1 (CostModel) + Contract 2 (Block / RenderedPrompt)
  context/
    context.py            # Context API, zones, canonicalization, breakpoints
  compaction/
    compactor.py          # Naive vs cost-aware compactors, Gate 0 spike
tests/
  test_contracts.py
  test_context.py
  test_compaction.py
  test_gate0.py           # the fixture for Gate 0
```

## Install & run

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

To run only the Gate 0 fixture:

```bash
python3 -m pytest tests/test_gate0.py -v
```

## What Gate 0 demonstrates

On a long rendered prompt (`stable` system+tools block + 8 tail turns):

- `NaiveOldestFirstCompactor` rewrites an early block → first cache divergence
  lands near the front of the prompt → the next prefill re-reads everything
  after it at uncached price.
- `CostAwareNewestFirstCompactor` rewrites a batch at the tail end → the
  pre-batch prefix stays **byte-identical** → the next prefill keeps almost
  everything cached.
- Under `FlatCostModel` (cached = 10% of uncached), cost-aware's next-call
  prefill cost is strictly lower than naive's. The test asserts this directly.

This is the mechanical scaffold for the real Gate 0 measurement: swap
`FlatCostModel` for an Anthropic-priced model and `fake_summarize` for an
actual summarizer to get a dollar number on a real agent trajectory.

## Where Lane 2 plugs in

- `keepwarm.contracts.Block` / `RenderedPrompt` is Contract 2. The proxy will
  emit `Trace` objects (Contract 3) carrying `Block.stable_hash` so divergence
  attribution lines up across Lane 1 and Lane 2.
- `keepwarm.contracts.CostModel` is Contract 1. The compactor depends on it
  through the protocol; the profiler will share the same models for reporting.
- `Context.render()` is the single seam where the proxy can capture the
  rendered prompt before it goes on the wire.

## Realistic Gate 0 — go / no-go

The toy fixture proves the mechanics. The realistic fixture is what decides
whether Lane 1's full context layer is worth building. Run:

```bash
python3 -m pytest tests/test_gate0_realistic.py -v
```

The fixture (`tests/fixtures/realistic_prompts.py`) builds a deterministic
long-agent prompt: large system prompt, 10–25 tools, a volatile memory block,
an `active_tools` constraint, and 30–90 mixed user / assistant-reasoning /
tool-result tail blocks. Provider-shaped pricing comes from
`keepwarm.cost_models` (`AnthropicLikeCostModel`, `OpenAILikeCostModel`).

### Metrics reported per strategy

- `prompt_tokens` — total token count after the strategy runs.
- `first_changed_index` — index of the first block whose hash differs from
  the original. Everything before it is a cache hit on the next call.
- `prefix_preserved_tokens` — tokens cached on the next call (sum of
  block tokens before `first_changed_index`).
- `destroyed_cache_tokens` — previously-cached tokens after the modification
  point that lose cache because the prefix changed.
- `next_call_prefill_cost` — dollar cost of the very next prefill, given the
  cost model's cached/uncached split.
- `recurring_saved_tokens` / `recurring_saved_cost` — tokens (and dollars) the
  compaction stops sending on every future call.

### Sample run (large fixture, Anthropic-shaped, batch_size=8)

|              | fired | tokens | first_div | prefix kept | destroyed | next-call cost | recur saved |
| ---          | ---   | ---    | ---       | ---         | ---       | ---            | ---         |
| baseline     | —     | 15,993 | 94        | 15,993      | 0         | $0.004798      | 0           |
| naive        | ✓     | 15,121 | **4**     | 3,670       | 11,439    | **$0.035454**  | 872         |
| cost_aware   | ✓     | 15,234 | **86**    | 15,222      | 0         | **$0.004603**  | 759         |

Naive shrinks the prompt by 872 tokens but pays for it by destroying 11,439
tokens of cache on the *next* call — net 7× more expensive than doing nothing.
Cost-aware compacts almost exactly as many tokens *and* keeps the prefix
intact, so the next call is *cheaper than baseline*.

### Go criteria (all must hold)

- Cost-aware preserves materially more prefix than naive
  (`first_changed_index` strictly larger).
- Cost-aware's estimated next-call prefill cost is lower than naive's.
- Cost-aware's recurring savings exceed its destroyed-cache cost under at
  least one realistic provider-shaped model.
- Behaviour is deterministic and covered by tests.

Today's state: ✅ on all four under both `AnthropicLikeCostModel` and
`OpenAILikeCostModel` (see `tests/test_gate0_realistic.py`).

### No-go signals (track these as we go)

- The realistic fixture flips: cost-aware ≤ naive on either prefix preservation
  or next-call cost.
- Recurring savings shrink to near-zero once the summarizer's overhead is real
  rather than a constant token-count stub.
- A provider's cache semantics (e.g. extremely small block size, no
  breakpoints) eliminate the headroom the compactor is trading on.

If any of those happen we should kill or radically rescope the compactor
before investing in the full context layer.

## Open holes (deliberate)

- Real per-provider breakpoint placement (Anthropic 4 / ~20-block lookback,
  OpenAI auto, vLLM prefix-cache). `BreakpointStrategy` is the hook;
  `DefaultBreakpointStrategy` is a placeholder.
- Real summarization. `fake_summarize` is deterministic and just enough to
  prove layout + cost accounting.
- Tokenizer-accurate token counts. `default_token_estimator` is bytes/4;
  swap in a real tokenizer per backend later.
- Framework adapters (LangGraph, CrewAI). Out of scope for Lane 1 foundation.
