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

## Open holes (deliberate)

- Real per-provider breakpoint placement (Anthropic 4 / ~20-block lookback,
  OpenAI auto, vLLM prefix-cache). `BreakpointStrategy` is the hook;
  `DefaultBreakpointStrategy` is a placeholder.
- Real summarization. `fake_summarize` is deterministic and just enough to
  prove layout + cost accounting.
- Tokenizer-accurate token counts. `default_token_estimator` is bytes/4;
  swap in a real tokenizer per backend later.
- Framework adapters (LangGraph, CrewAI). Out of scope for Lane 1 foundation.
