# Use Cases

Each case is a real, documented way frameworks break the prefix, and what the layer does about it. The dollar figures are illustrative (Opus-class input rates, ~10× cached/uncached delta) to show shape, not precision.

---

## 1. The long research/coding agent (the core case)

**Scenario.** A 40-step agent: large system prompt + 20 tool definitions (~20K tokens), growing conversation, runs hundreds of times a day.

**Without the layer.** A timestamp the framework injects into the system prompt makes the prefix unique every call → 0% reuse → every step re-reads all 20K+ tokens at full price. On a 40-step task that's ~800K tokens of pure redundant prefill.

**With the layer.** The timestamp is structurally forced into the tail; the 20K-token prefix is byte-stable across all 40 steps. Cache hit rate goes from single digits to ~90%; per-task input cost drops ~5–10×.

---

## 2. Plan mode / phase switching (tools that change mid-run)

**Scenario.** The agent enters a read-only "plan" phase: it should lose `edit` and `bash`, gain a `plan` tool.

**Without the layer.** Swapping the tools array rewrites part of the cached prefix (tools sit before the conversation), invalidating the entire history after it — every plan-mode toggle triggers a full re-read.

**With the layer.** `ctx.set_active_tools([...])` never mutates the tools array. The full tool set stays present in frozen order; "which tools are live now" is written as a tail-side constraint. Switching phases costs nothing in cache terms.

---

## 3. Mid-run memory updates

**Scenario.** A Mem0/Letta-style store writes a new fact ("user prefers metric units") mid-conversation.

**Without the layer.** The write lands in the middle of the context and detonates everything after it on the next call.

**With the layer.** Memory lives in a dedicated zone with its own breakpoint. A write costs the re-read of *only the memory zone*, not the whole conversation — bounded, predictable blast radius.

---

## 4. Hitting the context limit (compaction — the anchor case)

**Scenario.** A 200K-token session must be compacted to keep going.

**Without the layer.** Standard compaction summarizes the *oldest* turns and rewrites the middle of the prompt. That rewrite changes a low message index, so the cache busts from that point on *every subsequent turn* — and the re-prefill cost often exceeds the tokens the compaction saved.

**With the layer.** The compactor optimizes total cost, not token count: it compacts **newest-first** so the cached prefix stays byte-identical, summarizes into a **stable trailing block**, and only fires when prefill saved exceeds prefill destroyed. Compaction stops being a cache-bust event.

---

## 5. Multi-provider / self-hosted portability

**Scenario.** Same agent runs on Anthropic in prod, a self-hosted vLLM box for batch jobs, and OpenAI for a fallback.

**Without the layer.** Each backend has different breakpoint rules (Anthropic explicit `cache_control` + 20-block lookback; OpenAI automatic; vLLM prefix-cache). Teams hand-tune per provider and regress constantly.

**With the layer.** One zone-based layout; the renderer emits the right breakpoint strategy per backend. Write the agent once, stay cache-optimal everywhere.

---

## 6. CI regression guard (the paid hook, later)

**Scenario.** A PR adds a per-request `request_id` to the system prompt for debugging.

**Without the layer.** Cache hit rate quietly drops from 84% to 9%; nobody notices until the monthly bill 5×'s.

**With the layer.** The profiler runs in CI against a recorded trajectory and fails the PR: "cache hit rate regressed 84% → 9%, first divergence now at token 312 (`request_id`)." The regression never ships.

---

## Non-use-cases (be honest about these)

- **Short chatbots (1–3 turns).** Not enough repeated prefix to matter.
- **Single-shot completions.** No loop, nothing to keep warm.
- **Teams already fully on Claude Code / Codex / Deep Agents.** Those harnesses already solved this internally; the value here is for everyone *outside* them.
