# Benchmark traps + serving reference

The serving parser map, the serving flags that bite, and the traps checklist
that [`RUNBOOK.md`](RUNBOOK.md) links into. If a result looks wrong, walk the
checklist before blaming the model. Per-model-class failure modes live in
[`model-notes.md`](model-notes.md).

Traps 1-12 here are harness-usage traps (how to run lm-eval/mlx-bench
correctly) — `RUNBOOK.md` links into these by anchor. Three further
checklists continue the numbering without repeating it:
[benchmark-traps-model.md](benchmark-traps-model.md) (traps 13-18,
model/eval-specific failure modes),
[benchmark-traps-ops.md](benchmark-traps-ops.md) (traps 19-29,
infrastructure/operational traps — not specific to benchmarking), and
[benchmark-traps-consistency.md](benchmark-traps-consistency.md) (trap 30,
multi-source-of-truth disagreement).

## Parser map

Wrong `--tool-call-parser` → empty `function.name` repair storms and 500s at
swap-in. Match the family exactly:

| Family | `--tool-call-parser` | `--reasoning-parser` | Notes |
| --- | --- | --- | --- |
| Qwen3 / Qwen3.6 general + Instruct (dense, MoE) | `hermes` | `qwen3` | Production-verified for the resident brain; **not** `qwen3_coder` |
| Qwen3-Coder / Coder-Next | `qwen3_coder` | `qwen3` | Coder builds only |
| Qwen3-Next (hybrid attention) | `hermes` | `qwen3` | Never spec-decode/MTP; no prefix cache |
| GLM-4.7-Flash | `glm47` | `glm45` | Thinking on by default |
| gpt-oss (harmony) | `harmony` | `gpt_oss` | Add `--disable-prefix-cache`; thinking via `reasoning_effort` |
| DeepSeek-V4-Flash | `deepseek` (+ `--enable-auto-tool-choice`) | `deepseek_r1` | Native `<think>` |
| Hermes-4 | `hermes` | — | — |

## Serving flags that bite

- **`--timeout` defaults to 300 s** and acts as a disconnect guard that kills any
  generation running longer — which a legitimate long agentic turn does. Use
  `--timeout 3600` for any agent-brain serving or bench.
- **`--gpu-memory-utilization`** sets the KV-cache clear trip
  (`device_mem × (util + 0.05)`); 0.80 standard, never >0.85. Keep
  `Σ(weights + caches) < trip < wired ceiling` (see
  [RUNBOOK Step 2](RUNBOOK.md#step-2--fit-check-capacity-rules)).
- **gpt-oss (harmony)** additionally needs `--disable-prefix-cache` — its
  alternating sliding-window attention is incompatible with the paged/prefix
  cache and throws `[broadcast_shapes]` failures otherwise.

## Traps checklist

### Trap 1: plain coding extractors misfire on chat output

Plain `lm-eval` `humaneval`/`mbpp` score ~0 on chat-served models (the upstream
extractor grabs the echoed prompt instead of fenced Python blocks). Prefer
`gbench --evals-only --evals lcb bigcodebench` (`--kind gbench`) or
`evalscope/livecodebench` for chat-served coding models.

### Trap 2: read math_verify

On `math-hard`, read the `math_verify` metric, not `exact_match` (prose-depressed
on chat output).

### Trap 3: lm-eval tasks flag

lm-eval (0.4.x) needs `--tasks a,b`. Positional task names silently select zero
tasks and the run reports success with no data.

### Trap 4: mlx-bench loads directly

`mlx-bench --model <id> --prompts 10` loads its own copy of the model. The server
must be DOWN, or the weights double-load and OOM the host.

### Trap 5: both thinking tracks

Agentic: run thinking ON and OFF, judge at conc4 + thinking-ON + large-ctx,
record the multi-turn `first_degraded_round` for each. Single-shot validity is
not a passing verdict; serve to match the winning track.

### Trap 6: sampling parity

A bench winner can misbehave in production if the sampling differs. The agentic
bench uses client defaults (temp 1.0). Production requests with
`temperature=None` + `repetition_penalty=None` let 4-bit quants degenerate into
repetition loops (same sentence 100+ times, ~37 duplicate tool calls/turn). When
a bench winner misbehaves live, check the **sampling delta first**. Guardrail for
4-bit Qwen-family: `repetition_penalty ~1.05`, `temp 0.6–0.7` in thinking mode.

### Trap 7: parser map

Wrong `--tool-call-parser` → empty `function.name` storms. Qwen3 general/Instruct
→ `hermes` (not `qwen3_coder`); Qwen3-Coder → `qwen3_coder`; GLM-4.7 → `glm47`;
gpt-oss → `harmony` (+ `--disable-prefix-cache`). Full map [above](#parser-map).

### Trap 8: serving flags

`--timeout` defaults to 300 s and kills long agentic generations — use `3600`.
`--gpu-memory-utilization` also sets the KV-cache trip
(`device_mem × (util+0.05)`); 0.80 standard, never >0.85. Keep
`Σ(weights+caches) < trip < wired ceiling`.

### Trap 9: publish token

Ambient `HF_TOKEN` is read-only, and `mlx-bench-publish` reads the literal
env var `HF_TOKEN`. The environment normally carries both `HF_TOKEN`
(read-only) and `HF_WRITE_TOKEN` (write), and the read-only one is what the
publisher sees. Result: a 403 ("you must use a write token") on the
real publish call. **`--dry-run` does not catch this** — it validates the
envelope and schema but never makes the HTTP call that checks token scope, so
a dry-run can pass clean and the real publish still 403. Override `HF_TOKEN`
explicitly:

```sh
HF_TOKEN="$HF_WRITE_TOKEN" .venv/bin/mlx-bench-publish …
```

### Trap 10: run hygiene

Results land in `~/bench-runs*/` per host. Chain long runs with `nohup` and a
`===== <date> START/DONE <model> =====` log convention, and monitor those marker
lines to know where a chain is. Approximate suite timings (30B-A3B class):
coding ~3 h, math ~45 min, reasoning ~2.5–4 h, agentic full grid ~30 min.

### Trap 11: concurrency cascade masquerades as a serving failure

A driver run against a local llama-swap/MLX endpoint that returns a flood of
`429 {"error":"Too many requests"}` is **over-concurrency**, not a broken
model. llama-swap accepts `concurrencyLimit` in-flight requests; exceeding it
returns 429, the aiohttp session dies (`Session is closed` /
`ServerDisconnected`), and lm-eval's exception handler then throws
`UnboundLocalError: ... 'outputs'` — masking the real 429. Symptom on disk:
crashed tasks and invalid/zero results (the 2026-07-08 campaign: 9,262×429).

Fix: cap client concurrency to the endpoint's limit (`MLX_EVAL_CONCURRENT` /
`num_concurrent` / `--concurrency` = the endpoint's `concurrencyLimit`).
Verify: `grep -c 'Too many requests' <queue-log>` on a good run is 0.
Also expect **bimodal** concurrency scaling (2026-07-11 sweep): concurrent
requests either join one continuous batch (1.6–2.3× aggregate) or serialize
(~1.0×) — never assume a single c2 sample characterizes the endpoint.

### Trap 12: cold start folds into row 1 — warm before measuring

The first request after a model (re)load carries the whole cold-load cost
(measured: an 8.55 s 32-token first request vs 0.55 s warm — a +8 s artifact).
A naive sweep silently folds that into its first row. Always record the
pre-request residency state (`cold`, `resident`, or `unknown`), fire a
throwaway readiness request, archive its timing separately, and then measure
the warmed run. Never score or rank the readiness request. Guard two-point decode math with
`dt > 0` (a cold row makes the slope negative). For TTFT, count the first SSE
`data:` chunk with content — `time_starttransfer` only measures header
arrival (llama-swap flushes SSE headers immediately).
