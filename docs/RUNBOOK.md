# RUNBOOK — benchmark any model on any host

Benchmark on Apple Silicon hosts, publish to the
[HF dataset](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks), and
update [`../RANKINGS.md`](../RANKINGS.md).

This repo defines the result contract and publisher. Serving tools are thin
wrappers; this runbook gives their order and publishing rules
([traps + parser map](benchmark-traps.md)).

## What "fully benchmarked" means

A model is fully benchmarked only when all five required suites have a published
shard:

| Suite | `--kind` | `--suite` | Measures |
| --- | --- | --- | --- |
| Throughput | `vllm` | `throughput` | Output tokens/sec under batched load |
| Coding | `lm-eval` | `coding` | `humaneval`/`mbpp` pass@1 (chat overlay) |
| Math | `lm-eval` | `math-hard` | `minerva_math500` `math_verify` |
| Reasoning | `lm-eval` | `reasoning` | `arc_challenge_chat` / `gsm8k` |
| Agentic | `agentic` | `tool-calling` | Valid structured tool calls under load |

A model may be **disqualified for a role** before every suite runs, but the
catalog row in `RANKINGS.md` is "complete" only with all five present — and
still **provisional**: a verdict is final only per the
[verdict policy](verdict-policy.md) (**≥4 runs ≥5 days apart, validated pairs,
both environment classes**). Read it before any "best/worst" claim.

`promptstack`, `factual`, `coding-replay` (real-merged-PR replay suite —
invocation and scoring in [`coding-replay.md`](coding-replay.md)), and
`dense-layout-image-text` (`harness/dense_layout/`) feed no `RANKINGS.md` row.

## Decision tree

```text
model arrives
  └─ 1. Identify + size it   → resident weight GB, architecture, quant recipe
        └─ 2. Fit check      → Σ(weights + caches) < GPU trip < wired ceiling?
              ├─ fits MacBook → run there (workstation; concurrencyLimit=2)
              └─ needs room   → run on Studio jevans-ms (128 GB)
                    └─ 3. Serve → per environment class:
                          ├─ ISOLATED    → solo (managed window if it can't co-reside)
                          └─ UNDER-LOAD  → production stays live; NO managed window
                          └─ 4. Run the 5 suites — each a replicated pair (mind every trap)
                                └─ 5. Validate the pair, then publish (write token)
                                      └─ 6. Update RANKINGS.md; verdict PROVISIONAL, schedule re-bench
```

## Environments

Pick a host by fit and by who else needs the machine.

| | MacBook (workstation) | Mac Studio `jevans-ms` |
| --- | --- | --- |
| Memory | (workstation) | 128 GB unified, **wired ceiling ~118 GB** |
| Serving | `llama-swap`, `concurrencyLimit=4` (nix-ai#1190) | nix `dev.vllm-mlx.server` LaunchAgent |
| Endpoint | `http://localhost:11434/v1` | `http://127.0.0.1:11434/v1` (IPv4 plain HTTP) |
| Concurrency | **`MLX_EVAL_CONCURRENT` = `concurrencyLimit` (4)** | up to 4 |
| HF cache | default | `/Volumes/HuggingFace` (`HF_HOME`) |
| Role | benches compete with your work | production serving host (Hermes) |

Two host rules that silently ruin a run if missed:

- **MacBook:** `MLX_EVAL_CONCURRENT` must equal deployed `concurrencyLimit`
  (**4** since nix-ai#1190); higher values cause 429/lm-eval crashes (trap 11).
- **Studio: use `curl -s4 127.0.0.1`, never a hostname.** Caddy owns IPv6/TLS;
  vllm-mlx is IPv4 plain HTTP.

## Run a benchmark block end to end

See the [complete benchmark block procedure](benchmark-block-runbook.md).

**One actor at a time.** Never edit the `llama-swap` config, restart serving, or
start a second `vllm-mlx serve` while a bench is in flight — the second loader
contends for GPU memory and both runs corrupt or OOM. Check for a live
`mlx-eval`/`mlx-bench`/`vllm-mlx`/`run.py` and a bench-chain log heartbeat
([trap 10](benchmark-traps.md#trap-10-run-hygiene)) first.

## Step 1 — Identify and size the model

1. **Confirm the served name against the live catalog** — never trust a doc or a
   filename:

   ```sh
   curl -s4 http://127.0.0.1:11434/v1/models | grep -o '"id":"[^"]*"'
   ```

2. **Estimate resident footprint.** ~0.55 GB/B (4-bit), ~1.1 GB/B (8-bit); a MoE
   keeps its *whole* weight set resident — size by total params, not active. Add
   the KV cache budget (Step 2).

3. **Note architecture + quant recipe** — they pick the
   [parser map](benchmark-traps.md#parser-map) flags. Quant *recipe*
   (OptiQ/DWQ vs a stock uniform quant) matters more than bit
   width for agentic fitness — see [`model-notes.md`](model-notes.md).

## Step 2 — Fit check (capacity rules)

The GPU-memory-utilization flag also sets the point at which vllm-mlx **clears
the KV cache**:

```text
kv_cache_clear_trip = device_memory_gb × (gpu_memory_utilization + 0.05)
```

The invariant to satisfy:

```text
Σ(resident weights + KV caches for all resident models) < trip < wired ceiling
```

- `--gpu-memory-utilization 0.80` is standard. **Never exceed 0.85** — above it
  the trip crowds the wired ceiling (~118 GB on the Studio) and the machine
  swaps or the server is OOM-killed.
- If weights + caches don't clear the trip with headroom, the model doesn't fit
  on that host at that utilization — move to the Studio, drop to a smaller quant,
  or serve it solo.

## Step 3 — Serve the model

Serve in **each** environment class (a full verdict needs both): Option A =
**under-load** (production live), Option B = **isolated**. Record the class +
what else was running (`llama-swap` `/running` + load avg) with the run.

### Option A — existing `llama-swap` slot (no downtime) = under-load class

If the model is already in `llama-swap`, target the endpoint; it loads on first
request. Default on the MacBook and for any Studio run not needing a solo model.
Don't edit the swap config mid-run.

### Option B — solo `vllm-mlx serve` in a managed window (Studio) = isolated class

When the model isn't in the swap config, or you need the whole machine's memory
for one large model, take a **managed window** on the Studio. This takes
production serving (the Hermes brain) offline, so:

> **Notify the user before opening a managed window, and restore after.**
> Production Hermes is down for the duration.

With session approval for benchmark priority, log approval, window start/end,
services paused/restored, and readiness in the task. Approval never waives
paired, context, concurrency, or environment rules.

```sh
# 1. Bootout serving AND everything that could relaunch it mid-window
#    (the 30s mlx-night.watcher agent; KeepAlive resurrects a plain `kill`).
for a in vllm-mlx.server mlx-night.watcher mlx-night.rank mlx-night.prefetch; do
  launchctl bootout "gui/501/dev.$a" 2>/dev/null || true
done
# Rotation flips (00:00Z/12:00Z) need no pause: their curls fail harmlessly.
# To freeze across flips, touch the per-router rotation-paused sentinel
# (apps docs/BRAIN_ROTATION.md). Permanent policy = ai_rotation_enabled.

# 2. Serve the target model solo (parser flags from the parser map + Step 2)
vllm-mlx serve <model-id> \
  --port 11434 \
  --tool-call-parser <parser> \
  --reasoning-parser <parser> \
  --gpu-memory-utilization 0.80 \
  --timeout 3600

# 3. ... run your suites against http://127.0.0.1:11434/v1 ...

# 4. Restore (server first; the warmup agent re-faults the residents)
for a in vllm-mlx.server vllm-mlx.warmup mlx-night.watcher mlx-night.rank mlx-night.prefetch; do
  launchctl bootstrap gui/501 ~/Library/LaunchAgents/dev.$a.plist 2>/dev/null || true
done

# 5. Verify before closing the window: every resident "ready"
curl -s http://127.0.0.1:11434/running
```

Mind the [serving flags that bite](benchmark-traps.md#serving-flags-that-bite).
On the Studio, HF auth may be unset — a model that needs downloading reads
`HF_TOKEN` (cache on `/Volumes/HuggingFace`).

## Step 4 — Run the required suites

Run against the served endpoint. Timings are **one** 30B-A3B-class run — each
suite runs as a **replicated pair** (×2) in **both** environment classes, so
budget ~**4×**; discard + re-run a diverging pair.

### 4a. Throughput (`--kind vllm|throughput-probe --suite throughput`)

- **Running server** — vllm `benchmark_serving` ([config](../configs/vllm/benchmark_serving.toml)).
- **Load** — `mlx-bench` loads the model; server down first ([trap 4](benchmark-traps.md#trap-4-mlx-bench-loads-directly)).
- **Local probe** — publish as `throughput-probe`: cumulative tok/s is headline;
  decode/prefill/TTFT support it. Capped results are not quality.

### 4b. Coding (`--kind lm-eval --suite coding`) — ~3 h

Plain `humaneval`/`mbpp` score ~0 on chat-served models. **Always use the
overlay** ([trap 1](benchmark-traps.md#trap-1-coding-overlay-is-mandatory)):

```sh
HF_ALLOW_CODE_EVAL=1 MLX_EVAL_CONCURRENT=4 mlx-eval \
  --include_path configs/lm-eval/qwen3-tasks \
  --tasks humaneval_instruct_qwen3,mbpp_instruct_qwen3 \
  --confirm_run_unsafe_code --log_samples \
  --output_path ./run-output/<slug>
```

Coding benchmarks **execute model-generated code** — read
[`../SECURITY.md`](../SECURITY.md) before running outside a sandbox.

### 4c. Math (`--kind lm-eval --suite math-hard`) — ~45 min

```sh
MLX_EVAL_CONCURRENT=4 mlx-eval minerva_math500 --output_path ./run-output/<slug>
```

Read the **`math_verify`** metric, not `exact_match`
([trap 2](benchmark-traps.md#trap-2-read-math_verify)).

### 4d. Reasoning (`--kind lm-eval --suite reasoning`) — ~2.5–4 h full

`leaderboard_mmlu_pro` (`--samples` 420, `--suite knowledge`) for
a quick pass, `gsm8k` for the canonical run. Use `--tasks a,b`
([trap 3](benchmark-traps.md#trap-3-lm-eval-tasks-flag)). Exclusion
rationale: [`../configs/lm-eval/reasoning.toml`](../configs/lm-eval/reasoning.toml).

### 4e. Agentic tool-calling (`--kind agentic --suite tool-calling`) — ~30 min

The decisive suite for an agent brain; the runner lives in this repo:

```sh
uv run harness/agentic/run.py \
  --base-url http://127.0.0.1:11434/v1 \
  --api-key-env OPENAI_API_KEY \
  --model <model-id> \
  --output run-output/agentic_<slug>.json
```

Run **both** thinking tracks; judge at the pass gate (concurrency 4, thinking
ON, large context) — the multi-turn degradation track is where quants fail
([trap 5](benchmark-traps.md#trap-5-both-thinking-tracks)). Ship the winning
track's serving config. Full grid + pass gate: [`agentic.md`](agentic.md).
For router commands, see [cloud-router runs](cloud-router-runs.md).

## Step 5 — Publish each shard

Validate with `schema.json`; dry-run first. Use `HF_WRITE_TOKEN`, not read-only
`HF_TOKEN` ([trap 9](benchmark-traps.md#trap-9-publish-token)). vLLM reads
`date`/`duration`; `vllm`/`mlx` map to `vLLM`/`MLX`. Null model/unregistered
dataset fields use `not_applicable`.

```sh
.venv/bin/mlx-bench-publish run-output/<...>.json --kind vllm --suite throughput \
  --campaign-dimensions run-output/<...>/campaign-dimensions.json \
  --published-metadata run-output/<...>/published-metadata.json --dry-run
HF_TOKEN="$HF_WRITE_TOKEN" .venv/bin/mlx-bench-publish run-output/<...>.json --kind lm-eval --suite reasoning \
  --campaign-dimensions run-output/<...>/campaign-dimensions.json \
  --published-metadata run-output/<...>/published-metadata.json --unregistered-benchmark
```

See the [published-result contract](published-results.md). Retain completed
runs and publish caveats with `--tag caveat=<reason>`.

## Step 6 — Update RANKINGS.md

In the **same PR** as the publish, update the affected
[`../RANKINGS.md`](../RANKINGS.md) row (numbers pulled back via that file's
["Keeping this page current"](../RANKINGS.md#keeping-this-page-current) loop).
Bump **Maturity** only for a validated pair ≥5 days out; keep the verdict
**provisional** per the [policy](verdict-policy.md).

## See also

- [Standards](benchmark-standards.md).
- [Verdict policy](verdict-policy.md) — compare models and set verdicts.
- [Traps](benchmark-traps.md) — parser map and serving flags.
- [`agentic.md`](agentic.md) · [`promptstack.md`](promptstack.md) ·
  [`model-notes.md`](model-notes.md) — suite depth, per-class quirks.
- [`../configs/LAYOUT.md`](../configs/LAYOUT.md) · [`../RANKINGS.md`](../RANKINGS.md)
  · [`../examples/`](../examples/) — configs, leaderboard, walkthroughs.
