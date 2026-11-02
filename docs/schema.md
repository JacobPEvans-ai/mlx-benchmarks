# Envelope schema (v1)

Canonical schema: [`schema.json`](../schema.json). It wins when this guide differs.

Upstream Hub source URLs and the read date are recorded in
[`huggingface-metadata-sources.md`](huggingface-metadata-sources.md).

## Published score rows

See [`published-results.md`](published-results.md) for the fields, Hub
enrichment rules, validator, and historical migration procedure.

## Required top-level fields

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | `"1"` | Bump only on breaking changes. |
| `timestamp` | ISO 8601 UTC | Start-of-run time, not end. |
| `git_sha` | 7–64 hex or `null` | Repo SHA; `null` if not captured. |
| `trigger` | enum | How the run was kicked off; see the allowed values below. |
| `suite` | enum | Must be in the closed set below. |
| `model` | string | HF model ID (e.g. `mlx-community/Qwen3.5-9B-MLX-4bit`). |
| `system` | object | See below; `os`, `chip`, `memory_gb` are required. |
| `results` | array | Per-measurement rows; empty arrays cannot be published. |

Allowed trigger values: `local`, `schedule`, `pr`, and `workflow_dispatch`.

Closed suite set: `throughput`, `throughput-serving`, `ttft`, `mixed-context`,
`elastic`, `max-context`, `tool-calling`, `code-accuracy`,
`framework-eval`, `capability-comparison`, `coding`, `reasoning`,
`knowledge`, `evalplus`, `math-hard`, `promptstack`, `grounded-summary`, plus the
four hardware baselines `gpu-burn`, `nvbandwidth`, `mbw` and `fio` (no model under
test: publish them with `--model hardware-baseline`). Framework evaluations can
carry `results[].tags.domain` values `spl`, `tool-queries`, `agentic`, `triage`,
or `homelab-access`. The `repo-qa` suite records deterministic answers to
repository questions over packed context. Related runs should keep using an
existing suite such as `tool-calling` when that suite already describes the
measurement. Adding a suite means editing `schema.json` and filing a schema
update PR.

## Optional top-level fields

| Field | Type | Added when |
| --- | --- | --- |
| `pr_number` | integer \| null | `trigger == "pr"` |
| `env_class` | `isolated \| under-load` | Load class (policy gate 3). |
| `concurrency` | integer (≥1) | In-flight request count. |
| `reasoning_effort` | string, free-form | Declared level; see below. |
| `serving` | object | Server identity. New public rows may include `stack` and `served_model`; the publisher rejects legacy `endpoint_port`. |
| `model_revision` | string | Model provides HF revision or commit SHA. |
| `model_task` | string | Card `pipeline_tag`, else closest official HF task. |
| `model_task_source` | `model_card \| inferred` | Source for `model_task`. |
| `quantization` | string | Runtime reports it (e.g. `mlx-4bit`, `mxfp4`). |
| `skipped` | boolean | Suite intentionally skipped (CI without hardware). |
| `seed` | integer | Seeded generation. |
| `gen_kwargs` | object | `max_gen_toks` / `temperature` / `top_p` / `top_k`. |
| `memory_snapshots` | array | Future work — RSS / swap per phase. |
| `errors` | array of string | Non-fatal warnings recorded during the run. |
| `campaign` | object | Immutable id, cell id, and serving profile. |
| `cell_status` | enum | Only `success` rows are scored. |
| `context` | object | Catalog limits, window, prompt, and output reservation. |
| `campaign_dimensions` | object | Typed fields; leaves may be `null`. |
| `dimension_null_reasons` | object of string | `field.path` null reasons. |
| `readiness` | object | First request, excluded from warmed scoring. |

### `campaign_dimensions`

The object is optional and leaves may be `null`. The publisher flattens leaves
into stable `campaign_<group>_<field>` Parquet columns; selected object fields
use a `_json` suffix. The normalizer fills new columns with null unless the run
index supplies a per-shard projection.

`dimension_null_reasons` maps paths such as `run.kv_cache_dtype` to a concise
reason code. Use a reason for each explicit null in newly enriched campaign
envelopes; a missing property remains distinct from an observed-but-unknown
value.

### Cache fields

See [cache fields](cache-fields.md) for typed dimensions, units, and mappings.
Explicit nulls include a `dimension_null_reasons` entry.

### MoE expert offload fields

The optional `campaign_dimensions.run.moe_cpu_offload` field records `none`,
`ncmoe`, or `cpumoe`. `campaign_dimensions.run.moe_cpu_layers` records the
number of model layers covered by CPU expert offload; use `0` for `none` and
the full model layer count for `cpumoe`.

Converters keep both objects. `mlx-bench-publish --campaign-dimensions PATH`
reads a JSON file with a `campaign_dimensions` object and an optional
`dimension_null_reasons` object and copies them onto the envelope for every
`--kind`; `throughput-probe` also keeps them when its result file carries them.

See [campaign dimensions](campaign-dimensions.md) for the full field groups,
JSON projection details, serving-stack values, and optional power enrichment.

See [Stage 0](system-load-test.md).

### `reasoning_effort`

An arm is weights *plus* quant *plus* effort *plus* serving config, so two
efforts are two arms and a row without one cannot be placed against another.

Stored verbatim, and deliberately not an enum: each model family names its own
levels, so rewriting `on` into `high` merges arms that are not the same arm.

Declared by the caller, never inferred. Effort lives in the agent CLI's config
or the serving default, and a harness can see neither. Where nothing was
declared the value is `unstated` — **absent is not `off`**, and a plausible
default here would be indistinguishable from a measurement.

## `system` object

Required: `os`, `chip`, `memory_gb`.

Optional (all populated automatically by `detect_system()`):
`hostname` (removed from the public Parquet projection), `python_version`,
`mlx_version`, `mlx_lm_version`, `lm_eval_version`, `kernel`,
`runner` (for GitHub Actions), `vllm_mlx_version`.

NVIDIA / CUDA hosts add four optional fields (absent on Apple Silicon runs):

| Field | Type | Notes |
| --- | --- | --- |
| `gpu` | object | Optional model, VRAM, driver, and CUDA version. |
| `engine` | object | Engine name and version; complements `serving.stack`. |
| `power_limit_w` | number (≥0) | Enforced GPU board power limit. |
| `container` | string | Engine image; absent for bare-metal runs. |

These are **declared, not probed**: `detect_system()` reads them from
`MLX_BENCH_GPU_MODEL`, `MLX_BENCH_GPU_VRAM_GB`, `MLX_BENCH_GPU_DRIVER`,
`MLX_BENCH_GPU_CUDA`, `MLX_BENCH_ENGINE_NAME`, `MLX_BENCH_ENGINE_VERSION`,
`MLX_BENCH_POWER_LIMIT_W` and `MLX_BENCH_CONTAINER`, because the publisher may
run on a different machine than the GPU. Unset, blank, or invalid numeric
variables are omitted. The `nvidia-smi` snippet that fills them is in
[`configs/nvidia/README.md`](../configs/nvidia/README.md). In the
Parquet shard, `gpu` and `system_engine` are JSON-string columns (like
`topology`); `power_limit_w` and `container` are plain columns.

`topology` (object, multi-node runs only): `world_size`, `parallelism`
(`pipeline` / `tensor` / `none`), `interconnect` (e.g. `tb5-rdma`), and
`nodes[]` (`hostname` / `chip` / `memory_gb` per node). Populated from
`MLX_BENCH_WORLD_SIZE` / `MLX_BENCH_PARALLELISM` / `MLX_BENCH_INTERCONNECT` /
`MLX_BENCH_NODES` (JSON array); absent for single-node runs.

## `results[]` items

| Field | Type | Notes |
| --- | --- | --- |
| `name` | string (required) | Task or measurement ID. |
| `metric` | string (required) | Display metric name. |
| `value` | number (required) | The measurement. |
| `unit` | string (required) | `ratio`, `tok/s`, `ms`, etc. |
| `duration_seconds` | number | Wall-clock time; replaces the tag. |
| `prompt_tokens_per_second` | number | Prefill: prompt tokens / time. |
| `decode_tokens_per_second` | number | Decode: completion tokens / time. |
| `total_tokens_per_second` | number | **Headline:** total tokens / time. |
| `first_token_latency_ms` | number | First-token time for streaming runs. |
| `peak_rss_mb` | number | Peak per-result RSS, when available. |
| `tags` | object\[string\] | Free-form metadata; numeric values are strings. |
| `raw` | any | Original untransformed tool output (optional archive). |

See [evaluation result fields](evaluation-result-fields.md) for per-sample
metrics, data sources, calculations, and null-reason tags.

### Derived throughput dimensions

The optional `campaign_dimensions.speed` fields below are calculated from
measured vLLM serving results and are repeated on every published row in that
run. Missing inputs are stored as `null`; `dimension_null_reasons` records a
reason at `speed.<field>`.

- `potential_output_tok_s_p50`: `max_concurrency * 1000 / p50_tpot_ms`.
- `potential_output_tok_s_p90`: `max_concurrency * 1000 / p90_tpot_ms`.
- `potential_output_tok_s_p99`: `max_concurrency * 1000 / p99_tpot_ms`.
- `measured_mean_concurrency`: sum of request latencies / run duration.
  Each request latency is `ttft + sum(itls)`.
- `potential_output_tok_s_p50_at_mean_concurrency` (also p90/p99): for finite
  request rates, measured mean concurrency * 1000 / matching TPOT in ms.
- `peak_output_tok_s_1s_max`: maximum tokens in any fixed one-second window.
- `peak_output_tok_s_1s_p99`: nearest-rank p99 of one-second window token counts.
- `potential_efficiency`: `output_throughput / potential_output_tok_s_p50`.

vLLM `start_times`, `ttfts`, and `itls` are seconds; start times are absolute
monotonic values. For each request, token-chunk arrivals are `start_time +
ttft + cumulative itls`. If the request has fewer chunks than output tokens,
each chunk is assigned an equal fractional share of that request's output
length, so the shares sum exactly to `output_lens`. Times are rebased to the
earliest request start and bucketed into non-overlapping `[n, n + 1)` second
windows. The first token's bucket through the last token's bucket are included;
empty buckets between them count as zero. Buckets before the first token's
bucket or after the last token's bucket are excluded. Non-detailed and other
engine rows keep unavailable values null with a reason rather than treating
missing timing arrays as zero.

Sweep metadata may include:

- `prompt_tokens`: tokens in each request prompt.
- `concurrency`: in-flight request count, matching the top-level field.
- `context_len`: configured server window capacity in tokens, not prompt length.

### Headline throughput metric: `total_tokens_per_second`

As of 2026-07-27, `total_tokens_per_second` — cumulative (prompt + completion)
tokens divided by wall-clock duration — is the **primary** throughput number
to report and compare, not `decode_tokens_per_second`. A decode-only rate
hides prefill-engine improvements entirely, even though a faster prefill is a
real, felt latency win for any caller sending a non-trivial prompt: two models
with identical decode speed but a 4-6x prefill gap are not equivalent in
practice, and a decode-only headline reports them as if they were.
`decode_tokens_per_second` and `prompt_tokens_per_second` are kept as
supporting detail (useful for root-causing *why* the cumulative number moved)
— this is purely a description/policy change, no schema fields were added,
removed, or renamed, so older published rows remain fully valid and
comparable.

## Validation

Every envelope and published score row is validated inside
`mlx_benchmarks.publish.publish()`. Validation cannot be bypassed. The
validators collect *all* errors before raising, so a single run-through
surfaces everything wrong instead of one-at-a-time iteration.

Locally:

```python
from mlx_benchmarks.envelope import validate_envelope

# Raises EnvelopeValidationError with every problem
validate_envelope(my_envelope)
```

## Versioning

- Adding an optional field: non-breaking, no version bump.
- Adding an enum value to `suite`: non-breaking (downstream just ignores
  unknown suites in current viewers), but please file as a `feat:` PR.
- Removing or renaming a field, changing a type, tightening validation,
  changing the filename pattern: breaking. Bump `schema_version` to `"2"`
  and update `$id` accordingly.

Current `$id`: `https://github.com/JacobPEvans/mlx-benchmarks/schema/v1.json`.
