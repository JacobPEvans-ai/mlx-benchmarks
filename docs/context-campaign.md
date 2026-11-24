# Context campaign protocol

> **Retired runner.** The standalone `run-context-campaign.py` /
> `generate-context-inventory.py` / `harness/throughput/run.py` workflow has been
> retired in favor of declarative campaign recipes using `gbench --serving-only`,
> `GuideLLM`, and `vllm bench serve`. Historical `--kind throughput-probe` rows
> remain convertible and readable. The protocol semantics below document the
> historical row dimensions.

The runner owns only the protocol defaults selected for this campaign:
32k, 64k, 128k, and 192k prompt targets; a 512-token output reservation; and
the pair/repetition plan. The generated inventory owns model identity,
enabled status, profile identity, and `window_limit_tokens`.

```json
{
  "campaign_id": "context-20260828-a",
  "output_root": "~/bench-runs",
  "defaults": {
    "targets": [32000, 64000, 128000, 192000],
    "configured_windows": [32768, 65536, 131072, 196608],
    "output_tokens": 512,
    "repeats": 4,
    "concurrency": 1
  },
  "profiles": [
    {
      "model": "<physical-model-id-from-evaluated-config>",
      "window_limit_tokens": 131072
    }
  ]
}
```

Profile `id` is optional and defaults to the physical model basename. Disabled
profiles are omitted by the generator (or marked `"enabled": false`). A cell
whose configured window cannot admit prompt target plus reservation is recorded
as `not_applicable`; an absent live model is `unsupported`. Neither produces a
synthetic speed value.

The historical scripts accepted an evaluated window map alongside live endpoint
`/v1/models` inventory before invoking the retired throughput probe and
publisher in `--dry-run` mode: a cell was rejected when the server did not
report the expected prompt token count (within tolerance), or when actual
prompt plus reserved output exceeded the selected window.

The generator deliberately does not infer a prompt limit from a worker's
`--max-tokens` flag: in the supported MLX server that flag is a default
completion length, not an input-context admission limit. Actual prompt
capacity is therefore established by server usage plus the campaign telemetry,
not by relabeling a generation default as a context window.

## What every successful row means

The envelope records these dimensions independently:

| Dimension | Source | Meaning |
| --- | --- | --- |
| configured window | evaluated inventory | Admission limit selected for this cell. |
| requested prompt | campaign target | Predeclared comparable target. |
| actual prompt | server usage | Value used to accept or reject the cell. |
| synthetic repetitions | server calibration | Input units derived from the target, never a token claim. |
| output reservation | campaign protocol | Capacity held back for a normal completion. |
| catalog maximum | evaluated inventory | Declared prompt-plus-generation window for the selected profile. |

The raw result also preserves cache-busting, completion finish reasons, and
the campaign/cell identity. A long prompt that ends before the required normal
completion is useful for prefill and TTFT evidence, but not a comparable decode
throughput claim.

## Publication and interpretation

Only `success` cells with valid adjacent pairs can affect a ranking or maturity.
The other statuses (`failed`, `capacity_gated`, `unsupported`, `aborted`, and
`not_applicable`) remain durable diagnostics. Preserve historical evidence as
separate immutable rows; do not regenerate, relabel, or overwrite it when a
new campaign runs.

This protocol establishes isolated evidence only. Under-load evidence is a
separate campaign with the production service left live.
