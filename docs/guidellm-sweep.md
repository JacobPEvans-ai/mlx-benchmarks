# GuideLLM reasoning-on serving sweep

GuideLLM measures a **running** server under closed-loop concurrent load with
thinking **on** — the reasoning-on counterpart to the `vllm bench serve`
throughput numbers. The sweep runs concurrency levels 1/2/4/8 with ~1000-token
synthetic prompts, an 8192-token generation cap with EOS respected, a fixed
measured window per level, and two repeats per level
([config](../configs/guidellm/thinking-on-sweep.toml)). GuideLLM is an
external tool (`pip install "guidellm[recommended]"`; flags verified against
the v0.8.1 docs) — this repo ships only the wrapper and the converter.

## Run the sweep

The wrapper sequences one report per (concurrency level, repeat):

```sh
scripts/run-guidellm-sweep.sh \
  --base-url http://127.0.0.1:11434 \
  --model <model-id> \
  --api-key-env OPENAI_API_KEY \
  --output-dir run-output/guidellm-<slug>
```

The base URL is the server **root** (no `/v1`): GuideLLM takes the request
path from `request_format` instead. Publish each report with
`--kind guidellm --suite throughput`; rows are tagged `thinking=on` and
`concurrency=<n>` from the file. The [run rules](RUNBOOK.md) still apply —
one actor per host, replicated pairs, both environment classes.

## Published rows

GuideLLM (`--kind guidellm`, [config](../configs/guidellm/thinking-on-sweep.toml))
publishes under the existing `throughput` suite: it measures a served endpoint
under closed-loop concurrent load, like `vllm bench serve`, so no new suite
value was added. One file per (concurrency level, repeat) — each carries
exactly one benchmark entry. Every row is tagged `thinking=on` and
`concurrency=<streams>`; both come off the result file, not the CLI flags.

Metric naming follows the `--kind vllm` vocabulary
(`throughput_output_toks_per_s`, `throughput_total_toks_per_s`, `ttft_p50_ms`,
`ttft_p99_ms`, `tpot_p50_ms`, ...), plus GuideLLM-specific rows:

- `throughput_output_toks_per_s` — aggregate output tokens/second: the
  duration-weighted mean of GuideLLM's output-token rate distribution over
  the measured window.
- `output_toks_per_s_per_request` — median of the per-request output
  token/second distribution (also stored in
  `campaign_dimensions.speed.decode_tokens_per_second_per_agent`).
- `ttfa_p50_ms` / `ttfa_p99_ms` — time to the first **answer** token:
  GuideLLM's `time_to_first_output_token_ms` (first non-reasoning content
  token). With thinking on, `ttft_*` covers the first *reasoning* token and
  reads orders of magnitude faster than `ttfa_*`; compare answers on `ttfa_*`.
- `request_errors` / `requests_incomplete` / `requests_successful` /
  `requests_total` — status counts from `metrics.request_totals`.
  Incomplete requests were cancelled at the measurement boundary by the load
  generator; they are reported, not scored as answer failures.
- `truncated_or_empty_answers` — completed responses that ended at the
  request's `max_tokens` cap or returned an empty answer. These are
  **failures, not successes**: with thinking on, a response can spend the
  whole budget on reasoning and produce zero answer text. GuideLLM does not
  retain the server `finish_reason`, so "length" is detected as `output
  tokens >= max_tokens`, with the cap read per request from its serialized
  request args (falling back to the backend config's `extras.body.max_tokens`).
  The `failed_truncated` / `failed_empty_answers` tags split the two causes,
  and `truncation_rate` carries the fraction over successful requests. When
  no cap is recorded anywhere, only empty answers are countable and the
  envelope records the gap in `errors[]`.

The envelope schema for these rows: [`schema.md`](schema.md).
