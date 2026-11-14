# Evaluation result fields

| Field | Type | Description |
| --- | --- | --- |
| `output_tokens_mean` | number \| null | Mean output tokens per sample. |
| `output_tokens_median` | number \| null | Median output tokens per sample. |
| `output_tokens_p90` | number \| null | Nearest-rank p90 output tokens per sample. |
| `reasoning_tokens_mean` | number \| null | Mean reasoning tokens per sample when logged. |
| `tokens_per_correct` | number \| null | Total output tokens divided by estimated correct items. |
| `truncation_rate` | number \| null | Fraction of samples that reached the output-token limit. |
| `reasoning_arm` | string | Declared arm, or `unstated` if absent. |

Evaluation rows also publish `tokens_per_task` as a `results[]` metric. Use
observed output-token usage per scored task; scored rows include
`tokens_per_correct`. Compare token cost with the quality score and matching
tok/s. These fields use the existing v1 contract.

EvalScope and lm-eval read sibling sample logs. Inspect AI reads per-sample
`model_usage` from `.eval` logs. Converters prefer API usage counts and fall back
to tokenizing saved response text. The p90 uses nearest rank;
`tokens_per_correct` is total output tokens divided by the rounded estimate
from score ratio times sample count. Missing values are `null` with a
`<field>_unavailable_reason` tag.

## Derived speed field calculation

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
