# promptstack suite — system prompt as the independent variable

> **Retired.** Prompt evaluation moved to the promptfoo-based
> `dryvist/llm-prompt-evals` repo — see
> [`prompt-eval-framework.md`](prompt-eval-framework.md) for the decision
> record and migration path. The custom runner and configs have been retired
> from this repository while published promptstack shards and the
> `--kind promptstack` converter remain intact for historical dataset records.

Every other suite in this repo holds the system prompt fixed and varies the
model. `promptstack` does the opposite: it holds the **model** fixed and
varies the **system prompt** — `base_plus_variant` (the shared behavioral
base plus a surface's identity/tools delta) vs. `current` (whatever that
surface runs today). This is what decides whether a prompt-surface change
gets adopted.

Converter: [`src/mlx_benchmarks/converters/promptstack.py`](../src/mlx_benchmarks/converters/promptstack.py)
(retained for historical raw JSON conversions; the standalone runner was retired).

## What it measures

Four deterministic probe classes, originally stored as frozen task banks:

| Probe class | Measures | Scoring |
| --- | --- | --- |
| `reasoning` | Multi-step / numeric correctness | Last number in the response must equal the task's answer |
| `tool_call` | Tool-call validity + scope discipline | Schema-valid call for positive tasks; no fabricated call for negative tasks |
| `instruction` | Instruction-following under checkable constraints | Fraction of per-task constraints met |
| `homelab_qa` | Grounded Q&A over a provided evidence bundle | Every expected fact present; forbidden terms flagged separately |

`tool_call` has a **negative bank**: tasks where no tool fits, or a required
argument is missing. The correct behavior is to refuse or ask — the runner
scores a fabricated call as a failure, not a success. `homelab_qa` tasks carry
plausible-but-wrong "forbidden terms" alongside the expected facts; any of
those appearing in the answer counts as an unsupported claim, tracked
separately from correctness. These two checks are what test the base
prompt's own "ground truth before claims" and "explicit scope, no guessing"
rules — a suite that only rewards positive answers can't see either failure
mode.

Every task runs `--repeats` times per (prompt_variant, thinking) cell.
Per probe class, per cell, the runner reports `task_success_rate`, class-specific
extras (`valid_tool_call_rate`, `unsupported_claim_rate`, `instruction_adherence_rate`),
average prompt/completion tokens, and latency p50/p95.

## Historical invocation and publishing

Historically, the retired `promptstack` runner targeted OpenAI-compatible `/v1`
endpoints per surface (`<surface>.txt` vs `current-<surface>.txt`). Existing raw
outputs still publish with:

```bash
mlx-bench-publish run-output/promptstack_<slug>.json \
  --kind promptstack --suite promptstack \
  --published-metadata run-output/published-metadata.json
```

## Adoption rule (the Cline bar)

A prompt variant replaces a surface's current prompt only when, per probe
class:

```text
task_success_rate(base_plus_variant)    >= task_success_rate(current)
valid_tool_call_rate(base_plus_variant) >= valid_tool_call_rate(current)
unsupported_claim_rate(base_plus_variant) <= unsupported_claim_rate(current)
tokens_completion(base_plus_variant)    <= tokens_completion(current)
```

"≥ success and validity at ≤ tokens." A variant that wins on quality but
costs more tokens does not auto-adopt — it goes back for trimming. Ties on
quality break toward fewer tokens. This mirrors the [verdict
policy](verdict-policy.md)'s consecutive-pair discipline: don't score or
publish a single unreplicated run.

## Where this sits relative to the other suites

`promptstack` is not part of the "fully benchmarked" model-ranking set
(`throughput`, `coding`, `math-hard`, `reasoning`, `tool-calling` — see
[`AGENTS.md`](../AGENTS.md)). It answers a different question: not "how good
is this model," but "does this prompt change help, on this model." Results do
not feed [`RANKINGS.md`](../RANKINGS.md); they gate prompt-surface pull
requests in the repos that own those surfaces.
