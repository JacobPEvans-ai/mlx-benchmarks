# Benchmark traps — model/eval-specific (13-18)

Continues [benchmark-traps.md](benchmark-traps.md#traps-checklist) (traps
1-12, harness-usage). These are failure modes specific to how a particular
model or eval task behaves, not to running the harness itself. Next:
[benchmark-traps-ops.md](benchmark-traps-ops.md) (traps 19-29).

## Traps checklist

### Trap 13: `mlx-eval`'s default model is host-specific

The wrapper's `${MLX_DEFAULT_MODEL:-<default>}` bakes in a *different*
default per host (e.g. `Qwen3.8-27B-4bit` on one machine,
`Qwen3-Coder-30B-A3B-Instruct-4bit` on another). A run that omits
`MLX_DEFAULT_MODEL` silently scores whatever that host's default happens to
be — no error, a plausible-looking result attributed to the wrong model.
Always pass `MLX_DEFAULT_MODEL=<full model id>` explicitly; never rely on the
wrapper default across hosts.

### Trap 14: task-level `until` stop sequences can zero a model's output

Some lm-eval tasks (e.g. `arc_challenge_chat`) carry a built-in
`until: ['\n\n', '.']`. If a model's answers structurally begin with one of
those strings (Qwen3.8-27B's do, with `"\n\n"`), the stop sequence fires
before any content is emitted — empty `resps`, and lm-eval logs
`"Could not parse generations: 'content'"`. This is a different failure than
the reasoning-budget-exhaustion case (`finish_reason=length` with real
reasoning tokens burned) — this one is an instant zero-length stop, silent
otherwise. Pass `--gen_kwargs "...,until=[]"` on every suite run against a
model with this answer shape.

### Trap 15: `exact_match` filters can score a correct model as zero

`arc_challenge_chat`'s `remove_whitespace`/`exact_match` filter expects a
bare letter answer. A model that reasons in full prose ("The best answer is
C") scores `exact_match=0` even when correct — measured: 15/15 zero while the
model was actually right in 13-14 of 15 (verified by reading `filtered_resps`
against `target`, not clipped — responses topped out at ~330 tokens against a
4096 cap). Same shape as the coding extraction issue
([trap 1](benchmark-traps.md#trap-1-plain-coding-extractors-misfire-on-chat-output)),
but for reasoning: the extractor, not the model, is broken. Use a
flexible-extract task (`gsm8k`) or `gbench` instead of `arc_challenge_chat`
for models with this answer shape, and never report an `exact_match` number
without reading samples first.

### Trap 16: a standalone-server quant comparison isn't a production number

Serving through `llama-swap` applies per-model `filters.setParams` (e.g.
`frequency_penalty`/`presence_penalty`) that a bare standalone `mlx-lm-server`
does not — routing one arm of a comparison through llama-swap and the other
standalone samples them differently even at temperature 0, confounding the
one thing the comparison exists to isolate. Fix: run every arm as a
standalone server with byte-identical flags (only the model id differs).
Consequence: label results **"isolated from serving filters"** — they do not
describe production-through-llama-swap behavior. If the arms are also run
sequentially (e.g. hours apart, one quant fully before the next), only
per-item accuracy is a valid cross-arm comparison — temperature-0 greedy
decode makes correctness robust to ambient load, but latency/throughput is
not, so any timing delta between arms is within-arm descriptive only, never
attributed to the quant.

### Trap 17: a health gate that checks a claim, not an observation, can pass while false

A standalone server's `/v1/models` is a **claim** the process makes about
itself — an orphaned process from a prior run can keep answering that
endpoint with a stale or unrelated catalog after the port it's bound to was
meant to be freed. Health-gating on `grep <model-id> <(curl .../v1/models)`
only proves the string appears somewhere in that claim, not that the model
you asked for is the one actually generating. This nearly published a
complete, plausible 8-bit result set that was actually the 4-bit model still
resident from a prior phase — correct-looking sample counts, correct-looking
accuracy, no error anywhere, and nothing downstream could have told the
difference.

Two compounding causes, both worth guarding against on this stack
specifically:

- **`pkill -f "mlx-lm-server --model ..."` does not match the real process.**
  The actual binary launched is a wrapper, `mlx-lm-launch.py`, whose command
  line does not contain the literal string `mlx-lm-server` — a pattern kill
  aimed at the server name misses it, leaving an orphan alive on the port.
  Kill by **port** (`lsof -nP -tiTCP:<port> -sTCP:LISTEN`), not by a guessed
  process-name pattern.
- **A model *list* can be a static/stale claim; only a completion is an
  observation.** Verify readiness with a real chat completion and check the
  response's own `model` field against what was requested — a single-model
  server cannot lie about what it actually generated with, in the way a
  `/v1/models` listing can be stale, cached, or (as here) foreign.

The general lesson extends past this repo: a gate that asks the system what
it *offers* can pass on a claim; a gate that checks what the system just
*did* checks a fact. Prefer the latter for anything a wrong answer would
silently corrupt.

### Trap 18: a handful of sequential samples cannot characterize a bursty tier

A handful of sequential sub-second completions against a shared endpoint were
read as "the tier is idle and fast," while the server's own request log
showed a sustained, high rejection rate over the same window. A quiet minute
coexists fine with heavy contention on a `concurrencyLimit`-gated endpoint —
a handful of samples just has a good chance of landing in a gap. To
characterize load on a shared serving tier, read the server's own request
log for status-code distribution over a real window, not a handful of your
own latency probes.
