# Benchmark standards

Use these standards for comparable benchmark rows. Store raw results in the
[Hugging Face dataset](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks).
This page records the method and current decision summaries. Browse the
[Hugging Face dataset viewer](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks/viewer/default/train)
for result rows.

## Standards

1. **Engine releases.** Use the latest stable release shown by the upstream
   GitHub `releases/latest` page. Ignore prereleases. Record the engine, version,
   build or commit, and release date on each row. Keep rows from older versions;
   do not replace them.

2. **Fixed-length client.** Use the native grid command
   `vllm bench sweep serve --bench-params <json>` with
   `--ignore-eos --dataset-name random`. Run the fixed-length client from
   `vllm/vllm-openai-cpu:v0.31.0-arm64`. Give every cell a unique seed. For Mac
   MLX servers, pass EOS-token `logit_bias` through `--extra-body`. Do not use
   custom clients or loops.

3. **Concurrency.** Test c = 1, 4, 8, 12, and 16 wherever every request fits the
   KV pool. Also test the ceiling and ceiling + 1, where the ceiling is the
   smaller of the engine's floored capacity at that context and the profile's
   `max_num_seqs` or slot count. Read capacity from the engine: vLLM's startup
   line for maximum concurrency at the requested token length, or llama.cpp's
   `/props` slot count and per-slot `n_ctx`. Skip a standard concurrency only
   when it does not fit, and record the reason.

   Concurrent throughput rows use the same metric mapping as sequential rows
   and record the run's concurrency in the `concurrency` field.

4. **Load shapes.** Report batched load with `--request-rate inf` and staggered
   Poisson load with `--request-rate R --burstiness 1`, at the same maximum
   concurrency. Include a staggered agentic pair in every throughput block:
   4,096 input and 1,024 output tokens, plus at least 32,768 input tokens.

5. **Context.** Include true 128k rows and 256k rows when the model card permits
   them. Never exceed the model card's maximum context.

6. **MTP.** Run matched MTP-on and MTP-off arms for each supported model and
   engine. The Qwen3.8-27B vLLM comparison at 64k × 8 FP8 KV, 1,024/2,048
   tokens, c8 measured 774 output tok/s with two MTP draft tokens and 66% draft
   acceptance; MTP-3 and MTP-4 are being measured. Record depth and the ratio
   of `vllm:spec_decode_num_accepted_tokens_total` to
   `vllm:spec_decode_num_draft_tokens_total` on every MTP row.

7. **Intelligence.** Use quick suites with room below saturation. Drop any
   suite where a compared model scores 90% or higher. Use IFBench through
   EvalScope for the current quick suite; drop saturated MMLU-Pro generative
   CoT, ARC, and GSM8K. For coding-first comparisons, XHIGH is the primary
   intelligence arm; OFF and LOW are context arms. Measure all three arms on
   the Mac Studio too. OFF and LOW being expected serving modes is not a reason
   to skip them. Record each arm as `--reasoning-effort off|low|xhigh`; use
   `enable_thinking: false` for OFF, `reasoning_effort: "low"` for LOW, and
   `reasoning_effort: "xhigh"` for XHIGH. Record `n` and standard error. Claim
   a difference only when it exceeds about two standard errors. Every
   evaluation row reports tokens per task; scored rows also report tokens per
   correct answer. Rank models across intelligence, tokens
   per task, and tok/s.

8. **Judge-scored suites.** Use the router's judge route and its standard
   zero-data-retention fallback rungs. Use an external judge. Benchmark tooling
   must not read provider keys.

9. **Runtime budget.** Declare it before each run: 1–2 h by day or up to 8 h
   overnight, unless the operator approves more. Stop when the budget expires.

10. **Isolation.** Run one serving unit per GPU host during a cell. Prove that
    one unit is active, record the listening port's owner PID, and verify its
    start time follows the converge. Run no foreign clients. On Mac, stop
    resident model workers before the cell and restore them after it.

11. **Environment.** Record ambient temperature, fan duty, power limit,
    dedicated status, and host and guest uptime at block start on every row.
    Set GPU fan duty to 100%. On Mac, use the highest built-in power mode;
    macOS does not expose fan duty. Publish uptime as `host_uptime_s` and
    `guest_uptime_s`. If an uptime read fails, record `null` with its reason.
    Use these machine labels: `RTX PRO 6000 Max-Q 96GB`,
    `RTX 4080 SUPER 16GB`, `Mac Studio M4 Max 128GB`, and
    `2x Apple M4 Max 128GB (Thunderbolt RDMA)`.

12. **Models.** Use trusted sources only: nvidia, unsloth, bartowski,
    ISTA-DASLab, mlx-community, lmstudio-community, the official releaser, or an
    organization with high likes and downloads. Test current-generation models.
    Keep one registry entry for a superseded model and mark it
    `OLD — DONT BOTHER WITH THIS MODEL ANYMORE`. Comment out duplicate entries
    and point them to that registry entry.

13. **Pruning.** Disqualify a model/engine combination after two confirmed tests
    show it well below the host's best with no intelligence edge. Record the
    comparison and why testing stopped.

14. **Local model rows.** Benchmark model scores on local hardware. Do not add
    hosted-API model rows. Keep judge scoring on the existing judge route.

15. **Dates.** Write evaluation deadlines as full ISO dates, such as
    `2026-10-17`.

16. **Headline throughput metric.** Report and compare
    `total_tokens_per_second` — cumulative (prompt + completion) tokens
    divided by wall-clock duration — as the **primary** throughput number,
    not `decode_tokens_per_second`: a decode-only rate hides prefill engine
    improvements entirely. `decode_tokens_per_second` and
    `prompt_tokens_per_second` remain supporting detail for root-causing
    *why* the cumulative number moved. This is a description/policy rule
    only — no schema fields were added, removed, or renamed, so older
    published rows remain fully valid and comparable (since 2026-07-27).

## KV-cache and host-memory standards

Rows use the [HF
dataset](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks) and
record schema cache, workload, and measurement fields. Use `null` plus a
reason when inapplicable.

- **Limits/vLLM.** Set `host_ram_limit_gib=64` with process headroom; keep
  `weight_cpu_offload_gib=0`, `power_limit_w=300`, and `fan_duty_pct=100`.
  Enable prefix caching; record `max_model_len`/`max_num_seqs`. Long/short
  requests share paged KV; freed blocks use LRU, with no TTL.
- **Offload.** Compare off, native 24 GiB, largest safe native, and LMCache at
  the same size. Pin engine/model/connector versions.
- **Workloads.** `fits_gpu`≈0.6×GPU KV; `spills_to_ram`=1.5–2×GPU but below
  GPU+host; `exceeds_all`≈1.5×combined. Use 2,048 shared prompt tokens, ≥32,768
  history, ≥6 rounds, 512 answer tokens. Sample 15 seconds; cap 20 minutes.
- **Prompt stability.** No timestamps/random IDs; fix tool order/template. Qwen
  omits prior reasoning; cache hits stop at the assistant boundary.
- **LMCache.** `multi-round-qa` is unverified; [current
  docs](https://docs.lmcache.ai/cli/bench.html) use `multi-round-chat`. Record
  mapping; cross-check `spills_to_ram` with [vLLM
  `benchmarks/multi_turn`](https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn).

## 2026-10-17 suite decisions

### Saturated or rejected

- GPQA Diamond reached 90% for 27B OFF. AIME 2026 reached 94.1%,
  [HMMT February 2026](https://qwen.ai/blog?id=qwen3.7) reached 97.1%, and
  [LiveCodeBench v6](https://huggingface.co/Qwen/Qwen3.8-27B) reached 90.3%.
  AIME's result is reported by the [Qwen3.6-27B evaluation](https://qwen.ai/blog?id=qwen3.6-27b).
- IFBench XHIGH is near saturation at 89%; use a harder suite for future top-arm
  comparisons. LiveBench's public question snapshot stops at 2024-11-25 and has
  no active coding tasks, so the current snapshot is unsuitable for coding
  evaluation ([official datasets](https://huggingface.co/livebench/datasets)).
- [GitBench](https://gitbench.gitkraken.com/) is saturated and judge-scored;
  [MCPMark Verified](https://mcpmark.ai/leaderboard/verified) reaches 96.06%;
  [o11y-bench](https://o11ybench.ai/) requires a judge and reports a score above
  90%.

### Selected or pending

- **Python:** [BigCodeBench-Hard Complete](https://huggingface.co/datasets/bigcode/bigcodebench-hard),
  30 fixed IDs; [SWE-rebench](https://swe-rebench.com/), 30 Python tasks; and
  [SciCode v3-A](https://qwen.ai/blog?id=qwen3.7), 20 fixed main problems. No
  local scores are claimed here.
- **Scripting:** [Terminal-Bench 4.0](https://www.tbench.ai/news/terminal-bench-4-0),
  20 fixed tasks. A third-party tracker reports about 5.6% for Qwen3.8-27B,
  making this a hard-ceiling test ([tracker](https://frontier.warpcore.app/m/qwen3-8-27b));
  that is not a score from this program.
- [CloudEval-YAML](https://github.com/alibaba/CloudEval-YAML) is selected;
  [WebArena-Verified](https://github.com/ServiceNow/webarena-verified) remains
  a candidate. Keep the deterministic `llm-prompt-evals` cases in the local
  suite.
- NixBench 2.0 is parked.

## 2026-10-17 findings

The [Hugging Face viewer](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks/viewer/default/train)
holds the benchmark result rows. These summaries describe the decisions made
from those runs; this page does not duplicate row data.

- Flash-Next leads the 27B model on the 100-item IFBench XHIGH comparison, 89 to
  79, and on the [2026-06-25 LiveBench leaderboard](https://livebench.ai/),
  76.2 to 62.7. A 30-item
  paired slice is too small to claim parity. On the XHIGH IFBench slice,
  Flash-Next uses more output tokens per item but fewer tokens per correct
  answer.
- On the RTX PRO 6000 Max-Q 96GB, 27B NVFP4 with vLLM 0.31.0 MTP measured
  1,266 tok/s at c16; TPOT-derived potential was 1,447 tok/s and the measured
  one-second peak was 1,633 tok/s. MTP4 measured 167 tok/s at c1 versus 122 for
  MTP2 (+37%); MTP2 was faster at c4. Draft acceptance was 49% for MTP4 and 66%
  for MTP2.
- Long-context throughput is prefill-bound: 128k stays near 27 tok/s at c4–c12;
  256k stays near 8 tok/s. Use c2–c4 for 128k agentic work and c1 for 256k.
  Crossing the KV ceiling at c17 raised TTFT from about 0.25 s to 1.18 s.
- On the Mac Studio M4 Max 128GB, c8 improved short-context throughput by
  2.6–4.5×. The RTX 4080 SUPER 16GB has one slot, so higher concurrency adds no
  capacity. Measure OFF, LOW, and XHIGH on the Studio.
- Disqualified combinations: 27B Q4 GGUF on llama.cpp on the RTX PRO 6000 Max-Q
  96GB (about 3.5× slower than 27B NVFP4 on vLLM across two confirmed tests,
  with no intelligence gain);
  Flash-Next IQ3_S on llama.cpp on that machine (189 versus 245 tok/s for UD-Q2
  at c8, with no measured intelligence edge); 27B Q4 on llama.cpp on the Mac
  Studio (c4/c8 50/72 versus Flash-Next IQ3_S 86/109 tok/s, and lower
  intelligence); and 27B MLX 8-bit on the Mac Studio (55/71 versus 86/109 tok/s,
  and lower intelligence).
- DeepSeek V4.1-Flash-1017 is out of scope because its smallest copy is 318 GB.
  Qwen3-Coder-30B is retired. Nemotron 3.5 Bolt 74B-A9B is a candidate on
  llama.cpp with its official 24 GB GGUF; the vLLM path was dropped because it
  requires a nightly.

## Related docs

- [Runbook](RUNBOOK.md) — run and publish a benchmark.
- [Verdict policy](verdict-policy.md) — validate pairs and compare models.
- [Context campaign](context-campaign.md) — measure context sensitivity.
- [Dataset](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks) — raw results.
