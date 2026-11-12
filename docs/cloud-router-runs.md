# Cloud-router benchmark runs

Run these against an OpenAI-compatible router with the existing `lm_eval`,
EvalScope, and agentic harnesses. They do not load local model weights. The
run and publish commands are declared in [`configs/`](../configs/LAYOUT.md);
each section below names the config entries to execute. Set `ROUTER_BASE_URL`
to the router's `/v1` base URL, `ROUTER_API_KEY` to its key, and
`ROUTER_TOKENIZER` to the tokenizer files `lm_eval` should use. Config files
contain environment-variable names, not URL or key values.

Current router aliases: `qwen4-max-preview` and `qwen3.8-max`. Export `MODEL`
as one of them and `MODEL_PROVIDER` as its provider namespace (`Qwen` for
both) — the configs and commands below read both. Before running an alias,
confirm it against the live catalog: [RUNBOOK Step 1](RUNBOOK.md#step-1--identify-and-size-the-model)
and [`configs/LAYOUT.md`](../configs/LAYOUT.md#local-vs-cloud-execution)
require that check for every benchmark target, cloud endpoints included.

Cloud endpoint envelope identity and Hub lookup bypass were added in [PR #516](https://github.com/JacobPEvans-ai/mlx-benchmarks/pull/516),
branch `agent/zcode/j-ec94182ab0903ba0` (merged to `main`). Publish cloud
aliases with `--model-kind cloud-endpoint --model-provider "$MODEL_PROVIDER"`.

## Existing indexed rows and endpoint coverage

`metadata/run-index-v1.json` has Qwen3.8-27B rows in all eight expected suites.
The indexed status is not uniformly scored:

| Suite | Qwen3.8-27B rows | Endpoint run |
| --- | --- | --- |
| reasoning | 7 experimental | Yes, `lm_eval` GSM8K |
| math-hard | 8 experimental | Yes, `lm_eval` Minerva Math |
| throughput | 73 scored, 86 experimental, 41 invalid | Excluded; this suite needs locally loaded weights/engine |
| tool-calling | 3 experimental, 1 recovered | Yes, `harness/agentic/run.py` |
| knowledge / MMLU-Pro | 1 scored | Yes, if `/completions` returns prompt logprobs |
| framework-eval / IFBench | 5 scored, 1 experimental | Yes, EvalScope OpenAI API |
| framework-eval / repo-qa | 6 scored | Excluded; this repo has no run config or endpoint runner |
| coding / BigCodeBench-Hard | 4 scored, 1 invalid | Excluded; no converter, and EvalScope scoring is invalid ([policy](verdict-policy.md#benchmark-lessons)) |

The index's cloud-like historical rows name Gemini 2.5 Flash, Gemini 3 Flash,
GPT-5 mini, and GPT-5.4 mini. They are unclassified experimental rows, not
scored frontier entries. The current measured frontier in
[`RANKINGS.md`](../RANKINGS.md#verdict-pareto-frontier-over-speed-and-tool-calling-2026-09-10)
is Qwen3.6-35B-A3B-4bit and Qwen3-Coder-30B-A3B-Instruct-4bit. The rankings
catalog lists `gemini-2.5-flash`, `openrouter/auto`, and `openai/gpt-4.1-mini`
as older cloud-smoke references, outside that frontier.

## Tool-calling pilot and full matrix

[`configs/agentic/tool-calling.toml`](../configs/agentic/tool-calling.toml)
is the single source of truth for these commands: with the environment above
exported, execute its profiles in order.

- `[run.router_pilot]` — 3 repeats at concurrency 1, both thinking settings,
  both contexts, streaming — then `[publish.router_pilot]`.
- `[run.router_full]` — 10 repeats across concurrency 1/4, both thinking
  settings, both contexts, streaming and non-streaming, plus the 20-round
  multi-turn track — then `[publish.router_full]`.

## Reasoning, math-hard, and knowledge

These commands use the existing [reasoning](../configs/lm-eval/reasoning.toml),
[math-hard](../configs/lm-eval/math-hard.toml), and
[MMLU-Pro](../configs/lm-eval/quick-intelligence.toml) task definitions.
MMLU-Pro requires the router to support `/completions` prompt logprobs; skip
that run if it cannot return them. Each `lm_eval` output directory contains
`results_*.json`, which is passed to the shown publisher command. The `gsm8k`
run is a `--limit 100` subset, so its shard publishes with a caveat tag.

```sh
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-chat-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/chat/completions,\
model=$MODEL,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks gsm8k --batch_size 1 --num_fewshot 5 --limit 100 \
  --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs max_gen_toks=1024 --log_samples \
  --output_path run-output/cloud-router/$MODEL/reasoning
mlx-bench-publish run-output/cloud-router/$MODEL/reasoning/results_*.json \
  --kind lm-eval --suite reasoning --model "$MODEL" \
  --model-kind cloud-endpoint --model-provider "$MODEL_PROVIDER" \
  --tag caveat=gsm8k-limit-100-subset
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-chat-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/chat/completions,\
model=$MODEL,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks minerva_math500 --batch_size 1 --num_fewshot 0 \
  --apply_chat_template --fewshot_as_multiturn --log_samples \
  --output_path run-output/cloud-router/$MODEL/math-hard
mlx-bench-publish run-output/cloud-router/$MODEL/math-hard/results_*.json \
  --kind lm-eval --suite math-hard --model "$MODEL" \
  --model-kind cloud-endpoint --model-provider "$MODEL_PROVIDER"
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/completions,\
model=$MODEL,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks leaderboard_mmlu_pro --batch_size 1 --num_fewshot 5 --log_samples \
  --output_path run-output/cloud-router/$MODEL/knowledge
mlx-bench-publish run-output/cloud-router/$MODEL/knowledge/results_*.json \
  --kind lm-eval --suite knowledge --model "$MODEL" \
  --model-kind cloud-endpoint --model-provider "$MODEL_PROVIDER"
```

## IFBench

[`configs/evalscope/ifbench-openai.toml`](../configs/evalscope/ifbench-openai.toml)
is the single source of truth: export `MODEL`, `OUTPUT_PATH`,
`ROUTER_BASE_URL`, `ROUTER_API_KEY`, and `MODEL_PROVIDER`, then execute its
`[run]` command followed by its `[publish]` command (for example,
`OUTPUT_PATH=run-output/cloud-router/$MODEL/ifbench`). EvalScope's
[OpenAI API runner](https://github.com/modelscope/evalscope/blob/main/docs/en/benchmarks/ifbench.md)
writes `reports/<model>/ifbench.json` beneath the work directory.
