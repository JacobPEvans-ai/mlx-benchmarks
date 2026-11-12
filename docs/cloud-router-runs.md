# Cloud-router benchmark runs

Run these against an OpenAI-compatible router with the existing `lm_eval`,
EvalScope, and agentic harnesses. They do not load local model weights. The
commands assume `ROUTER_BASE_URL` is set to the router's `/v1` base URL,
`ROUTER_API_KEY` contains its key, and `ROUTER_TOKENIZER` names the tokenizer
files `lm_eval` should use. Config files contain environment-variable names,
not URL or key values.

Cloud endpoint envelope identity and Hub lookup bypass were added in [PR #516](https://github.com/JacobPEvans-ai/mlx-benchmarks/pull/516),
branch `agent/zcode/j-ec94182ab0903ba0` (merged to `main`). Publish cloud
aliases with `--model-kind cloud-endpoint --model-provider Qwen`.

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

Declarations are in [`configs/agentic/tool-calling.toml`](../configs/agentic/tool-calling.toml).
The pilot runs 3 repeats at concurrency 1, both thinking settings, both
contexts, and streaming. The full matrix runs 10 repeats across concurrency
1/4, both thinking settings, both contexts, streaming and non-streaming, plus
20 multi-turn rounds per thinking setting.

For `qwen4-max-preview`:

```sh
uv run harness/agentic/run.py \
  --base-url "$ROUTER_BASE_URL" --api-key-env ROUTER_API_KEY \
  --model qwen4-max-preview --repeats 3 --concurrency 1 \
  --thinking on,off --context small,large --stream stream \
  --output run-output/cloud-router/qwen4-max-preview/tool-calling/pilot.json
mlx-bench-publish run-output/cloud-router/qwen4-max-preview/tool-calling/pilot.json \
  --kind agentic --suite tool-calling --model qwen4-max-preview \
  --model-kind cloud-endpoint --model-provider Qwen --tag stage=pilot
uv run harness/agentic/run.py \
  --base-url "$ROUTER_BASE_URL" --api-key-env ROUTER_API_KEY \
  --model qwen4-max-preview --repeats 10 --concurrency 1,4 \
  --thinking on,off --context small,large --stream stream,nostream \
  --multiturn-rounds 20 \
  --output run-output/cloud-router/qwen4-max-preview/tool-calling/full.json
mlx-bench-publish run-output/cloud-router/qwen4-max-preview/tool-calling/full.json \
  --kind agentic --suite tool-calling --model qwen4-max-preview \
  --model-kind cloud-endpoint --model-provider Qwen
```

For `qwen3.8-max`:

```sh
uv run harness/agentic/run.py \
  --base-url "$ROUTER_BASE_URL" --api-key-env ROUTER_API_KEY \
  --model qwen3.8-max --repeats 3 --concurrency 1 \
  --thinking on,off --context small,large --stream stream \
  --output run-output/cloud-router/qwen3.8-max/tool-calling/pilot.json
mlx-bench-publish run-output/cloud-router/qwen3.8-max/tool-calling/pilot.json \
  --kind agentic --suite tool-calling --model qwen3.8-max \
  --model-kind cloud-endpoint --model-provider Qwen --tag stage=pilot
uv run harness/agentic/run.py \
  --base-url "$ROUTER_BASE_URL" --api-key-env ROUTER_API_KEY \
  --model qwen3.8-max --repeats 10 --concurrency 1,4 \
  --thinking on,off --context small,large --stream stream,nostream \
  --multiturn-rounds 20 \
  --output run-output/cloud-router/qwen3.8-max/tool-calling/full.json
mlx-bench-publish run-output/cloud-router/qwen3.8-max/tool-calling/full.json \
  --kind agentic --suite tool-calling --model qwen3.8-max \
  --model-kind cloud-endpoint --model-provider Qwen
```

## Reasoning, math-hard, and knowledge

These commands use the existing [reasoning](../configs/lm-eval/reasoning.toml),
[math-hard](../configs/lm-eval/math-hard.toml), and
[MMLU-Pro](../configs/lm-eval/quick-intelligence.toml) task definitions.
MMLU-Pro requires the router to support `/completions` prompt logprobs; skip
that run if it cannot return them. Each `lm_eval` output directory contains
`results_*.json`, which is passed to the shown publisher command.

For `qwen4-max-preview`:

```sh
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-chat-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/chat/completions,\
model=qwen4-max-preview,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks gsm8k --batch_size 1 --num_fewshot 5 --limit 100 \
  --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs max_gen_toks=1024 --log_samples \
  --output_path run-output/cloud-router/qwen4-max-preview/reasoning
mlx-bench-publish run-output/cloud-router/qwen4-max-preview/reasoning/results_*.json \
  --kind lm-eval --suite reasoning --model qwen4-max-preview \
  --model-kind cloud-endpoint --model-provider Qwen
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-chat-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/chat/completions,\
model=qwen4-max-preview,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks minerva_math500 --batch_size 1 --num_fewshot 0 \
  --apply_chat_template --fewshot_as_multiturn --log_samples \
  --output_path run-output/cloud-router/qwen4-max-preview/math-hard
mlx-bench-publish run-output/cloud-router/qwen4-max-preview/math-hard/results_*.json \
  --kind lm-eval --suite math-hard --model qwen4-max-preview \
  --model-kind cloud-endpoint --model-provider Qwen
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/completions,\
model=qwen4-max-preview,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks leaderboard_mmlu_pro --batch_size 1 --num_fewshot 5 --log_samples \
  --output_path run-output/cloud-router/qwen4-max-preview/knowledge
mlx-bench-publish run-output/cloud-router/qwen4-max-preview/knowledge/results_*.json \
  --kind lm-eval --suite knowledge --model qwen4-max-preview \
  --model-kind cloud-endpoint --model-provider Qwen
```

For `qwen3.8-max`:

```sh
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-chat-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/chat/completions,\
model=qwen3.8-max,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks gsm8k --batch_size 1 --num_fewshot 5 --limit 100 \
  --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs max_gen_toks=1024 --log_samples \
  --output_path run-output/cloud-router/qwen3.8-max/reasoning
mlx-bench-publish run-output/cloud-router/qwen3.8-max/reasoning/results_*.json \
  --kind lm-eval --suite reasoning --model qwen3.8-max \
  --model-kind cloud-endpoint --model-provider Qwen
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-chat-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/chat/completions,\
model=qwen3.8-max,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks minerva_math500 --batch_size 1 --num_fewshot 0 \
  --apply_chat_template --fewshot_as_multiturn --log_samples \
  --output_path run-output/cloud-router/qwen3.8-max/math-hard
mlx-bench-publish run-output/cloud-router/qwen3.8-max/math-hard/results_*.json \
  --kind lm-eval --suite math-hard --model qwen3.8-max \
  --model-kind cloud-endpoint --model-provider Qwen
OPENAI_API_KEY="$ROUTER_API_KEY" lm_eval --model local-completions \
  --model_args "base_url=${ROUTER_BASE_URL}/completions,\
model=qwen3.8-max,tokenizer=${ROUTER_TOKENIZER},num_concurrent=1,\
max_length=32768,timeout=3600" \
  --tasks leaderboard_mmlu_pro --batch_size 1 --num_fewshot 5 --log_samples \
  --output_path run-output/cloud-router/qwen3.8-max/knowledge
mlx-bench-publish run-output/cloud-router/qwen3.8-max/knowledge/results_*.json \
  --kind lm-eval --suite knowledge --model qwen3.8-max \
  --model-kind cloud-endpoint --model-provider Qwen
```

## IFBench

[`configs/evalscope/ifbench-openai.toml`](../configs/evalscope/ifbench-openai.toml)
uses EvalScope's [OpenAI API runner](https://github.com/modelscope/evalscope/blob/main/docs/en/benchmarks/ifbench.md).
Each run writes `reports/<model>/ifbench.json` beneath its work directory.

```sh
uvx --from evalscope==1.13.0 evalscope eval \
  --model qwen4-max-preview --api-url "$ROUTER_BASE_URL" \
  --api-key "$ROUTER_API_KEY" --eval-type openai_api --datasets ifbench \
  --seed 1 --eval-batch-size 1 \
  --work-dir run-output/cloud-router/qwen4-max-preview/ifbench --no-timestamp
mlx-bench-publish run-output/cloud-router/qwen4-max-preview/ifbench/reports/qwen4-max-preview/ifbench.json \
  --kind evalscope --suite framework-eval --model qwen4-max-preview \
  --model-kind cloud-endpoint --model-provider Qwen
uvx --from evalscope==1.13.0 evalscope eval \
  --model qwen3.8-max --api-url "$ROUTER_BASE_URL" \
  --api-key "$ROUTER_API_KEY" --eval-type openai_api --datasets ifbench \
  --seed 1 --eval-batch-size 1 \
  --work-dir run-output/cloud-router/qwen3.8-max/ifbench --no-timestamp
mlx-bench-publish run-output/cloud-router/qwen3.8-max/ifbench/reports/qwen3.8-max/ifbench.json \
  --kind evalscope --suite framework-eval --model qwen3.8-max \
  --model-kind cloud-endpoint --model-provider Qwen
```
