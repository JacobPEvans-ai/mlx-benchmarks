# Benchmark campaign target inputs

The parameterized campaign TOMLs are read by a separate Ansible consumer that
uses `community.general.from_toml` and argv-based command dispatch. This
repository has no TOML-to-argv runner. The consumer supplies its checkout path;
`config_name` selects one allow-listed recipe.

The consumer validates these survey values at runtime:

- `machine`: inventory alias for the selected target.
- `artifact_id`: exact artifact key in the Ansible model registry.
- `engine`, `model_size`, `concurrency_list`, and `context_list`: recipe axes
  resolved against the selected registry artifact and inventory.
- `benchmark_endpoint_root`: HTTPS endpoint origin for the target.
- `benchmark_cache_path`: writable cache path on that target.
- `power_cap_w`: inventory-derived cap checked against target telemetry.

The consumer gathers target system facts from inventory and records them in the
result envelope. For MLX, it accepts a zero power cap when no power limit
applies and requires the registered artifact to exist under the supplied cache
path. Recipes contain no target identity, endpoint value, model ID, artifact
location, or numeric cap. The `power_cap_w = "inventory"` recipe value is a
validation instruction; the consumer records an inventory-derived numeric cap
when present and omits the envelope field when the target has no cap.

Before dispatch, the consumer checks each prompt-plus-output reservation
against the authoritative artifact and engine context limit, and each
concurrency against the selected serving profile cap. It fails closed if a
limit is unavailable or exceeded. For throughput-probe rows, it resolves
`context_window_tokens` from the selected model registry record and passes it
as `--window-limit-tokens`; an absent registry value fails before dispatch.
When conversion runs on the controller, it sets `MLX_BENCH_SYSTEM_OS`,
`MLX_BENCH_SYSTEM_CHIP`, and `MLX_BENCH_SYSTEM_MEMORY_GB` from the selected
target's inventory facts so the envelope describes the benchmark target.

## Recipes

- `llama-cpp/throughput` (`llama_cpp`): OpenAI-compatible serving throughput by
  prompt length and concurrency.
- `llama-cpp/vllm-fixed-length` (`llama_cpp`): fixed-output OpenAI chat cells
  against llama.cpp using `vllm bench serve --ignore-eos`.
- `strata/throughput` (`strata`): OpenAI-compatible serving throughput by
  prompt length and concurrency.
- `llama-cpp/cross-card` (`llama_cpp`): llama.cpp native diagnostics and serving
  throughput.
- `vllm/cross-card` (`vllm`): `vllm bench serve` throughput at two output sizes.
- `vllm/agentic-throughput` (`vllm`): shared-prefix agent load with the prefix cache on, plus a generation
  ceiling and a cold-prompt ceiling; rows vary concurrency only.
- `mlx/cross-card` (`mlx_lm`): `mlx_lm.benchmark` diagnostics and serving
  throughput through a 262144-token window.
- `mlx/vlm-native` (`mlx_vlm_native`): Native mlx-vlm serving throughput with
  MTP disabled.
- `mlx/vlm-native-mtp` (`mlx_vlm_native`): Native mlx-vlm serving throughput with
  MTP enabled.
- `mlx/vllm-mlx` (`vllm_mlx`): vllm-mlx serving throughput with continuous
  batching.
- `lm-eval/quick-intelligence` (`llama_cpp`, `vllm`, `mlx_lm`): MMLU-Pro
  quick screen (20 items); x2.
- `lm-eval/gpqa-diamond` (`llama_cpp`, `vllm`, `mlx_lm`): GPQA-Diamond
  chain-of-thought, zero-shot.
- `lm-eval/ruler` (`llama_cpp`, `vllm`, `mlx_lm`): RULER
  64k/128k/256k/model-card max.
- `lm-eval/coding` (`llama_cpp`, `vllm`, `mlx_lm`): MBPP code generation.
- `lm-eval/reasoning` (`llama_cpp`, `vllm`, `mlx_lm`): GSM8K reasoning.
- `evalscope/livecodebench` (`llama_cpp`, `vllm`): LiveCodeBench v6 through
  EvalScope's Docker sandbox.
- `evalscope/mmlu-pro-reasoning-effort` (`llama_cpp`): seeded MMLU-Pro generative
  samples with OFF, LOW, and XHIGH reasoning arms at concurrency one.

Recipes declare concurrency and context axes in TOML. Most recipes use concurrency
values 1, 2, 4, and 8; Strata also tests 6. MLX serving recipes include a
262016-token prompt with 128 output tokens to reach a 262144-token window. The
RULER recipe sweeps 64k, 128k, 256k, and the selected model-card maximum; other
context ranges are listed in each recipe. The target, model class (`small`,
`medium`, or `large`), and the engine are survey values resolved against the
inventory and the model registry, so one recipe covers every class on every target
that serves one of its engines.

For Strata, each row fixes a prompt/output pair: 25/512, 1024/2048, or 4096/1024
tokens. The recipe sweeps concurrency at 1, 2, 4, 6, and 8. Configure batch slots
to 1, 4, or 8 on the target before dispatching each campaign; the recipe does not
change the running engine configuration between cells.

Short intelligence screens use two repetitions to check score stability. Choose
the remaining suites for measurable headroom at current scores, and weigh their
results equally with throughput when selecting a model.

`lm-eval/gpqa-diamond` reads a gated dataset: the benchmark target needs a Hugging
Face read credential in its own environment. `evalscope/livecodebench` needs
`evalscope[sandbox]` and a running Docker engine on the benchmark target; the recipe
declares no converter, so its raw reports stay in the target results directory.
`tests/test_campaign_recipes.py` pins this list, its axes, and the placeholders
the consumer substitutes.
