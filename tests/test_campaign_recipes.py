"""Campaign recipes keep the contract the external Ansible consumer relies on.

Nothing in-repo reads these TOML files at run time, so without this test a renamed
converter kind, a placeholder the consumer does not substitute, or a recipe that lost
an axis would fail only when a campaign is dispatched on a target.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.envelope import load_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
TARGET_INPUTS_DOC = REPO_ROOT / "docs" / "benchmark-campaign-target-inputs.md"

EXPECTED = {
    "llama-cpp/throughput",
    "llama-cpp/vllm-fixed-length",
    "strata/throughput",
    "llama-cpp/cross-card",
    "vllm/cross-card",
    "mlx/cross-card",
    "mlx/vlm-native",
    "mlx/vlm-native-mtp",
    "mlx/vllm-mlx",
    "lm-eval/quick-intelligence",
    "lm-eval/gpqa-diamond",
    "lm-eval/ruler",
    "lm-eval/coding",
    "lm-eval/reasoning",
    "evalscope/livecodebench",
    "evalscope/mmlu-pro-reasoning-effort",
}
LONG_CONTEXT_RECIPES = EXPECTED - {
    "lm-eval/coding",
    "lm-eval/reasoning",
    "evalscope/mmlu-pro-reasoning-effort",
}
ENGINES = {"llama_cpp", "vllm", "mlx_lm", "mlx_vlm_native", "vllm_mlx", "strata"}
AGENT_COUNTS = {1, 2, 4, 8}
LONG_CONTEXT_TOKENS = 131072
CONFIG_NAME = re.compile(r"^(llama-cpp|strata|vllm|mlx|lm-eval|evalscope)/[a-z0-9-]+$")
# The only tokens the consumer's argv template substitutes.
PLACEHOLDERS = {
    "artifact_path",
    "benchmark_endpoint_root",
    "benchmark_endpoint",
    "model_id",
    "serving_model",
    "context_window_tokens",
    "context",
    "concurrency",
    "repetition",
    "output_tokens",
    "sample_slice",
    "result_dir",
    "result_filename",
    "result_path",
    "output_path",
}
MODEL_PLACEHOLDERS = ("{{ model_id }}", "{{ artifact_path }}")
ENDPOINT_LITERAL = re.compile(r"https?://|\b\d{1,3}(?:\.\d{1,3}){3}\b")


def _load_recipes() -> dict[str, dict[str, Any]]:
    """Every structured campaign recipe, keyed by its path below ``configs/``."""
    recipes: dict[str, dict[str, Any]] = {}
    for path in sorted(CONFIGS.glob("*/*.toml")):
        data = tomllib.loads(path.read_text())
        if "config_name" in data:
            recipes[path.relative_to(CONFIGS).with_suffix("").as_posix()] = data
    return recipes


RECIPES = _load_recipes()
RUNS = [(name, run) for name, recipe in RECIPES.items() for run in recipe["run"]]
RUN_IDS = [f"{name}:{run['name']}" for name, run in RUNS]
CONVERTED_RUNS = [(name, run) for name, run in RUNS if "converter" in run]
CONVERTED_RUN_IDS = [f"{name}:{run['name']}" for name, run in CONVERTED_RUNS]


def test_declared_recipes_are_the_expected_set() -> None:
    assert set(RECIPES) == EXPECTED


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_config_name_matches_its_path_and_the_consumer_pattern(name: str) -> None:
    assert RECIPES[name]["config_name"] == name
    assert CONFIG_NAME.match(name)


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_recipe_declares_the_agent_and_long_context_axes(name: str) -> None:
    recipe = RECIPES[name]
    assert recipe["power_cap_w"] == "inventory"
    assert recipe["engines"]
    assert set(recipe["engines"]) <= ENGINES
    for axis in ("concurrency_list", "context_list"):
        values = recipe[axis]
        assert all(
            (isinstance(value, int) and value > 0)
            or (name == "lm-eval/ruler" and axis == "context_list" and value == "model_card_max")
            for value in values
        )
        assert len(set(values)) == len(values)
    if name != "evalscope/mmlu-pro-reasoning-effort":
        assert set(recipe["concurrency_list"]) >= AGENT_COUNTS
    if name == "strata/throughput":
        assert set(recipe["context_list"]) == {25, 1024, 4096}
    elif name in LONG_CONTEXT_RECIPES:
        assert max(value for value in recipe["context_list"] if isinstance(value, int)) >= LONG_CONTEXT_TOKENS


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_run_names_are_unique_within_a_recipe(name: str) -> None:
    names = [run["name"] for run in RECIPES[name]["run"]]
    assert names
    assert len(set(names)) == len(names)


@pytest.mark.parametrize(("name", "run"), RUNS, ids=RUN_IDS)
def test_run_declares_the_dispatch_fields(name: str, run: dict[str, Any]) -> None:
    assert run["artifact_id"] == "selected_model"
    assert run["run_on"] in {"benchmark_target", "controller"}
    assert set(run["dimensions"]) <= {"context_list", "concurrency_list", "sample_slice_list"}
    assert isinstance(run["repetitions"], int) and run["repetitions"] >= 1
    assert isinstance(run["output_tokens"], int) and run["output_tokens"] > 0
    assert isinstance(run["executable"], str) and run["executable"]
    assert run["argv"] and all(isinstance(argument, str) for argument in run["argv"])


@pytest.mark.parametrize(("name", "run"), RUNS, ids=RUN_IDS)
def test_run_uses_only_placeholders_the_consumer_substitutes(name: str, run: dict[str, Any]) -> None:
    for argument in run["argv"]:
        for token in re.findall(r"\{\{.*?\}\}", argument):
            match = re.fullmatch(r"\{\{ ([a-z_]+) \}\}", token)
            assert match, f"{token!r} is not a single-space placeholder in {argument!r}"
            assert match.group(1) in PLACEHOLDERS, f"{token!r} is not substituted by the consumer"


@pytest.mark.parametrize(("name", "run"), RUNS, ids=RUN_IDS)
def test_run_takes_model_and_endpoint_from_the_survey(name: str, run: dict[str, Any]) -> None:
    """Identity comes from the registry placeholders, never from the recipe."""
    assert any(token in argument for argument in run["argv"] for token in MODEL_PLACEHOLDERS)
    assert not [argument for argument in run["argv"] if ENDPOINT_LITERAL.search(argument)]


@pytest.mark.parametrize(("name", "run"), RUNS, ids=RUN_IDS)
def test_axis_placeholders_are_declared_dimensions(name: str, run: dict[str, Any]) -> None:
    """A placeholder whose axis is not a dimension would render the unselected value."""
    argv = "\n".join(run["argv"])
    if "{{ context }}" in argv:
        assert "context_list" in run["dimensions"] or run.get("prompt_tokens", 0) > 0
    if "{{ concurrency }}" in argv:
        assert "concurrency_list" in run["dimensions"]
    if "{{ sample_slice }}" in argv:
        assert "sample_slice_list" in run["dimensions"]


def test_quick_intelligence_declares_one_sample_list_and_a_slice_dimension() -> None:
    recipe = RECIPES["lm-eval/quick-intelligence"]
    (run,) = recipe["run"]
    config_text = (CONFIGS / "lm-eval" / "quick-intelligence.toml").read_text()

    assert run["sample_task"] == _argument_after(run, "--tasks")
    assert len(run["sample_indices"]) == 420
    assert len(set(run["sample_indices"])) == 420
    assert run["slice_size"] == 30
    assert "sample_slice_list" in run["dimensions"]
    assert _argument_after(run, "--samples") == "{{ sample_slice }}"
    assert config_text.count("sample_indices = [") == 1


@pytest.mark.parametrize(("name", "run"), CONVERTED_RUNS, ids=CONVERTED_RUN_IDS)
def test_converter_names_a_registered_kind_and_schema_suite(name: str, run: dict[str, Any]) -> None:
    converter = run["converter"]
    assert converter["executable"] == "mlx-bench-publish"
    assert converter["run_on"] == "controller"
    get_converter(converter["kind"])  # raises ValueError for an unregistered kind
    assert converter["suite"] in set(load_schema()["properties"]["suite"]["enum"])


def _argument_after(run: dict[str, Any], flag: str) -> str:
    argv = run["argv"]
    return argv[argv.index(flag) + 1]


def test_gpqa_diamond_runs_its_task_through_the_lm_eval_converter() -> None:
    (run,) = RECIPES["lm-eval/gpqa-diamond"]["run"]
    assert run["executable"] == "lm_eval"
    assert _argument_after(run, "--tasks") == "gpqa_diamond_cot_zeroshot"
    assert _argument_after(run, "--num_fewshot") == "0"
    assert "--apply_chat_template" in run["argv"]
    assert run["converter"]["kind"] == "lm-eval"


def test_ruler_sweeps_standard_and_model_card_context_lengths() -> None:
    recipe = RECIPES["lm-eval/ruler"]
    (run,) = recipe["run"]
    argv = "\n".join(run["argv"])

    assert recipe["context_list"] == [65536, 131072, 262144, "model_card_max"]
    assert _argument_after(run, "--tasks") == "ruler"
    assert "max_seq_lengths" in argv and "{{ context }}" in argv
    assert "model={{ serving_model }}" in argv
    assert "tokenizer={{ model_id }}" in argv
    assert run["api_key_env"] == "OPENAI_API_KEY"
    assert run["run_on"] == "controller"
    assert run["converter"]["suite"] == "reasoning"


def test_coding_and_reasoning_use_runtime_chat_completions_recipes() -> None:
    (coding,) = RECIPES["lm-eval/coding"]["run"]
    (reasoning,) = RECIPES["lm-eval/reasoning"]["run"]

    assert _argument_after(coding, "--tasks") == "mbpp"
    assert "--confirm_run_unsafe_code" in coding["argv"]
    assert coding["converter"]["suite"] == "coding"
    assert _argument_after(reasoning, "--tasks") == "gsm8k"
    assert reasoning["converter"]["suite"] == "reasoning"
    for run in (coding, reasoning):
        assert run["run_on"] == "controller"
        assert run["api_key_env"] == "OPENAI_API_KEY"
        assert any("model={{ serving_model }}" in arg for arg in run["argv"])


def test_livecodebench_runs_through_evalscope_in_the_sandbox() -> None:
    (run,) = RECIPES["evalscope/livecodebench"]["run"]
    assert run["executable"] == "evalscope"
    assert _argument_after(run, "--datasets") == "live_code_bench"
    assert _argument_after(run, "--eval-type") == "openai_api"
    assert '"v6"' in _argument_after(run, "--dataset-args")
    assert '"enabled": true' in _argument_after(run, "--sandbox")
    assert _argument_after(run, "--eval-batch-size") == "{{ concurrency }}"


def test_mmlu_pro_reasoning_effort_recipe_runs_fixed_seeded_generative_arms() -> None:
    recipe = RECIPES["evalscope/mmlu-pro-reasoning-effort"]
    assert recipe["engines"] == ["llama_cpp"]
    assert recipe["context_list"] == [32768]
    assert recipe["concurrency_list"] == [1]
    assert {run["name"] for run in recipe["run"]} == {
        "mmlu-pro-off",
        "mmlu-pro-low",
        "mmlu-pro-xhigh",
    }
    for run in recipe["run"]:
        assert run["executable"] == "evalscope"
        assert run["output_tokens"] == 16384
        assert run["repetitions"] == 1
        assert "--collect-perf" in run["argv"]
        assert _argument_after(run, "--datasets") == "mmlu_pro"
        assert _argument_after(run, "--seed") == "1"
        assert _argument_after(run, "--limit") == "4"
        assert _argument_after(run, "--eval-batch-size") == "1"
        assert "--judge" not in run["argv"]
    by_name = {run["name"]: _argument_after(run, "--generation-config") for run in recipe["run"]}
    assert '"enable_thinking": false' in by_name["mmlu-pro-off"]
    assert '"reasoning_effort": "low"' in by_name["mmlu-pro-low"]
    assert '"reasoning_effort": "xhigh"' in by_name["mmlu-pro-xhigh"]


def test_llama_cpp_throughput_recipe_matches_the_headline_cell_shape() -> None:
    (run,) = RECIPES["llama-cpp/throughput"]["run"]
    assert run["output_tokens"] == 512
    assert run["repetitions"] == 1
    assert run["dimensions"] == ["context_list", "concurrency_list"]
    assert {25, 65536} <= set(RECIPES["llama-cpp/throughput"]["context_list"])
    assert _argument_after(run, "--target-prompt-tokens") == "{{ context }}"
    assert _argument_after(run, "--max-tokens") == "{{ output_tokens }}"
    assert _argument_after(run, "--repeats") == "4"


def test_llama_cpp_fixed_length_recipe_uses_the_native_vllm_chat_client() -> None:
    recipe = RECIPES["llama-cpp/vllm-fixed-length"]
    reference = RECIPES["vllm/cross-card"]

    assert recipe["context_list"] == reference["context_list"]
    assert recipe["concurrency_list"] == reference["concurrency_list"]
    assert {run["output_tokens"] for run in recipe["run"]} == {256, 2048}
    assert all(run["executable"] == "vllm" for run in recipe["run"])
    for run in recipe["run"]:
        assert _argument_after(run, "--backend") == "openai-chat"
        assert "--ignore-eos" in run["argv"]
        assert run["converter"]["kind"] == "vllm"


def test_strata_throughput_recipe_matches_the_batch_campaign_shape() -> None:
    recipe = RECIPES["strata/throughput"]
    assert set(recipe["concurrency_list"]) == {1, 2, 4, 6, 8}
    assert recipe["context_list"] == [25, 1024, 4096]
    assert [(run["prompt_tokens"], run["output_tokens"]) for run in recipe["run"]] == [
        (25, 512),
        (1024, 2048),
        (4096, 1024),
    ]
    for run in recipe["run"]:
        assert run["dimensions"] == ["concurrency_list"]
        assert _argument_after(run, "--base-url") == "{{ benchmark_endpoint }}"
        assert _argument_after(run, "--target-prompt-tokens") == "{{ context }}"
        assert _argument_after(run, "--max-tokens") == "{{ output_tokens }}"
        assert run["converter"]["kind"] == "throughput-probe"


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_recipe_is_listed_in_the_target_inputs_doc(name: str) -> None:
    assert f"`{name}`" in TARGET_INPUTS_DOC.read_text()


def test_mlx_recipe_reaches_a_full_256k_window_with_128_output_tokens() -> None:
    recipe = RECIPES["mlx/cross-card"]
    assert 262016 in recipe["context_list"]
    assert all(run["output_tokens"] == 128 for run in recipe["run"])


def test_native_mlx_vlm_recipes_separate_mtp_modes() -> None:
    off = RECIPES["mlx/vlm-native"]
    on = RECIPES["mlx/vlm-native-mtp"]
    assert off["engines"] == on["engines"] == ["mlx_vlm_native"]
    assert off["context_list"] == on["context_list"]
    assert off["concurrency_list"] == on["concurrency_list"]
    assert off["description"] == "MTP disabled"
    assert on["description"] == "MTP enabled"


def test_vllm_mlx_recipe_declares_its_continuous_batching_engine() -> None:
    recipe = RECIPES["mlx/vllm-mlx"]
    assert recipe["engines"] == ["vllm_mlx"]
    assert 262016 in recipe["context_list"]
