"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def lm_eval_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "lm_eval_results_sample.json").read_text())


@pytest.fixture
def vllm_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "vllm_benchmark_serving_sample.json").read_text())


@pytest.fixture
def agentic_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "agentic_sample.json").read_text())


@pytest.fixture
def promptstack_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "promptstack_sample.json").read_text())


@pytest.fixture
def promptfoo_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "promptfoo_sample.json").read_text())


@pytest.fixture
def evalscope_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "evalscope_report_sample.json").read_text())


@pytest.fixture
def factual_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "factual_sample.json").read_text())


@pytest.fixture
def coding_replay_sample() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "coding_replay_sample.json").read_text())


@pytest.fixture
def tau2_results_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "tau2_results_sample.json").read_text())


@pytest.fixture
def vllm_bench_serve_nvidia_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "vllm_bench_serve_nvidia_synthetic.json").read_text())


@pytest.fixture
def fio_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "fio_randread_synthetic.json").read_text())


@pytest.fixture
def nvbandwidth_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "nvbandwidth_synthetic.json").read_text())


@pytest.fixture
def mbw_sample() -> dict[str, Any]:
    # Same shape the CLI builds from a .txt file: {"output": <raw text>}.
    return {"output": (FIXTURES / "mbw_synthetic.txt").read_text()}


@pytest.fixture
def gpu_burn_sample() -> dict[str, Any]:
    # Stored as JSON because the real output is CR-redrawn and a text fixture would
    # lose its carriage returns to the repo's line-ending hook.
    return json.loads((FIXTURES / "gpu_burn_synthetic.json").read_text())


@pytest.fixture
def valid_envelope() -> dict[str, Any]:
    envelope = json.loads((EXAMPLES / "envelope.valid.json").read_text())
    envelope["model_revision"] = "a" * 40
    return envelope


@pytest.fixture
def cluster_envelope() -> dict[str, Any]:
    return json.loads((EXAMPLES / "envelope.cluster.json").read_text())


@pytest.fixture
def nvidia_envelope() -> dict[str, Any]:
    envelope = json.loads((EXAMPLES / "envelope.nvidia.json").read_text())
    envelope["model_revision"] = "a" * 40
    return envelope


@pytest.fixture
def invalid_envelope() -> dict[str, Any]:
    return json.loads((EXAMPLES / "envelope.invalid.json").read_text())


@pytest.fixture
def published_metadata() -> dict[str, Any]:
    return {
        "dataset_id": "owner/benchmark",
        "dataset_task_id": "default",
        "dataset_revision": "b" * 40,
        "evaluation_framework": "inspect-ai",
        "config": "default",
        "split": "test",
        "engine": "MLX",
        "engine_version": "1.0",
        "profile": "test",
        "hardware": {"chip": "Example CPU"},
        "power_limit_w": 100.0,
        "concurrency": 1,
        "ctx_per_slot": 4096,
        "kv_cache_dtype": "float16",
        "thinking": False,
        "temperature": 0.0,
        "max_tokens": 128,
        "prompt_chars": 100,
        "prompt_tokens": 25,
        "system_prompt_chars": 20,
        "system_prompt_tokens": 5,
        "runner": "mlx-benchmarks",
        "harness": "inspect-ai",
        "router_key_alias": "local",
        "run_id": "test-run",
        "start_utc": "2026-04-24T18:30:00Z",
        "end_utc": "2026-04-24T18:32:03Z",
        "dimension_null_reasons": {"quant": "not_applicable"},
    }


@pytest.fixture
def mock_hf_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "name: Example benchmark\n"
        "description: Synthetic test benchmark.\n"
        "evaluation_framework: inspect-ai\n"
        "tasks:\n"
        "  - id: default\n"
        "    config: default\n"
        "    split: test\n"
    )

    def model_info(self: Any, repo_id: str, revision: str | None = None, **kwargs: Any) -> Any:
        return SimpleNamespace(
            id=repo_id,
            sha=revision,
            pipeline_tag="text-generation",
            card_data={
                "pipeline_tag": "text-generation",
                "license": "apache-2.0",
                "base_model": None,
                "base_model_relation": None,
            },
            library_name="transformers",
            safetensors={"total": 1000, "parameters": {"F16": 1000}},
            gguf=None,
            config={
                "architectures": ["ExampleForCausalLM"],
                "model_type": "example",
                "max_position_embeddings": 4096,
            },
            gated=False,
            tags=["text-generation"],
        )

    def dataset_info(self: Any, repo_id: str, revision: str | None = None, **kwargs: Any) -> Any:
        return SimpleNamespace(sha=revision, tags=["benchmark:official", "benchmark:eval-yaml"])

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", model_info)
    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", dataset_info)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", lambda *args, **kwargs: ["eval.yaml"])
    monkeypatch.setattr("huggingface_hub.HfApi.hf_hub_download", lambda *args, **kwargs: str(eval_yaml))
