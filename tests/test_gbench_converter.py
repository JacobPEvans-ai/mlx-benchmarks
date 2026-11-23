"""Tests for the gbench converter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mlx_benchmarks.cli import main
from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.publish import envelope_to_rows

FIXTURE = Path(__file__).parent / "fixtures" / "gbench_summary_sample.json"


def _ctx(**overrides: Any) -> ConverterContext:
    kwargs: dict[str, Any] = {
        "git_sha": "abc1234",
        "trigger": "local",
        "suite": "reasoning",
        "model": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "system": {"os": "Linux", "chip": "x86_64", "memory_gb": 128},
        "timestamp_override": "2026-11-23T14:05:00Z",
    }
    kwargs.update(overrides)
    return ConverterContext(**kwargs)


def test_summary_envelope_roundtrip() -> None:
    raw = json.loads(FIXTURE.read_text())
    envelope = get_converter("gbench").build_envelope(raw, _ctx(extra_tags={"env": "lab"}))
    validate_envelope(envelope)
    assert envelope["timestamp"] == "2026-11-23T14:05:00Z"

    by_key = {(r["name"], r["metric"]): r for r in envelope["results"]}
    assert by_key[("golden", "pass_rate")]["value"] == pytest.approx(0.9167)
    assert by_key[("golden/tool_calling", "pass_rate")]["value"] == pytest.approx(0.5)
    assert by_key[("golden/structured_output", "pass_rate")]["value"] == pytest.approx(1.0)

    ifeval = by_key[("ifeval", "accuracy")]
    assert ifeval["value"] == pytest.approx(0.8595)
    assert ifeval["unit"] == "ratio"
    assert ifeval["reasoning_arm"] == "thinking_on"
    assert ifeval["truncation_rate"] == pytest.approx(4 / 541, rel=1e-4)
    assert ifeval["tags"]["env"] == "lab"
    assert by_key[("ifeval/prompt_strict", "accuracy")]["value"] == pytest.approx(0.8595)

    assert by_key[("throughput", "output_tokens_per_second")]["value"] == pytest.approx(72.4)
    assert by_key[("ttft_p50", "latency_ms")]["value"] == pytest.approx(115.2)
    assert by_key[("stress_test", "max_sustainable_qps")]["value"] == pytest.approx(3.5)
    assert by_key[("stress_test", "max_sustainable_qps")]["tags"]["certified"] == "true"

    assert len(envelope["errors"]) == 1
    assert "bfcl" in envelope["errors"][0]

    rows = envelope_to_rows(envelope)
    assert len(rows) == len(envelope["results"])


def test_standalone_eval_and_stress_inputs() -> None:
    conv = get_converter("gbench")
    eval_raw = {
        "eval_name": "bigcodebench",
        "accuracy": 58.25,
        "thinking": False,
        "total_questions": 200,
    }
    env = conv.build_envelope(eval_raw, _ctx(suite="coding"))
    validate_envelope(env)
    assert env["results"][0]["value"] == pytest.approx(0.5825)
    assert env["results"][0]["reasoning_arm"] == "thinking_off"

    list_raw = [
        {
            "benchmark_type": "stress_test_multimodal",
            "max_sustainable_qps": 1.25,
            "multimodal": True,
            "no_rate_reason": "capped by image encoder warmup",
        }
    ]
    env_mm = conv.build_envelope(list_raw, _ctx(suite="throughput-serving"))
    validate_envelope(env_mm)
    assert env_mm["results"][0]["name"] == "stress_test_multimodal/stress_test"
    assert env_mm["results"][0]["value"] == pytest.approx(1.25)
    assert "capped by image encoder warmup" in env_mm["errors"][0]


def test_quality_and_empty_error_cases() -> None:
    conv = get_converter("gbench")
    quality_env = conv.build_envelope(
        {
            "benchmark_type": "quality",
            "passed_scenarios": 8,
            "total_scenarios": 10,
            "pass_rate": 80.0,
        },
        _ctx(suite="tool-calling"),
    )
    validate_envelope(quality_env)
    assert quality_env["results"][0]["value"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="no benchmark results"):
        conv.build_envelope({"models": [], "failures": []}, _ctx())

    with pytest.raises(ValueError, match="HTTP 500"):
        conv.build_envelope(
            {"models": [], "failures": [{"benchmark_type": "eval", "eval_name": "lcb", "error": "HTTP 500"}]},
            _ctx(),
        )


def test_cli_dry_run_gbench(
    tmp_path: Path,
    published_metadata: dict[str, Any],
    mock_hf_registry: None,
) -> None:
    sample = json.loads(FIXTURE.read_text())
    sample["model_revision"] = "a" * 40
    results_path = tmp_path / "summary.json"
    results_path.write_text(json.dumps(sample))

    meta_path = tmp_path / "published_metadata.json"
    meta_path.write_text(json.dumps(published_metadata))

    exit_code = main(
        [
            str(results_path),
            "--kind",
            "gbench",
            "--suite",
            "reasoning",
            "--git-sha",
            "deadbeef",
            "--timestamp",
            str(published_metadata["start_utc"]),
            "--published-metadata",
            str(meta_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
