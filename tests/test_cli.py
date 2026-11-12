"""CLI smoke tests — argparse, Hub metadata lookup, and dry-run dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_benchmarks.cli import build_parser, main
from mlx_benchmarks.envelope import validate_envelope


def _write_sample(tmp_path: Path, sample: dict) -> Path:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(sample))
    return path


def _write_metadata(tmp_path: Path, metadata: dict) -> Path:
    path = tmp_path / "published-metadata.json"
    path.write_text(json.dumps(metadata))
    return path


def test_cli_dry_run_happy_path(
    tmp_path: Path,
    lm_eval_sample: dict,
    published_metadata: dict,
    mock_hf_registry: None,
    capsys: pytest.CaptureFixture,
) -> None:
    lm_eval_sample["model_revision"] = "a" * 40
    results_path = _write_sample(tmp_path, lm_eval_sample)
    exit_code = main(
        [
            str(results_path),
            "--kind",
            "lm-eval",
            "--suite",
            "reasoning",
            "--git-sha",
            "deadbeef",
            "--timestamp",
            "2026-04-24T18:30:00Z",
            "--published-metadata",
            str(_write_metadata(tmp_path, published_metadata)),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.err.lower() or "planned" in captured.err.lower()


def test_cli_keeps_list_input_for_single_cell_converters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_path = tmp_path / "bench-serve.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "model_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
                    "model_revision": "a" * 40,
                    "prompt_set": "short",
                    "concurrency": 1,
                    "repetition": 0,
                    "gen_tps": 100.0,
                    "throughput_tps": 110.0,
                    "ttft_ms": 70.0,
                    "tpot_ms": 7.5,
                }
            ]
        )
    )
    published: list[dict] = []
    monkeypatch.setattr("mlx_benchmarks.cli.detect_system", dict)
    monkeypatch.setattr(
        "mlx_benchmarks.cli.publish",
        lambda envelope, **_: published.append(envelope) or "data/run-test.parquet",
    )

    assert (
        main(
            [
                str(results_path),
                "--kind",
                "bench-serve",
                "--suite",
                "throughput",
                "--git-sha",
                "deadbeef",
            ]
        )
        == 0
    )
    assert len(published) == 1
    assert published[0]["model"] == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert len(published[0]["results"]) == 4


def test_cli_writes_schema_valid_envelope_without_publishing(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_path = _write_sample(tmp_path, lm_eval_sample)
    output_dir = tmp_path / "envelopes"
    monkeypatch.setattr(
        "mlx_benchmarks.cli.detect_system",
        lambda: {"os": None, "chip": None, "memory_gb": None},
    )
    monkeypatch.delenv("MLX_BENCH_METRICS_URL", raising=False)

    def unexpected_publish(*_: object, **__: object) -> str:
        pytest.fail("publish was called with --output-dir")

    monkeypatch.setattr("mlx_benchmarks.cli.publish", unexpected_publish)

    assert (
        main(
            [
                str(results_path),
                "--kind",
                "lm-eval",
                "--suite",
                "reasoning",
                "--git-sha",
                "deadbeef",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    [envelope_path] = output_dir.glob("*.json")
    envelope = json.loads(envelope_path.read_text())
    validate_envelope(envelope)


def test_cli_publishes_by_default(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_path = _write_sample(tmp_path, lm_eval_sample)
    published: list[dict] = []
    monkeypatch.setattr("mlx_benchmarks.cli.detect_system", dict)
    monkeypatch.setattr(
        "mlx_benchmarks.cli.publish",
        lambda envelope, **_: published.append(envelope) or "data/run.json",
    )

    assert main([str(results_path), "--kind", "lm-eval", "--suite", "reasoning"]) == 0
    assert len(published) == 1


def test_cli_rejects_serving_port_flag(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["results.json", "--suite", "reasoning", "--serving-port"])
    assert "unrecognized arguments: --serving-port" in capsys.readouterr().err


def test_cli_vllm_dry_run(
    tmp_path: Path,
    vllm_sample: dict,
    published_metadata: dict,
    mock_hf_registry: None,
    capsys: pytest.CaptureFixture,
) -> None:
    vllm_sample["model_revision"] = "a" * 40
    results_path = _write_sample(tmp_path, vllm_sample)
    exit_code = main(
        [
            str(results_path),
            "--kind",
            "vllm",
            "--suite",
            "throughput",
            "--model",
            "mlx-community/gpt-oss-120b-MXFP4-Q8",
            "--git-sha",
            "deadbeef",
            "--timestamp",
            "2026-04-24T18:30:00Z",
            "--published-metadata",
            str(_write_metadata(tmp_path, published_metadata)),
            "--tag",
            "host=mac-studio",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.err.lower() or "planned" in captured.err.lower()


def test_cli_agentic_dry_run(
    tmp_path: Path,
    agentic_sample: dict,
    published_metadata: dict,
    mock_hf_registry: None,
    capsys: pytest.CaptureFixture,
) -> None:
    agentic_sample["model_revision"] = "a" * 40
    results_path = _write_sample(tmp_path, agentic_sample)
    exit_code = main(
        [
            str(results_path),
            "--kind",
            "agentic",
            "--suite",
            "tool-calling",
            "--git-sha",
            "deadbeef",
            "--timestamp",
            "2026-04-24T18:30:00Z",
            "--published-metadata",
            str(_write_metadata(tmp_path, published_metadata)),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.err.lower() or "planned" in captured.err.lower()


def test_cli_promptstack_dry_run(
    tmp_path: Path,
    promptstack_sample: dict,
    published_metadata: dict,
    mock_hf_registry: None,
    capsys: pytest.CaptureFixture,
) -> None:
    promptstack_sample["model_revision"] = "a" * 40
    results_path = _write_sample(tmp_path, promptstack_sample)
    exit_code = main(
        [
            str(results_path),
            "--kind",
            "promptstack",
            "--suite",
            "promptstack",
            "--git-sha",
            "deadbeef",
            "--timestamp",
            "2026-04-24T18:30:00Z",
            "--published-metadata",
            str(_write_metadata(tmp_path, published_metadata)),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.err.lower() or "planned" in captured.err.lower()


def test_cli_rejects_hostname_override(tmp_path: Path, lm_eval_sample: dict) -> None:
    results_path = _write_sample(tmp_path, lm_eval_sample)
    with pytest.raises(SystemExit):
        main([str(results_path), "--kind", "lm-eval", "--suite", "reasoning", "--hostname", "redacted"])


def _publish_via_cli(tmp_path: Path, sample: dict, monkeypatch: pytest.MonkeyPatch, *extra_argv: str) -> dict:
    results_path = _write_sample(tmp_path, sample)
    captured: dict[str, object] = {}

    def fake_publish(envelope: dict, **_: object) -> str:
        captured["envelope"] = envelope
        return "data/x.parquet"

    monkeypatch.setattr("mlx_benchmarks.cli.publish", fake_publish)
    assert (
        main(
            [
                str(results_path),
                "--kind",
                "lm-eval",
                "--suite",
                "reasoning",
                "--git-sha",
                "deadbeef",
                *extra_argv,
            ]
        )
        == 0
    )
    envelope = captured["envelope"]
    assert isinstance(envelope, dict)
    return envelope


def test_cli_reasoning_effort_reaches_the_envelope(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = _publish_via_cli(
        tmp_path, lm_eval_sample, monkeypatch, "--reasoning-effort", "xhigh", "--dry-run"
    )
    assert envelope["reasoning_effort"] == "xhigh"


def test_cli_reasoning_effort_takes_any_level_verbatim(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `choices=` on the flag: a family that names its levels differently must
    not be silently coerced, and `off` must stay `off`."""
    for level in ("off", "low", "max", "think-harder"):
        envelope = _publish_via_cli(
            tmp_path, lm_eval_sample, monkeypatch, "--reasoning-effort", level, "--dry-run"
        )
        assert envelope["reasoning_effort"] == level


def test_cli_omits_reasoning_effort_when_not_declared(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent, not defaulted. A harness cannot see the serving default, so
    inventing one here would be indistinguishable from having measured it."""
    envelope = _publish_via_cli(tmp_path, lm_eval_sample, monkeypatch, "--dry-run")
    assert "reasoning_effort" not in envelope


def test_cli_cloud_endpoint_identity_reaches_publisher(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = _publish_via_cli(
        tmp_path,
        lm_eval_sample,
        monkeypatch,
        "--model",
        "qwen4-max-preview",
        "--model-kind",
        "cloud-endpoint",
        "--model-provider",
        "Qwen",
    )

    assert envelope["model"] == "qwen4-max-preview"
    assert envelope["model_kind"] == "cloud-endpoint"
    assert envelope["model_provider"] == "Qwen"
    assert "model_revision" not in envelope
    validate_envelope(envelope)


def test_cli_cloud_endpoint_requires_provider(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "results.json",
                "--kind",
                "lm-eval",
                "--suite",
                "reasoning",
                "--model-kind",
                "cloud-endpoint",
            ]
        )
    assert "--model-provider is required" in capsys.readouterr().err


def test_cli_rejects_invalid_tag(tmp_path: Path, lm_eval_sample: dict) -> None:
    results_path = _write_sample(tmp_path, lm_eval_sample)
    with pytest.raises(SystemExit, match="invalid --tag"):
        main(
            [
                str(results_path),
                "--kind",
                "lm-eval",
                "--suite",
                "reasoning",
                "--git-sha",
                "deadbeef",
                "--tag",
                "no-equals-sign",
                "--dry-run",
            ]
        )


def test_extract_model_reads_model_id_from_a_json_lines_run() -> None:
    """A coding-replay row must publish under the served model, not "unknown".

    The extractor reads ``model_id`` from a list-shaped run. The coding-replay
    runner's ``model`` field is the agent-CLI reference and carries a provider
    prefix, so it is not the right key: two arms served behind different
    provider names would publish as different models. A row missing
    ``model_id`` yields "unknown" *silently* — the extractor falls back rather
    than raising, and the documented publish command passes no ``--model``.
    """
    from mlx_benchmarks.cli import _extract_model

    row = {
        "model": "kimi/mlx-community/Kimi-Linear-48B-A3B-Instruct-6bit",
        "model_id": "mlx-community/Kimi-Linear-48B-A3B-Instruct-6bit",
        "task": "repo-1",
    }
    assert _extract_model([row]) == "mlx-community/Kimi-Linear-48B-A3B-Instruct-6bit"

    # Anti-vacuity: the assertion above must be carried by model_id, not by the
    # prefixed `model` field happening to be picked up.
    assert _extract_model([{k: v for k, v in row.items() if k != "model_id"}]) == "unknown"


def test_extract_model_still_reads_model_id_for_other_kinds() -> None:
    """bench-serve records already carry model_id; that path must not change."""
    from mlx_benchmarks.cli import _extract_model

    assert _extract_model([{"model_id": "mlx-community/Qwen3.6-35B-A3B-4bit"}]) == (
        "mlx-community/Qwen3.6-35B-A3B-4bit"
    )


def test_cli_rejects_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{this is not json")
    exit_code = main(
        [
            str(bad),
            "--kind",
            "lm-eval",
            "--suite",
            "reasoning",
            "--git-sha",
            "deadbeef",
            "--dry-run",
        ]
    )
    assert exit_code == 2


_DIMENSIONS_FILE = {
    "campaign_dimensions": {
        "hardware": {"machine": "Example GPU", "power_cap_w": 300.0, "ups_circuit": None},
        "run": {
            "concurrent_agents": 4,
            "prefix_cache": True,
            "kv_offload_backend": "none",
            "kv_offload_cpu_gib": 0.0,
            "kv_offload_disk_gib": 0.0,
            "weight_cpu_offload_gib": 0.0,
            "llamacpp_cache_ram_mib": 8192,
            "gpu_memory_utilization": 0.88,
            "max_num_batched_tokens": 8192,
            "kv_pool_tokens": 123456,
        },
    },
    "dimension_null_reasons": {"hardware.ups_circuit": "NOT_SOURCED"},
}


def _dimensions_argv(tmp_path: Path, lm_eval_sample: dict, dimensions_path: Path) -> list[str]:
    return [
        str(_write_sample(tmp_path, lm_eval_sample)),
        "--kind",
        "lm-eval",
        "--suite",
        "reasoning",
        "--git-sha",
        "deadbeef",
        "--campaign-dimensions",
        str(dimensions_path),
        "--dry-run",
    ]


def test_cli_campaign_dimensions_reach_the_envelope(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    dimensions_path = tmp_path / "dimensions.json"
    dimensions_path.write_text(json.dumps(_DIMENSIONS_FILE))

    envelope = _publish_via_cli(
        tmp_path, lm_eval_sample, monkeypatch, "--campaign-dimensions", str(dimensions_path), "--dry-run"
    )

    assert envelope["campaign_dimensions"] == _DIMENSIONS_FILE["campaign_dimensions"]
    assert envelope["dimension_null_reasons"] == _DIMENSIONS_FILE["dimension_null_reasons"]


def test_cli_campaign_model_task_reaches_the_envelope(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    dimensions_path = tmp_path / "dimensions.json"
    dimensions_path.write_text(
        json.dumps(
            {
                "campaign_dimensions": {"model": {"id": "embeddinggemma-2"}},
                "model_task": "feature-extraction",
                "model_task_source": "model_card",
            }
        )
    )

    envelope = _publish_via_cli(
        tmp_path, lm_eval_sample, monkeypatch, "--campaign-dimensions", str(dimensions_path), "--dry-run"
    )

    assert envelope["model_task"] == "feature-extraction"
    assert envelope["model_task_source"] == "model_card"


def test_cli_omits_campaign_dimensions_when_not_declared(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = _publish_via_cli(tmp_path, lm_eval_sample, monkeypatch, "--dry-run")

    assert "campaign_dimensions" not in envelope
    assert "dimension_null_reasons" not in envelope


def test_cli_campaign_dimensions_without_reasons(
    tmp_path: Path, lm_eval_sample: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    dimensions_path = tmp_path / "dimensions.json"
    dimensions_path.write_text(json.dumps({"campaign_dimensions": {"run": {"concurrent_agents": 2}}}))

    envelope = _publish_via_cli(
        tmp_path, lm_eval_sample, monkeypatch, "--campaign-dimensions", str(dimensions_path), "--dry-run"
    )

    assert envelope["campaign_dimensions"] == {"run": {"concurrent_agents": 2}}
    assert "dimension_null_reasons" not in envelope


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"dimension_null_reasons": {}}),
        json.dumps({"campaign_dimensions": []}),
        json.dumps({"campaign_dimensions": {}, "dimension_null_reasons": []}),
        json.dumps({"campaign_dimensions": {}, "extra": 1}),
        json.dumps({"campaign_dimensions": {}, "model_task": "feature-extraction"}),
        json.dumps(
            {"campaign_dimensions": {}, "model_task": "feature-extraction", "model_task_source": "unknown"}
        ),
    ],
    ids=[
        "malformed",
        "not-an-object",
        "no-dimensions",
        "dimensions-not-object",
        "reasons-not-object",
        "extra-key",
        "task-without-source",
        "invalid-task-source",
    ],
)
def test_cli_rejects_a_malformed_campaign_dimensions_file(
    tmp_path: Path, lm_eval_sample: dict, content: str
) -> None:
    dimensions_path = tmp_path / "dimensions.json"
    dimensions_path.write_text(content)

    assert main(_dimensions_argv(tmp_path, lm_eval_sample, dimensions_path)) == 2


def test_cli_rejects_a_missing_campaign_dimensions_file(tmp_path: Path, lm_eval_sample: dict) -> None:
    assert main(_dimensions_argv(tmp_path, lm_eval_sample, tmp_path / "absent.json")) == 2


def test_cli_validates_campaign_dimension_field_names(tmp_path: Path, lm_eval_sample: dict) -> None:
    dimensions_path = tmp_path / "dimensions.json"
    dimensions_path.write_text(json.dumps({"campaign_dimensions": {"hardware": {"invented_field": 1}}}))

    assert main(_dimensions_argv(tmp_path, lm_eval_sample, dimensions_path)) == 3
