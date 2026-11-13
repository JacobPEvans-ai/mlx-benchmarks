"""End-to-end: GuideLLM benchmarks.json report -> envelope -> schema validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from mlx_benchmarks.cli import main
from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.publish import envelope_to_rows
from mlx_benchmarks.system import detect_system

FIXTURE = Path(__file__).parent / "fixtures/guidellm_benchmarks_report_synthetic.json"


def _ctx(suite: str = "throughput", **overrides: Any) -> ConverterContext:
    kwargs: dict[str, Any] = {
        "suite": suite,
        # Empty on purpose: exercises the served-model fallback from the report.
        "model": "",
        "git_sha": "deadbeef",
        "system": detect_system(),
    }
    kwargs.update(overrides)
    return ConverterContext(**kwargs)


def _metric(envelope: dict, name: str) -> dict:
    return next(row for row in envelope["results"] if row["metric"] == name)


def test_guidellm_round_trip(guidellm_report: dict) -> None:
    envelope = get_converter("guidellm").build_envelope(guidellm_report, _ctx())

    validate_envelope(envelope)

    assert envelope["suite"] == "throughput"
    assert envelope["model"] == "example-org/example-model"  # from the report backend
    assert envelope["timestamp"] == "2025-11-30T10:53:20Z"  # report start_time
    assert envelope["concurrency"] == 4  # strategy.streams, not the --concurrency flag

    values = {row["metric"]: row["value"] for row in envelope["results"]}
    assert values["throughput_output_toks_per_s"] == pytest.approx(198.2)
    assert values["throughput_total_toks_per_s"] == pytest.approx(298.45)
    assert values["throughput_prompt_toks_per_s"] == pytest.approx(100.25)
    assert values["output_toks_per_s_per_request"] == pytest.approx(42.1)
    assert values["ttft_p50_ms"] == pytest.approx(198.0)
    assert values["ttft_p99_ms"] == pytest.approx(512.5)
    # First ANSWER token (non-reasoning), distinct from ttft_*.
    assert values["ttfa_p50_ms"] == pytest.approx(18742.0)
    assert values["ttfa_p99_ms"] == pytest.approx(41305.0)
    assert values["tpot_p50_ms"] == pytest.approx(1.18)
    assert values["request_errors"] == 1.0
    assert values["requests_incomplete"] == 2.0
    assert values["requests_successful"] == 6.0
    assert values["requests_total"] == 9.0

    failures = _metric(envelope, "truncated_or_empty_answers")
    assert failures["value"] == 2.0  # cap-truncated + empty answer
    assert failures["truncation_rate"] == pytest.approx(2 / 6)
    assert failures["tags"]["failed_truncated"] == "2"
    assert failures["tags"]["failed_empty_answers"] == "1"

    # Token-stat fields attach to the headline throughput row.
    headline = _metric(envelope, "throughput_output_toks_per_s")
    assert headline["output_tokens_mean"] == pytest.approx(1982.0)
    assert headline["output_tokens_median"] == pytest.approx(900.0)
    assert headline["output_tokens_p90"] == pytest.approx(8192.0)
    assert headline["reasoning_tokens_mean"] == pytest.approx(1600.0)

    # The task-level metric rows carry the measured window; count rows do not.
    assert _metric(envelope, "throughput_output_toks_per_s")["duration_seconds"] == 60.0
    assert "duration_seconds" not in _metric(envelope, "request_errors")

    for row in envelope["results"]:
        assert row["name"] == "guidellm"
        assert row["tags"]["thinking"] == "on"
        assert row["tags"]["concurrency"] == "4"
        assert row["tags"]["max_tokens"] == "8192"
        assert row["tags"]["served_model"] == "example-org/example-model"

    assert envelope["errors"] == ["Backend reported a deprecated response field; metrics are unaffected."]


def test_guidellm_counts_cap_and_empty_failures_separately(guidellm_report: dict) -> None:
    envelope = get_converter("guidellm").build_envelope(guidellm_report, _ctx())
    tags = {row["metric"]: row["tags"] for row in envelope["results"]}

    assert tags["truncated_or_empty_answers"]["failed_truncated"] == "2"
    assert tags["truncated_or_empty_answers"]["failed_empty_answers"] == "1"
    # The scheduler-cancelled boundary requests are reported, never counted
    # as finish_reason failures.
    assert tags["requests_incomplete"]["note"] == "cancelled at the measurement boundary"


def test_guidellm_campaign_dimensions_for_throughput_suite(guidellm_report: dict) -> None:
    envelope = get_converter("guidellm").build_envelope(guidellm_report, _ctx())
    dimensions = envelope["campaign_dimensions"]

    assert dimensions["run"]["thinking"] is True
    assert dimensions["run"]["closed_loop"] is True
    assert dimensions["run"]["concurrent_agents"] == 4
    assert dimensions["run"]["prompt_tokens"] == 1002
    assert dimensions["run"]["output_tokens"] == 8192
    assert dimensions["run"]["measured_output_tokens"] == 1982
    assert dimensions["speed"]["ttft_p50_ms"] == pytest.approx(198.0)
    assert dimensions["speed"]["aggregate_output_tokens_per_second"] == pytest.approx(198.2)
    assert dimensions["speed"]["decode_tokens_per_second_per_agent"] == pytest.approx(42.1)

    non_throughput = get_converter("guidellm").build_envelope(guidellm_report, _ctx(suite="ttft"))
    assert "campaign_dimensions" not in non_throughput


def test_guidellm_per_request_cap_overrides_backend_config(guidellm_report: dict) -> None:
    raw = copy.deepcopy(guidellm_report)
    benchmark = raw["benchmarks"][0]
    # Backend cap removed; the empty-answer request carries its own cap of 100
    # in request_args and must still be counted as truncated.
    del benchmark["config"]["backend"]["extras"]["body"]["max_tokens"]
    del benchmark["config"]["backend"]["extras"]["body"]["ignore_eos"]

    envelope = get_converter("guidellm").build_envelope(raw, _ctx())
    failures = _metric(envelope, "truncated_or_empty_answers")
    # The 8192-token response is now unclassifiable; the 100-token empty one
    # is still truncated via its request_args cap.
    assert failures["value"] == 1.0
    assert any("length-truncation detection unavailable" in error for error in envelope["errors"])


def test_guidellm_without_any_cap_still_counts_empty_answers(guidellm_report: dict) -> None:
    raw = copy.deepcopy(guidellm_report)
    benchmark = raw["benchmarks"][0]
    del benchmark["config"]["backend"]["extras"]
    for request in benchmark["requests"]["successful"]:
        request["request_args"] = None

    envelope = get_converter("guidellm").build_envelope(raw, _ctx())
    failures = _metric(envelope, "truncated_or_empty_answers")
    assert failures["value"] == 1.0  # only the empty answer
    assert "truncation_rate" not in failures or failures["truncation_rate"] == pytest.approx(1 / 6)


def test_guidellm_model_and_timestamp_overrides(guidellm_report: dict) -> None:
    envelope = get_converter("guidellm").build_envelope(
        guidellm_report,
        _ctx(model="mlx-community/example-4bit", timestamp_override="2026-01-01T00:00:00Z"),
    )
    assert envelope["model"] == "mlx-community/example-4bit"
    assert envelope["timestamp"] == "2026-01-01T00:00:00Z"


def test_guidellm_missing_model_fails_closed(guidellm_report: dict) -> None:
    raw = copy.deepcopy(guidellm_report)
    del raw["benchmarks"][0]["config"]["backend"]["model"]

    with pytest.raises(ValueError, match="no model id"):
        get_converter("guidellm").build_envelope(raw, _ctx())


def test_guidellm_rejects_multi_benchmark_report(guidellm_report: dict) -> None:
    raw = copy.deepcopy(guidellm_report)
    raw["benchmarks"].append(copy.deepcopy(raw["benchmarks"][0]))

    with pytest.raises(ValueError, match="holds 2 benchmarks"):
        get_converter("guidellm").build_envelope(raw, _ctx())


def test_guidellm_rejects_non_report_and_missing_streams(guidellm_report: dict) -> None:
    with pytest.raises(ValueError, match=r"not a GuideLLM benchmarks\.json report"):
        get_converter("guidellm").build_envelope({"benchmarks": []}, _ctx())

    raw = copy.deepcopy(guidellm_report)
    del raw["benchmarks"][0]["config"]["strategy"]["streams"]
    with pytest.raises(ValueError, match="streams"):
        get_converter("guidellm").build_envelope(raw, _ctx())


def test_guidellm_rows_round_trip_to_parquet(guidellm_report: dict) -> None:
    envelope = get_converter("guidellm").build_envelope(guidellm_report, _ctx())
    validate_envelope(envelope)
    rows = envelope_to_rows(envelope)

    assert rows
    assert all(row["tag_thinking"] == "on" for row in rows)
    assert all(row["tag_concurrency"] == "4" for row in rows)
    failures = next(row for row in rows if row["metric"] == "truncated_or_empty_answers")
    failure_tags = json.loads(failures["tags_json"])
    assert failure_tags["failed_truncated"] == "2"
    assert failure_tags["failed_empty_answers"] == "1"
    assert failures["truncation_rate"] == pytest.approx(2 / 6)
    assert rows[0]["campaign_run_thinking"] is True
    assert rows[0]["campaign_speed_aggregate_output_tokens_per_second"] == pytest.approx(198.2)


def test_cli_guidellm_dry_run(
    tmp_path: Path,
    guidellm_report: dict,
    published_metadata: dict,
    mock_hf_registry: None,
    capsys: pytest.CaptureFixture,
) -> None:
    results_path = tmp_path / "guidellm_report.json"
    results_path.write_text(json.dumps(guidellm_report))
    metadata_path = tmp_path / "published-metadata.json"
    metadata_path.write_text(json.dumps(published_metadata))
    exit_code = main(
        [
            str(results_path),
            "--kind",
            "guidellm",
            "--suite",
            "throughput",
            "--git-sha",
            "deadbeef",
            "--model-revision",
            "a" * 40,
            "--timestamp",
            "2026-04-24T18:30:00Z",
            "--published-metadata",
            str(metadata_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.err.lower() or "planned" in captured.err.lower()
