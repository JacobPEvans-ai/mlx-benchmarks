"""End-to-end: dense-layout runner sample -> envelope -> passes schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import detect_system


def _ctx(**overrides: object) -> ConverterContext:
    defaults: dict = {
        "suite": "dense-layout-image-text",
        "model": "example-dense-layout-2b",
        "git_sha": "deadbeef",
        "system": detect_system(),
    }
    defaults.update(overrides)
    return ConverterContext(**defaults)


def _rows(envelope: dict, label: str) -> dict[str, dict]:
    return {row["metric"]: row for row in envelope["results"] if row["tags"]["label"] == label}


def test_dense_layout_round_trip(dense_layout_sample: dict) -> None:
    envelope = get_converter("dense-layout").build_envelope(dense_layout_sample, _ctx())
    validate_envelope(envelope)

    assert envelope["suite"] == "dense-layout-image-text"
    assert envelope["timestamp"] == "2026-11-13T00:00:00Z"
    assert envelope["model"] == "example-dense-layout-2b"
    assert all(row["name"] == "dense-layout" for row in envelope["results"])
    # 3 blocks (2 labels + overall) x 5 metrics
    assert len(envelope["results"]) == 15


def test_every_row_declares_unit_and_direction(dense_layout_sample: dict) -> None:
    envelope = get_converter("dense-layout").build_envelope(dense_layout_sample, _ctx())
    expected = {
        "cer": ("edit_distance_per_reference_char", "lower_is_better"),
        "wer": ("edit_distance_per_reference_word", "lower_is_better"),
        "field_recall": ("ratio", "higher_is_better"),
        "seconds_per_image": ("seconds", "lower_is_better"),
        "error_rate": ("ratio", "lower_is_better"),
    }
    for row in envelope["results"]:
        unit, direction = expected[row["metric"]]
        assert row["unit"] == unit
        # CER/WER are unbounded above, so their unit must never read as a [0, 1] ratio.
        assert row["tags"]["direction"] == direction


def test_label_thinking_and_concurrency_travel_as_tags(dense_layout_sample: dict) -> None:
    envelope = get_converter("dense-layout").build_envelope(dense_layout_sample, _ctx())
    labels = {row["tags"]["label"] for row in envelope["results"]}
    assert labels == {"tables", "forms", "all"}
    for row in envelope["results"]:
        assert row["tags"]["thinking"] == "n/a"
        assert row["tags"]["concurrency"] == "1"
    n_records_by_label = {
        **{k: str(v["n_records"]) for k, v in dense_layout_sample["labels"].items()},
        "all": str(dense_layout_sample["overall"]["n_records"]),
    }
    for row in envelope["results"]:
        assert row["tags"]["n_records"] == n_records_by_label[row["tags"]["label"]]


def test_aggregate_values_survive_conversion(dense_layout_sample: dict) -> None:
    envelope = get_converter("dense-layout").build_envelope(dense_layout_sample, _ctx())
    overall = _rows(envelope, "all")
    assert overall["cer"]["value"] == pytest.approx(0.045)
    assert overall["wer"]["value"] == pytest.approx(0.09)
    assert overall["field_recall"]["value"] == pytest.approx(0.833333)
    assert overall["seconds_per_image"]["value"] == pytest.approx(1.22)
    # The harness writes aggregates rounded to 6 decimals; assert that value,
    # not the unrounded 1/6, which sits outside pytest.approx's tolerance.
    assert overall["error_rate"]["value"] == pytest.approx(round(1 / 6, 6))
    tables = _rows(envelope, "tables")
    assert tables["error_rate"]["value"] == pytest.approx(0.25)
    assert tables["cer"]["value"] == pytest.approx(0.033333)


def test_duration_seconds_only_on_overall_rows(dense_layout_sample: dict) -> None:
    envelope = get_converter("dense-layout").build_envelope(dense_layout_sample, _ctx())
    for row in envelope["results"]:
        if row["tags"]["label"] == "all":
            assert row["duration_seconds"] == 7.0
        else:
            assert "duration_seconds" not in row


def test_extra_tags_are_carried(dense_layout_sample: dict) -> None:
    envelope = get_converter("dense-layout").build_envelope(
        dense_layout_sample, _ctx(extra_tags={"env_class": "isolated"})
    )
    assert all(row["tags"]["env_class"] == "isolated" for row in envelope["results"])


def test_missing_overall_rejected() -> None:
    with pytest.raises(ValueError, match="no overall"):
        get_converter("dense-layout").build_envelope({"labels": {}}, _ctx())


def test_kind_is_registered() -> None:
    assert get_converter("dense-layout").kind == "dense-layout"


def test_cli_dry_run_accepts_the_committed_fixture(
    tmp_path: Path,
    dense_layout_sample: dict,
    published_metadata: dict,
    mock_hf_registry: None,
) -> None:
    # The full publisher import chain (pyarrow, huggingface_hub) is only present
    # where the package is installed with its dependencies.
    pytest.importorskip("pyarrow")
    from mlx_benchmarks.cli import main

    results_path = tmp_path / "dense_layout_sample.json"
    results_path.write_text(json.dumps(dense_layout_sample))
    metadata_path = tmp_path / "published-metadata.json"
    metadata_path.write_text(json.dumps(published_metadata))
    exit_code = main(
        [
            str(results_path),
            "--kind",
            "dense-layout",
            "--suite",
            "dense-layout-image-text",
            "--git-sha",
            "deadbeef",
            # Must match published_metadata.start_utc: the publisher validates
            # that the envelope date equals the recorded run start.
            "--timestamp",
            "2026-04-24T18:30:00Z",
            "--published-metadata",
            str(metadata_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
