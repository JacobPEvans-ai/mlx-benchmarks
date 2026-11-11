"""``mlx-bench-publish-envelope`` CLI and the publish-envelopes workflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from mlx_benchmarks.publish_envelope import main

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-envelopes.yml"


def _write_run(run_dir: Path, envelope: dict[str, Any], metadata: dict[str, Any] | None) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    envelope_path = run_dir / "envelope.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    metadata_path = run_dir / "metadata.json"
    if metadata is not None:
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return envelope_path, metadata_path


def test_dry_run_plans_valid_pair(
    valid_envelope: dict[str, Any],
    published_metadata: dict[str, Any],
    mock_hf_registry: None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    envelope_path, metadata_path = _write_run(tmp_path / "run-1", valid_envelope, published_metadata)
    caplog.set_level(logging.INFO, logger="mlx_benchmarks")

    assert main(["--pair", str(envelope_path), str(metadata_path), "--dry-run"]) == 0
    assert "dry-run: would publish" in caplog.text
    assert f"{envelope_path} -> data/run-canonical-" in caplog.text


def test_missing_metadata_exits_1_and_names_file(
    valid_envelope: dict[str, Any],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    envelope_path, metadata_path = _write_run(tmp_path / "run-1", valid_envelope, None)
    caplog.set_level(logging.ERROR, logger="mlx_benchmarks")

    assert main(["--pair", str(envelope_path), str(metadata_path), "--dry-run"]) == 1
    assert str(metadata_path) in caplog.text


def test_workflow_is_read_only_and_token_is_scoped_to_publish_step() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    # PyYAML parses the bare `on:` key as boolean True (YAML 1.1).
    triggers = doc.get("on", doc.get(True))

    assert doc["permissions"] == {"contents": "read"}
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["push"]["paths"] == ["published/pending/**.json"]
    assert "workflow_dispatch" in triggers
    assert doc["concurrency"]["cancel-in-progress"] is False

    assert "HF_TOKEN" not in doc.get("env", {})
    for job in doc["jobs"].values():
        assert "HF_TOKEN" not in job.get("env", {})
    token_steps = [
        step for job in doc["jobs"].values() for step in job["steps"] if "HF_TOKEN" in step.get("env", {})
    ]
    assert len(token_steps) == 1
    [publish_step] = token_steps
    assert publish_step["env"] == {"HF_TOKEN": "${{ secrets.HF_TOKEN_WRITE_ALL }}"}
    assert "mlx-bench-publish-envelope" in publish_step["run"]
    assert text.count("HF_TOKEN_WRITE_ALL") == 1
