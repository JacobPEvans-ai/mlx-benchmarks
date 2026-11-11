"""Serialize envelope -> Parquet and upload to the HuggingFace dataset repo."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import subprocess
from pathlib import PurePosixPath
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

from mlx_benchmarks.dataset_schema import (
    CAMPAIGN_DIMENSION_JSON_FIELDS,
    CAMPAIGN_DIMENSION_TYPES,
    DERIVED_THROUGHPUT_SPEED_FIELDS,
    JSON_STRING_COLUMNS,
    OPTIONAL_RESULT_COLUMNS,
    OPTIONAL_RESULT_STRING_COLUMNS,
    PARQUET_ROW_SCHEMA,
    PUBLISHED_JSON_COLUMNS,
    SYSTEM_COLUMNS,
    TAG_KEYS,
    THROUGHPUT_SUITES,
    campaign_dimension_column,
    empty_parquet_row,
)
from mlx_benchmarks.engine_versions import populate_engine_release_metadata
from mlx_benchmarks.envelope import Envelope, model_identity, validate_envelope
from mlx_benchmarks.power_metrics import PowerMetricsError, enrich_power_metrics
from mlx_benchmarks.privacy import remove_private_identifiers
from mlx_benchmarks.published_results import build_published_results, eval_results_path, eval_results_yaml
from mlx_benchmarks.result_contract import PublishedResultValidationError

log = logging.getLogger(__name__)

DEFAULT_REPO_ID = "JacobPEvans/mlx-benchmarks"
DEFAULT_REPO_TYPE = "dataset"


class PublishError(RuntimeError):
    """Raised for publisher-layer failures (auth, upload, empty results, ...)."""


def prepare_envelope(envelope: Envelope) -> None:
    """Apply shared metadata enrichment and schema checks before serialization or publishing."""
    populate_engine_release_metadata(envelope)
    validate_envelope(envelope)
    try:
        enrich_power_metrics(envelope)
    except PowerMetricsError as exc:
        raise PublishError(str(exc)) from exc
    validate_envelope(envelope)


def slugify(model: str) -> str:
    """Filesystem-safe slug of a model identifier.

    ``mlx-community/Qwen3.5-9B-MLX-4bit`` -> ``mlx-community-qwen3-5-9b-mlx-4bit``.
    """
    return re.sub(r"[^a-zA-Z0-9-]", "-", model).lower().strip("-")


def current_git_sha(fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=3).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return fallback


def envelope_to_rows(envelope: Envelope) -> list[dict[str, Any]]:
    """Explode ``envelope['results']`` into one flat row per measurement.

    Optional result fields (``duration_seconds``, the ``*_tokens_per_second``
    set, etc.) and tag-derived columns are normalized across rows so PyArrow's
    schema inference sees a consistent column set. Without this, a parquet
    written from a mixed envelope where the *first* row lacks a field but a
    later row has it would silently drop the column.
    """
    system = dict(envelope.get("system") or {})
    system.pop("hostname", None)
    system = remove_private_identifiers(system)
    base = empty_parquet_row()
    base.update(
        {
            "schema_version": envelope.get("schema_version"),
            "timestamp": envelope.get("timestamp"),
            "git_sha": envelope.get("git_sha"),
            "trigger": envelope.get("trigger"),
            "suite": envelope.get("suite"),
            "model": envelope.get("model"),
        }
    )
    for key in SYSTEM_COLUMNS:
        value = system.get(key)
        column = "system_engine" if key == "engine" else key
        if column in JSON_STRING_COLUMNS and isinstance(value, dict | list):
            value = json.dumps(value, sort_keys=True)
        base[column] = value
    system_extra = {key: value for key, value in system.items() if key not in SYSTEM_COLUMNS}
    base["system_extra"] = json.dumps(system_extra, sort_keys=True)
    base["extra_json"] = "{}"

    # Dynamic-key access over a plain mapping view — these optional top-level
    # scalars are copied through verbatim; envelope is a JSON object at runtime.
    env_map = cast("dict[str, Any]", envelope)
    campaign_dimensions = env_map.get("campaign_dimensions") or {}
    dimension_null_reasons = dict(env_map.get("dimension_null_reasons") or {})
    for category, fields in CAMPAIGN_DIMENSION_TYPES.items():
        values = campaign_dimensions.get(category) or {}
        for field in fields:
            value = values.get(field)
            if value is not None and (category, field) in CAMPAIGN_DIMENSION_JSON_FIELDS:
                value = json.dumps(value, sort_keys=True)
            base[campaign_dimension_column(category, field)] = value
    if envelope.get("suite") in THROUGHPUT_SUITES:
        speed = campaign_dimensions.get("speed") or {}
        for field in DERIVED_THROUGHPUT_SPEED_FIELDS:
            if speed.get(field) is None:
                dimension_null_reasons.setdefault(f"speed.{field}", "derived_input_unavailable")
    base["campaign_dimension_null_reasons_json"] = json.dumps(dimension_null_reasons, sort_keys=True)

    for key in (
        "model_kind",
        "model_provider",
        "model_revision",
        "model_task",
        "model_task_source",
        "quantization",
        "seed",
        "env_class",
        "concurrency",
        "reasoning_effort",
    ):
        if key in env_map:
            base[key] = env_map[key]
    if "serving" in envelope:
        base["serving"] = json.dumps(envelope["serving"], sort_keys=True)

    results = envelope.get("results", [])
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            **base,
            "name": r.get("name"),
            "metric": r.get("metric"),
            "value": r.get("value"),
            "unit": r.get("unit"),
        }
        for col in OPTIONAL_RESULT_COLUMNS:
            row[col] = r.get(col)
        for col in OPTIONAL_RESULT_STRING_COLUMNS:
            row[col] = r.get(col)
        tags = remove_private_identifiers(r.get("tags") or {})
        for key in TAG_KEYS:
            row[f"tag_{key}"] = tags.get(key)
        row["tags_json"] = json.dumps(tags, sort_keys=True)
        rows.append(row)
    return rows


def rows_to_parquet(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise PublishError("No result rows to serialize — envelope has empty results[]")
    unexpected = {key for row in rows for key in row} - set(PARQUET_ROW_SCHEMA.names)
    if unexpected:
        raise PublishError(
            f"Rows contain columns outside the canonical schema: {', '.join(sorted(unexpected))}"
        )
    table = pa.Table.from_pylist(rows, schema=PARQUET_ROW_SCHEMA)
    buf = io.BytesIO()
    pq.write_table(table, buf)  # type: ignore[unused-ignore,no-untyped-call]
    return buf.getvalue()


def canonical_shard_path(source_path: str) -> str:
    """Return the immutable normalized path for a top-level dataset shard."""
    filename = PurePosixPath(source_path).name
    if (
        not source_path.startswith("data/")
        or "/" in source_path.removeprefix("data/")
        or not filename.endswith(".parquet")
        or filename.startswith("run-canonical-")
    ):
        raise ValueError(f"not an original dataset shard: {source_path}")
    return f"data/run-canonical-{filename}"


def target_path(envelope: Envelope, payload: bytes | None = None) -> str:
    """Deterministic HF-dataset path for this envelope.

    Format: ``data/run-canonical-run-<ts_slug>-<git_sha>-<suite>-<model_slug>-<payload_hash>.parquet``.

    ``payload_hash`` is an 8-char SHA-256 prefix of the serialized parquet,
    which (a) guarantees that two runs producing different metrics in the same
    second cannot collide even if every other dimension matches, and (b)
    content-addresses the shard so a deterministic re-publish is a no-op
    instead of silently overwriting. Pass ``payload=None`` to get the
    pre-hash prefix (useful only for tests inspecting the slug composition).
    """
    timestamp: str = envelope["timestamp"]
    ts_slug = timestamp.replace(":", "-").rstrip("Z")
    # Cloud-endpoint records slug the declared provider plus model identity;
    # local models slug the Hub model id.
    model_slug = slugify(model_identity(envelope))[:50]
    prefix = f"data/run-canonical-run-{ts_slug}-{envelope['git_sha']}-{envelope['suite']}-{model_slug}"
    if payload is None:
        return f"{prefix}.parquet"
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{prefix}-{digest}.parquet"


def publish(
    envelope: Envelope,
    *,
    published_metadata: dict[str, Any] | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    dry_run: bool = False,
    token: str | None = None,
) -> str:
    """Validate, serialize and optionally upload ``envelope`` to the HF dataset.

    A dry run still reads public Hub metadata for the pinned model and benchmark;
    it never writes. Real uploads also require ``HF_TOKEN`` (or ``token=...``).
    """
    if repo_type == DEFAULT_REPO_TYPE:
        serving = envelope.get("serving")
        if isinstance(serving, dict) and "endpoint_port" in serving:
            raise PublishError("serving.endpoint_port is retained for legacy rows and cannot be published")
    prepare_envelope(envelope)
    if not isinstance(published_metadata, dict):
        raise PublishError("--published-metadata is required for every published score")

    rows = envelope_to_rows(envelope)
    effective_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=effective_token)
    try:
        published_results = build_published_results(envelope, metadata=published_metadata, api=api)
    except (HfHubHTTPError, PublishedResultValidationError, TypeError, ValueError) as exc:
        raise PublishError(f"published result validation failed: {exc}") from exc
    if len(published_results) != len(rows):
        raise PublishError("published result metadata count differs from the benchmark score count")
    operations: list[CommitOperationAdd] = []
    for index, (row, result) in enumerate(zip(rows, published_results, strict=True), start=1):
        for field, value in result.items():
            if (
                field in PUBLISHED_JSON_COLUMNS
                and value is not None
                and not (field == "base_model" and isinstance(value, str))
            ):
                value = json.dumps(value, sort_keys=True, allow_nan=False)
            elif field == "gated" and value is not None:
                value = str(value).lower()
            row[field] = value
        if result["dataset_id"] is not None:
            operations.append(
                CommitOperationAdd(
                    path_in_repo=eval_results_path(result, index),
                    path_or_fileobj=eval_results_yaml(result).encode("utf-8"),
                )
            )
    parquet_bytes = rows_to_parquet(rows)
    path = target_path(envelope, payload=parquet_bytes)
    eval_results_count = sum(result["dataset_id"] is not None for result in published_results)
    operations.insert(
        0,
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=parquet_bytes),
    )

    if dry_run:
        log.info(
            "dry-run: would publish %d Parquet bytes and %d HF eval result record(s) to %s in %s",
            len(parquet_bytes),
            eval_results_count,
            path,
            repo_id,
        )
        return path

    if not effective_token:
        raise PublishError("HF_TOKEN not set — publishing needs HF_TOKEN or token=...")

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=f"feat: add {envelope['suite']} run for {envelope['model']}",
        )
    except HfHubHTTPError as exc:
        raise PublishError(f"HF upload failed for {path}: {exc}") from exc
    log.info("published %d-byte parquet to %s", len(parquet_bytes), path)

    # Local import: events reuses envelope_to_rows from this module.
    from mlx_benchmarks.events import append_events

    try:
        # Keep the original run basename while the stored path is canonical.
        run_id = PurePosixPath(path).stem.removeprefix("run-canonical-")
        append_events(envelope, run_id=run_id)
    except OSError as exc:
        # The feed is derived state (replayable from the dataset); never fail
        # a successful publish over it — but say so loudly.
        log.warning("bench-events append failed (publish succeeded, replay later): %s", exc)
    return path
