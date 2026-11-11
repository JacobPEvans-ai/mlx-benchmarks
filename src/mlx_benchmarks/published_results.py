"""Build Hub metadata records and HF ``.eval_results`` YAML entries."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi

from mlx_benchmarks.envelope import CLOUD_ENDPOINT_MODEL_KIND, Envelope, model_identity
from mlx_benchmarks.privacy import contains_private_identifier_keys
from mlx_benchmarks.result_contract import (
    MODEL_FIELDS,
    RESULT_FIELDS,
    validate_published_metadata,
    validate_published_result,
)

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_CONTEXT_FIELDS = ("max_position_embeddings", "max_seq_len", "max_sequence_length", "n_ctx")
DATASET_FIELDS = (
    "dataset_id",
    "dataset_task_id",
    "dataset_revision",
    "evaluation_framework",
    "config",
    "split",
)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    return {}


def _revision(revision: object, label: str) -> str:
    if not isinstance(revision, str) or not _SHA_RE.fullmatch(revision):
        raise ValueError(f"{label} must be the recorded 40-character Hub commit SHA")
    return revision


def _license_link(link: object, model_id: str, revision: str) -> object:
    # Model cards may link a repo file ("LICENSE"); resolve it to the pinned Hub blob URL.
    if isinstance(link, str) and link and "://" not in link:
        return f"https://huggingface.co/{model_id}/blob/{revision}/{link.lstrip('./')}"
    return link


def _model_fields(api: HfApi, model_id: str, revision: str) -> dict[str, Any]:
    info = api.model_info(repo_id=model_id, revision=_revision(revision, "model_revision"))
    card = _mapping(_value(info, "card_data"))
    config = _mapping(_value(info, "config"))
    safetensors = _value(info, "safetensors")
    gguf = _value(info, "gguf")
    if isinstance(gguf, list):
        gguf = gguf[0] if gguf else None

    card_pipeline = card.get("pipeline_tag")
    pipeline_tag = card_pipeline or _value(info, "pipeline_tag")
    quantization_config = _mapping(config.get("quantization_config"))
    gguf_type = _value(gguf, "type") or _value(gguf, "file_type")
    parameters = _value(safetensors, "parameters")
    model_sha = _value(info, "sha")
    if not isinstance(model_sha, str) or not _SHA_RE.fullmatch(model_sha):
        raise ValueError(f"Hub API returned no commit SHA for {model_id}@{revision}")
    if model_sha.casefold() != revision.casefold():
        raise ValueError(f"Hub API returned a different model SHA than the recorded revision for {model_id}")

    context_length = next((config[key] for key in _CONTEXT_FIELDS if config.get(key) is not None), None)
    return {
        "model_id": _value(info, "id") or model_id,
        "model_revision": model_sha,
        "pipeline_tag": pipeline_tag,
        "model_task": pipeline_tag,
        "model_task_source": "model_card" if card_pipeline else "inferred" if pipeline_tag else None,
        "library_name": _value(info, "library_name"),
        "license": card.get("license"),
        "license_name": card.get("license_name"),
        "license_link": _license_link(card.get("license_link"), model_id, model_sha),
        "base_model": card.get("base_model"),
        "base_model_relation": card.get("base_model_relation"),
        "tags": list(_value(info, "tags", []) or []),
        "parameters_total": _value(safetensors, "total") or _value(gguf, "total"),
        "dtype": parameters,
        "quant": gguf_type or quantization_config.get("quant_method"),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "context_length": context_length or _value(gguf, "context_length"),
        "gated": _value(info, "gated"),
    }


def _cloud_endpoint_fields(envelope: Envelope) -> dict[str, Any]:
    """Declared model fields for a cloud-endpoint record, with no Hub lookup.

    ``model_id`` is the declared provider-plus-model identity; every
    Hub-derived field stays null and picks up a ``not_applicable`` reason from
    the shared null-reason pass in :func:`build_published_results`.
    """
    provider = envelope.get("model_provider")
    if not isinstance(provider, str) or not provider:
        raise ValueError("cloud-endpoint records must declare model_provider")
    if envelope.get("model_revision") is not None:
        raise ValueError("cloud-endpoint records must not record a Hub model_revision")
    fields: dict[str, Any] = dict.fromkeys(MODEL_FIELDS)
    fields["model_id"] = model_identity(envelope)
    return fields


def _registered_task(api: HfApi, metadata: dict[str, Any]) -> dict[str, Any]:
    dataset_id = metadata.get("dataset_id")
    dataset_revision = _revision(metadata.get("dataset_revision"), "dataset_revision")
    task_id = metadata.get("dataset_task_id")
    if not isinstance(dataset_id, str) or not dataset_id or not isinstance(task_id, str) or not task_id:
        raise ValueError("dataset_id and dataset_task_id are required")

    info = api.dataset_info(repo_id=dataset_id, revision=dataset_revision)
    dataset_sha = _value(info, "sha")
    if not isinstance(dataset_sha, str) or not _SHA_RE.fullmatch(dataset_sha):
        raise ValueError(f"Hub API returned no dataset SHA for {dataset_id}@{dataset_revision}")
    if dataset_sha.casefold() != dataset_revision.casefold():
        raise ValueError(
            f"Hub API returned a different dataset SHA than the recorded revision for {dataset_id}"
        )
    tags = set(_value(info, "tags", []) or [])
    # The Hub tags HF Benchmarks as benchmark:official / benchmark:eval-yaml.
    if not any(tag == "benchmark" or str(tag).startswith("benchmark:") for tag in tags):
        raise ValueError(f"{dataset_id} at the requested revision is not tagged as an HF Benchmark")

    files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset", revision=dataset_revision)
    if "eval.yaml" not in files:
        raise ValueError(f"{dataset_id} has no eval.yaml and is not a registered HF Benchmark")
    eval_yaml_path = api.hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        filename="eval.yaml",
        revision=dataset_revision,
    )
    with Path(eval_yaml_path).open(encoding="utf-8") as source:
        definition = yaml.safe_load(source)

    if not isinstance(definition, dict):
        raise ValueError(f"{dataset_id}/eval.yaml must contain a mapping")
    tasks = definition.get("tasks", [])
    task = next((candidate for candidate in tasks if candidate.get("id") == task_id), None)
    if task is None:
        raise ValueError(f"task {task_id!r} is not defined in {dataset_id}/eval.yaml")
    framework = definition.get("evaluation_framework")
    config = task.get("config") or "default"
    split = task.get("split") or "test"
    if metadata.get("evaluation_framework") != framework:
        raise ValueError("evaluation_framework does not match the registered eval.yaml")
    if metadata.get("config") != config or metadata.get("split") != split:
        raise ValueError("config and split must match the registered eval.yaml task")
    return {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_sha,
        "dataset_task_id": task_id,
        "evaluation_framework": framework,
        "config": config,
        "split": split,
    }


def _dataset_result(metadata: dict[str, Any], api: HfApi) -> dict[str, Any]:
    if all(metadata.get(field) is None for field in DATASET_FIELDS):
        reasons = metadata.get("dimension_null_reasons") or {}
        if not all(reasons.get(field) == "not_applicable" for field in DATASET_FIELDS):
            raise ValueError("dataset fields may all be null only when each has a not_applicable reason")
        return dict.fromkeys(DATASET_FIELDS)
    return _registered_task(api, metadata)


def build_published_results(
    envelope: Envelope,
    *,
    metadata: dict[str, Any],
    api: HfApi,
) -> list[dict[str, Any]]:
    """Enrich result scores with pinned HF model and registered benchmark metadata."""
    validate_published_metadata(metadata)
    reserved = MODEL_FIELDS | RESULT_FIELDS
    overlaps = reserved & metadata.keys()
    if overlaps:
        raise ValueError(
            f"published metadata must not override Hub or score fields: {', '.join(sorted(overlaps))}"
        )

    model_id = envelope.get("model")
    model_revision = envelope.get("model_revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model id is required")
    if envelope.get("model_kind") == CLOUD_ENDPOINT_MODEL_KIND:
        model = _cloud_endpoint_fields(envelope)
    else:
        model = _model_fields(api, model_id, _revision(model_revision, "model_revision"))
    if envelope.get("campaign", {}).get("profile") == "stage0-system-load":
        dimensions = envelope.get("campaign_dimensions") or {}
        model_dimensions = dimensions.get("model") or {}
        selected_pipeline_tag = model_dimensions.get("pipeline_tag")
        if not isinstance(selected_pipeline_tag, str) or not selected_pipeline_tag:
            raise ValueError("Stage 0 requires the selected catalog pipeline_tag")
        if model["pipeline_tag"] != selected_pipeline_tag or model["model_task"] != selected_pipeline_tag:
            raise ValueError("published model_task must equal the selected catalog pipeline_tag")
    benchmark = _dataset_result(metadata, api)

    timestamps = envelope.get("timestamp")
    base_reasons = dict(metadata.get("dimension_null_reasons") or {})
    if model["license"] != "other":
        model["license_name"] = None
        model["license_link"] = None
    for field, value in model.items():
        if value is None:
            base_reasons[field] = "not_applicable"
    hardware = metadata.get("hardware")
    if contains_private_identifier_keys(hardware):
        raise ValueError("hardware metadata must not include host or network identifiers")
    alias = metadata.get("router_key_alias")
    if isinstance(alias, str) and re.search(r"(?:hf|sk)-[A-Za-z0-9_-]{20,}", alias):
        raise ValueError("router_key_alias looks like a credential rather than an alias")

    run_fields = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "dimension_null_reasons",
            "dataset_id",
            "dataset_revision",
            "dataset_task_id",
            "evaluation_framework",
            "config",
            "split",
        }
    }

    output: list[dict[str, Any]] = []
    for result in envelope.get("results", []):
        row = {
            **model,
            **benchmark,
            **run_fields,
            "value": result.get("value"),
            "metric": result.get("metric"),
            "date": timestamps,
            "dimension_null_reasons": dict(base_reasons),
        }
        validate_published_result(row)
        output.append(row)
    return output


def eval_results_yaml(row: dict[str, Any]) -> str:
    """Serialize the current HF ``.eval_results`` shape without adding Hub fields."""
    required = ("dataset_id", "dataset_task_id", "dataset_revision", "value", "metric", "date")
    missing = [field for field in required if row.get(field) is None]
    if missing:
        raise ValueError(f"cannot emit an HF eval result with null required fields: {', '.join(missing)}")
    record: dict[str, Any] = {
        "dataset": {
            "id": row["dataset_id"],
            "task_id": row["dataset_task_id"],
            "revision": row["dataset_revision"],
        },
        "value": row["value"],
        "date": row["date"],
    }
    source_url = row.get("source_url")
    if source_url:
        record["source"] = {"url": source_url}
    notes = row.get("notes")
    metric = row.get("metric")
    metric_note = f"metric={metric}" if metric else None
    combined_notes = "; ".join(value for value in (notes, metric_note) if value)
    if combined_notes:
        record["notes"] = combined_notes
    return yaml.safe_dump([record], sort_keys=False, allow_unicode=True)


def eval_results_path(row: dict[str, Any], index: int) -> str:
    """Return a collision-resistant, public-safe score sidecar path."""
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", str(row["run_id"])).strip("-.")
    metric = re.sub(r"[^A-Za-z0-9._-]+", "-", str(row["metric"])).strip("-.")
    if not run_id or not metric:
        raise ValueError("run_id and metric must produce a non-empty sidecar path")
    stamp = datetime.fromisoformat(row["date"].replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%S")
    return f".eval_results/{stamp}-{run_id}-{index:03d}-{metric}.yaml"
