"""``mlx-bench-publish`` command-line entry point.

Typical use::

    mlx-bench-publish run-output/mymodel/results_*.json \\
        --kind lm-eval --suite reasoning

``--dry-run`` prints the planned upload without touching the network. Pass
``--output-dir`` to write validated envelope JSON without publishing.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, cast

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.converters.inspect_ai import read_inspect_header
from mlx_benchmarks.converters.nvidia_offload import (
    KV_ENGINE,
    KV_MACHINE,
    MOE_ENGINE,
    MOE_MACHINE,
    engine_metadata,
)
from mlx_benchmarks.envelope import CampaignDimensions, Envelope, EnvelopeValidationError, Serving
from mlx_benchmarks.publish import PublishError, current_git_sha, prepare_envelope, publish
from mlx_benchmarks.published_results import DATASET_FIELDS
from mlx_benchmarks.system import detect_system

log = logging.getLogger("mlx_benchmarks.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-bench-publish",
        description="Convert raw benchmark results to envelope v1 and publish to the HF dataset.",
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to raw tool output (e.g. lm-eval JSON, Inspect .eval log, KVO_RESULT JSONL, or MoE CSV; "
        ".jsonl is otherwise one JSON object per line; .txt/.log is wrapped as {'output': text} for mbw/gpu-burn)",
    )
    parser.add_argument(
        "--kind",
        default="lm-eval",
        choices=[
            "agentic",
            "agentic-partial",
            "bench-serve",
            "coding-replay",
            "dense-layout",
            "evalscope",
            "factual",
            "fio",
            "guidellm",
            "gpu-burn",
            "inspect-ai",
            "kv-offload",
            "llamacpp-server",
            "lm-eval",
            "mbw",
            "moe-offload",
            "nvbandwidth",
            "promptfoo",
            "promptstack",
            "tau2",
            "throughput-probe",
            "vllm",
        ],
        help="Source format of results_json",
    )
    parser.add_argument("--suite", required=True, help="Envelope suite (must be in schema enum)")
    parser.add_argument("--model", help="Override model ID (default: extract from results_json)")
    parser.add_argument(
        "--model-kind",
        choices=["local", "cloud-endpoint"],
        help="Declare a cloud endpoint so publication skips the Hub model lookup",
    )
    parser.add_argument(
        "--model-provider",
        help="Provider namespace for --model-kind cloud-endpoint (for example, Qwen)",
    )
    parser.add_argument("--model-revision", help="Recorded 40-character Hugging Face model commit")
    parser.add_argument("--git-sha", help="Override git SHA recorded in envelope (default: git rev-parse)")
    parser.add_argument(
        "--trigger",
        default="local",
        choices=["local", "schedule", "pr", "workflow_dispatch"],
        help="Envelope trigger field",
    )
    parser.add_argument("--pr-number", type=int, help="Pull request number when trigger=pr")
    parser.add_argument(
        "--env-class",
        choices=["isolated", "under-load"],
        help="Machine load class during the run (verdict-policy gate 3)",
    )
    parser.add_argument("--concurrency", type=int, help="In-flight request count the run drove (>=1)")
    # No `choices=`: each model family names its own levels, and normalising one
    # family's name into another's merges arms that are not the same arm.
    parser.add_argument(
        "--reasoning-effort",
        help="Thinking effort the run asked for, verbatim (low/high/xhigh/max/off/unstated)",
    )
    parser.add_argument("--serving-stack", help="Serving stack, e.g. 'mlx_lm.server' or 'vllm-mlx'")
    parser.add_argument("--serving-model", help="Model id as advertised by the serving endpoint")
    parser.add_argument("--engine-version", help="Target engine version for offload input formats")
    parser.add_argument("--engine-build", help="Target engine build for offload run-index entries")
    parser.add_argument("--engine-commit", help="Target engine commit for offload run-index entries")
    parser.add_argument(
        "--campaign-dimensions",
        type=Path,
        metavar="PATH",
        help="JSON file holding the envelope's campaign_dimensions object and, optionally, "
        "dimension_null_reasons (field path -> reason code); copied onto the envelope verbatim",
    )
    parser.add_argument(
        "--published-metadata",
        type=Path,
        metavar="PATH",
        help="JSON metadata for the registered HF Benchmark task and required run variables",
    )
    parser.add_argument(
        "--unregistered-benchmark",
        action="store_true",
        help="Null all benchmark dataset fields and mark them not_applicable",
    )
    parser.add_argument("--timestamp", help="Override envelope timestamp (ISO 8601 UTC, rarely needed)")
    parser.add_argument(
        "--run-index",
        type=Path,
        default=Path("metadata/run-index-v1.json"),
        help="Run index updated after successful offload-cell publication",
    )
    parser.add_argument(
        "--run-index-status",
        choices=["experimental", "scored"],
        default="experimental",
        help="Status written to offload-cell run-index entries",
    )
    parser.add_argument(
        "--run-index-caveat",
        help="Caveat written to offload-cell run-index entries (experimental runs get a default caveat)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Attach a tag to every result (repeat flag for multiple)",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--dry-run", action="store_true", help="Validate + plan only; do not upload")
    output_mode.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Write validated envelope JSON to DIR instead of publishing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--repo-id", default=None, help="Override HF dataset repo (default: JacobPEvans/mlx-benchmarks)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model_kind == "cloud-endpoint" and not args.model_provider:
        parser.error("--model-provider is required with --model-kind cloud-endpoint")
    if args.model_provider and args.model_kind != "cloud-endpoint":
        parser.error("--model-provider requires --model-kind cloud-endpoint")
    if args.model_kind == "cloud-endpoint" and args.model_revision:
        parser.error("--model-revision cannot be used with --model-kind cloud-endpoint")
    # Every log line starts with an ISO-8601 UTC timestamp ending in 'Z'.
    # gmtime makes asctime UTC; the literal Z in datefmt marks the zone. force=True
    # re-attaches a stderr handler instead of no-opping when root already has one.
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        force=True,
    )

    try:
        raw: Any
        if args.kind == "kv-offload":
            raw = _read_kvo_results(args.results_json)
        elif args.kind == "moe-offload":
            raw = _read_moe_csv(args.results_json)
        elif args.kind == "inspect-ai":
            raw = read_inspect_header(args.results_json)
        elif args.results_json.suffix in (".txt", ".log"):
            raw = {"output": args.results_json.read_text(newline="")}  # keep gpu-burn's CRs
        elif args.results_json.suffix == ".jsonl":
            raw = [json.loads(line) for line in args.results_json.read_text().splitlines() if line]
        else:
            raw = json.loads(args.results_json.read_text())
    except OSError as exc:
        log.error("cannot read %s: %s", args.results_json, exc)
        return 2
    except json.JSONDecodeError as exc:
        log.error("%s is not valid JSON: %s", args.results_json, exc)
        return 2
    except ValueError as exc:
        label = "a valid Inspect .eval log" if args.kind == "inspect-ai" else "valid input"
        log.error("%s is not %s: %s", args.results_json, label, exc)
        return 2

    campaign_dimensions: CampaignDimensions | None = None
    dimension_null_reasons: dict[str, str] | None = None
    model_task: str | None = None
    model_task_source: str | None = None
    if args.campaign_dimensions is not None:
        if args.kind in {"kv-offload", "moe-offload"}:
            log.error("offload converters derive per-cell dimensions from their input rows")
            return 2
        try:
            (
                campaign_dimensions,
                dimension_null_reasons,
                model_task,
                model_task_source,
            ) = _load_campaign_dimensions(args.campaign_dimensions)
        except OSError as exc:
            log.error("cannot read %s: %s", args.campaign_dimensions, exc)
            return 2
        except ValueError as exc:
            log.error("%s is not a campaign dimensions file: %s", args.campaign_dimensions, exc)
            return 2

    published_metadata: dict[str, Any] | None = None
    if args.published_metadata is not None:
        try:
            metadata = json.loads(args.published_metadata.read_text())
            if not isinstance(metadata, dict):
                raise ValueError("expected a JSON object")
            published_metadata = metadata
        except OSError as exc:
            log.error("cannot read %s: %s", args.published_metadata, exc)
            return 2
        except json.JSONDecodeError as exc:
            log.error("%s is not valid JSON: %s", args.published_metadata, exc)
            return 2
        except ValueError as exc:
            log.error("%s is not published metadata: %s", args.published_metadata, exc)
            return 2

    if campaign_dimensions is not None:
        software = campaign_dimensions.get("software")
        if isinstance(software, dict) and isinstance(software.get("engine"), str):
            software["engine"] = _normalize_engine_name(software["engine"])
    if published_metadata is not None and isinstance(published_metadata.get("engine"), str):
        published_metadata["engine"] = _normalize_engine_name(published_metadata["engine"])

    if args.unregistered_benchmark and published_metadata is not None:
        published_metadata.update(dict.fromkeys(DATASET_FIELDS))
        null_reasons = published_metadata.get("dimension_null_reasons")
        if null_reasons is None:
            null_reasons = {}
        if isinstance(null_reasons, dict):
            null_reasons.update(dict.fromkeys(DATASET_FIELDS, "not_applicable"))
            published_metadata["dimension_null_reasons"] = null_reasons

    git_sha = args.git_sha or current_git_sha()
    extra_tags = dict(_parse_kv_pairs(args.tag))
    offload_kind = args.kind in {"kv-offload", "moe-offload"}
    system = {} if offload_kind else detect_system()
    system_engine = system.get("engine")
    if isinstance(system_engine, dict) and isinstance(system_engine.get("name"), str):
        system_engine["name"] = _normalize_engine_name(system_engine["name"])

    serving: Serving = {}
    if args.serving_stack is not None:
        serving["stack"] = _normalize_engine_name(args.serving_stack)
    if args.serving_model is not None:
        serving["served_model"] = args.serving_model

    converter = get_converter(args.kind)
    batch = offload_kind
    cells = raw if batch else [raw]
    if not isinstance(cells, list) or not cells:
        log.error("%s contains no benchmark cells", args.results_json)
        return 2
    if batch and not args.dry_run:
        try:
            _load_run_index(args.run_index)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            log.error("cannot use run index %s: %s", args.run_index, exc)
            return 2

    for index, cell in enumerate(cells, start=1):
        if batch and not isinstance(cell, dict):
            log.error("cell %d is not a JSON object", index)
            return 2
        model = args.model or _extract_model(cell)
        cell_version = args.engine_version
        cell_build = args.engine_build
        cell_commit = args.engine_commit
        if published_metadata is not None:
            cell_version = cell_version or published_metadata.get("engine_version")
            cell_build = cell_build or published_metadata.get("engine_build")
            cell_commit = cell_commit or published_metadata.get("engine_commit")
        ctx = ConverterContext(
            suite=args.suite,
            model=model,
            git_sha=git_sha,
            trigger=args.trigger,
            pr_number=args.pr_number,
            env_class=args.env_class,
            concurrency=args.concurrency,
            engine_version=cell_version,
            engine_build=cell_build,
            engine_commit=cell_commit,
            reasoning_effort=args.reasoning_effort,
            serving=serving or None,
            campaign_dimensions=campaign_dimensions,
            dimension_null_reasons=dimension_null_reasons,
            model_task=model_task,
            model_task_source=model_task_source,
            timestamp_override=args.timestamp,
            system=system,
            extra_tags=extra_tags,
            source_path=args.results_json,
        )
        try:
            if (
                args.kind == "kv-offload"
                and isinstance(cell, dict)
                and cell.get("suite") not in (None, args.suite)
            ):
                raise ValueError(f"KVO_RESULT suite {cell['suite']!r} does not match --suite {args.suite!r}")
            envelope = converter.build_envelope(cell, ctx)
            if args.model_kind is not None:
                envelope["model_kind"] = args.model_kind
            if args.model_provider is not None:
                envelope["model_provider"] = args.model_provider
            input_revision = _extract_model_revision(cell)
            if args.model_kind == "cloud-endpoint" and input_revision is not None:
                raise ValueError("cloud-endpoint results must not contain a Hub model revision")
            model_revision = (
                None if args.model_kind == "cloud-endpoint" else (input_revision or args.model_revision)
            )
            if batch and not model_revision:
                raise ValueError("offload publication needs the model's recorded --model-revision")
            if model_revision is not None:
                envelope["model_revision"] = model_revision
            cell_metadata = copy.deepcopy(published_metadata)
            if batch:
                engine_name = KV_ENGINE if args.kind == "kv-offload" else MOE_ENGINE
                machine = KV_MACHINE if args.kind == "kv-offload" else MOE_MACHINE
                cell_metadata = cell_metadata or {}
                cell_metadata.update(engine_metadata(cell, ctx, engine_name))
                cell_metadata["machine"] = machine
                if cell.get("gpu_power_limit_w") not in (None, ""):
                    cell_metadata["power_limit_w"] = float(cell["gpu_power_limit_w"])
                _validate_run_index_metadata(args, cell_metadata)
            log.info(
                "built envelope for cell %d/%d with %d results for model=%s suite=%s",
                index,
                len(cells),
                len(envelope.get("results", [])),
                model,
                args.suite,
            )
            if args.output_dir is not None:
                prepare_envelope(envelope)
                path = _write_local_envelope(args.output_dir, envelope)
            else:
                publish_kwargs: dict[str, Any] = {
                    "dry_run": args.dry_run,
                    "published_metadata": cell_metadata,
                }
                if args.repo_id:
                    publish_kwargs["repo_id"] = args.repo_id
                path = publish(envelope, **publish_kwargs)
                if batch and not args.dry_run:
                    entry = _run_index_entry(args, cell, envelope, path, cell_metadata or {})
                    _append_run_index_entry(args.run_index, entry)
        except EnvelopeValidationError as exc:
            log.error("%s", exc)
            return 3
        except PublishError as exc:
            action = "envelope preparation" if args.output_dir is not None else "publish"
            log.error("%s failed: %s", action, exc)
            return 4
        except (OSError, ValueError, KeyError) as exc:
            log.error("offload cell %d failed: %s", index, exc)
            return 5 if batch else 2

        action = "saved" if args.output_dir is not None else ("planned" if args.dry_run else "published")
        log.info("%s -> %s", action, path)
    return 0


def _write_local_envelope(output_dir: Path, envelope: Envelope) -> Path:
    payload = json.dumps(envelope, allow_nan=False, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    timestamp = envelope["timestamp"].replace(":", "-").rstrip("Z")
    path = output_dir / f"run-{timestamp}-{envelope['suite']}-{digest}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as result:
        result.write(payload)
    return path


_MOE_CSV_FIELDS = {
    "profile",
    "model",
    "arm",
    "prompt_tokens",
    "output_tokens",
    "concurrency",
    "decode_tok_s",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "vram_used_gib",
    "guest_ram_peak_gib",
    "n_cpu_moe",
    "cache_ram_mib",
    "semaphore_task",
}


def _read_kvo_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        prefix, separator, payload = line.partition(" ")
        if prefix != "KVO_RESULT" or not separator:
            raise ValueError(f"line {line_number} must start with 'KVO_RESULT '")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError("no KVO_RESULT lines found")
    return rows


def _read_moe_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing")
        missing = sorted(_MOE_CSV_FIELDS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no MoE cells")
    return rows


def _validate_run_index_metadata(args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    engine = metadata.get("engine")
    version = metadata.get("engine_version")
    if not isinstance(engine, str) or not engine or not isinstance(version, str) or not version:
        raise ValueError(
            "offload publication needs an engine version in the result, --engine-version, or published metadata"
        )
    if args.run_index_status == "scored" and engine == "llama.cpp":
        build = metadata.get("engine_build")
        commit = metadata.get("engine_commit")
        if not isinstance(build, str) or not isinstance(commit, str):
            raise ValueError("scored llama.cpp entries need --engine-build and --engine-commit")
        if not build.startswith("b") or not build[1:].isdigit() or build != version:
            raise ValueError("scored llama.cpp engine version and build must use the same bNNNNN release tag")
        if not 7 <= len(commit) <= 40 or any(char not in "0123456789abcdef" for char in commit):
            raise ValueError(
                "scored llama.cpp engine commit must be 7 to 40 lowercase hexadecimal characters"
            )
    if args.run_index_status == "scored" and engine == "vLLM" and version not in {"0.31.0", "0.32.0"}:
        raise ValueError(f"unsupported scored vLLM version for run index: {version}")


def _run_index_entry(
    args: argparse.Namespace,
    raw: dict[str, Any],
    envelope: Envelope,
    path: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if args.kind == "kv-offload":
        arm = str(raw.get("arm", "unknown"))
        workload = str(raw.get("workload", "unknown"))
        variant = f"{arm} {workload}"
        comparison_group = (
            f"{metadata['machine']}|{metadata['engine']}|{envelope['model']}|{envelope['suite']}|{workload}"
        )
        context_band = "multi-round"
        max_output_tokens = raw.get("answer_tokens")
        request_count = raw.get("csv_request_count")
        completion_state = f"complete {request_count} requests" if request_count is not None else "complete"
    else:
        profile = str(raw.get("profile", "unknown"))
        arm = str(raw.get("arm", "unknown"))
        prompt_tokens = _index_integer(raw.get("prompt_tokens"))
        output_tokens = _index_integer(raw.get("output_tokens"))
        concurrency = _index_integer(raw.get("concurrency"))
        variant = f"{profile} {arm}"
        comparison_group = (
            f"{metadata['machine']}|{metadata['engine']}|{envelope['model']}|{profile}|"
            f"{prompt_tokens}|{output_tokens}|{concurrency}"
        )
        context_band = f"{max(1, math.ceil(prompt_tokens / 1024))}k" if prompt_tokens else "unknown"
        max_output_tokens = output_tokens
        completion_state = "complete"
    caveat = args.run_index_caveat
    if caveat is None:
        caveat = (
            "Exploratory single-cell result; replicate and satisfy both environment-class gates before scoring."
            if args.run_index_status == "experimental"
            else ""
        )
    entry: dict[str, Any] = {
        "path": path,
        "status": args.run_index_status,
        "variant": variant,
        "comparison_group": comparison_group,
        "context_band": context_band,
        "max_output_tokens": max_output_tokens,
        "completion_state": completion_state,
        "caveat": caveat,
        "machine": metadata["machine"],
        "engine": metadata["engine"],
        "engine_version": metadata["engine_version"],
        "campaign_dimensions": envelope.get("campaign_dimensions", {}),
        "dimension_null_reasons": envelope.get("dimension_null_reasons", {}),
    }
    for field in ("engine_build", "engine_commit"):
        if metadata.get(field) is not None:
            entry[field] = metadata[field]
    return entry


def _index_integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _append_run_index_entry(path: Path, entry: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    payload = _load_run_index(path, text=text)
    if any(
        existing.get("path") == entry["path"] for existing in payload["runs"] if isinstance(existing, dict)
    ):
        return
    runs_key = text.find('"runs"')
    array_start = text.find("[", runs_key)
    if runs_key < 0 or array_start < 0:
        raise ValueError(f"{path} has no canonical runs array terminator")
    _runs, array_end = json.JSONDecoder().raw_decode(text, array_start)
    closing = array_end - 1
    prefix = text[:closing].rstrip()
    separator = ",\n    " if payload["runs"] else "\n    "
    updated = (
        prefix
        + separator
        + json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        + "\n  ]"
        + text[array_end:]
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        output.write(updated)
    temporary.replace(path)


def _load_run_index(path: Path, *, text: str | None = None) -> dict[str, Any]:
    payload = json.loads(text if text is not None else path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "run-index-v1"
        or not isinstance(payload.get("runs"), list)
        or not all(isinstance(entry, dict) for entry in payload.get("runs", []))
    ):
        raise ValueError(f"{path} is not a run-index-v1 document")
    return payload


def _load_campaign_dimensions(
    path: Path,
) -> tuple[
    CampaignDimensions,
    dict[str, str] | None,
    str | None,
    Literal["model_card", "inferred"] | None,
]:
    """Read typed campaign dimensions and optional model-task metadata from ``path``.

    Only the declared envelope keys are accepted. Field names and dimension value
    types are left to ``publish()``, which validates the finished envelope against
    ``schema.json``. Task provenance must be explicit when a task is supplied.
    ``json.JSONDecodeError`` is a ``ValueError``, so malformed JSON takes the same path.
    """
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    allowed_keys = {
        "campaign_dimensions",
        "dimension_null_reasons",
        "model_task",
        "model_task_source",
    }
    unexpected = sorted(set(payload) - allowed_keys)
    if unexpected:
        raise ValueError(f"unexpected keys: {', '.join(unexpected)}")
    dimensions = payload.get("campaign_dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("campaign_dimensions must be an object")
    reasons = payload.get("dimension_null_reasons")
    if reasons is not None and not isinstance(reasons, dict):
        raise ValueError("dimension_null_reasons must be an object")
    model_task = payload.get("model_task")
    model_task_source: Literal["model_card", "inferred"] | None = payload.get("model_task_source")
    if model_task is not None and (not isinstance(model_task, str) or not model_task.strip()):
        raise ValueError("model_task must be a non-empty string")
    if model_task_source is not None and (
        not isinstance(model_task_source, str) or model_task_source not in {"model_card", "inferred"}
    ):
        raise ValueError("model_task_source must be 'model_card' or 'inferred'")
    if (model_task is None) != (model_task_source is None):
        raise ValueError("model_task and model_task_source must be provided together")
    return (
        cast("CampaignDimensions", dimensions),
        cast("dict[str, str] | None", reasons),
        cast("str | None", model_task),
        cast("Literal['model_card', 'inferred'] | None", model_task_source),
    )


def _extract_model(raw: dict[str, Any] | list[Any]) -> str:
    if isinstance(raw, list):
        # bench-serve emits a list of per-run records, each carrying model_id.
        for run in raw:
            if isinstance(run, dict) and isinstance(run.get("model_id"), str) and run["model_id"]:
                return run["model_id"]
        return "unknown"
    cfg = raw.get("config") or {}
    model_args = cfg.get("model_args") or {}
    eval_header = raw.get("eval") or {}
    for candidate in (
        raw.get("model_name"),
        model_args.get("model"),
        raw.get("model"),
        eval_header.get("model") if isinstance(eval_header, dict) else None,
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown"


def _extract_model_revision(raw: dict[str, Any] | list[Any]) -> str | None:
    """Read the recorded Hub SHA from run output, never a publisher-time alias."""
    if isinstance(raw, dict):
        revision = raw.get("model_revision")
        return revision if isinstance(revision, str) else None
    revisions = {
        item["model_revision"]
        for item in raw
        if isinstance(item, dict) and isinstance(item.get("model_revision"), str)
    }
    return next(iter(revisions)) if len(revisions) == 1 else None


def _parse_kv_pairs(raw_pairs: list[str]) -> Iterator[tuple[str, str]]:
    for pair in raw_pairs:
        if "=" not in pair:
            raise SystemExit(f"invalid --tag {pair!r}; expected KEY=VALUE")
        key, _, value = pair.partition("=")
        if not key:
            raise SystemExit(f"invalid --tag {pair!r}; empty key")
        yield key, value


def _normalize_engine_name(value: str) -> str:
    return {"vllm": "vLLM", "mlx": "MLX"}.get(value.casefold(), value)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
