"""
MLX Benchmarks Viewer — Gradio Space

Reads all parquet shards from JacobPEvans/mlx-benchmarks and renders
interactive comparison charts. Auto-refreshes data every 10 minutes.

Deploy to HF Spaces (SDK: gradio, Python 3.11+).
"""

import json
import logging
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from threading import Lock, Thread

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import snapshot_download

DATASET = "datasets/JacobPEvans/mlx-benchmarks"
DATASET_REPO_ID = "JacobPEvans/mlx-benchmarks"
RUN_INDEX_PATH = "metadata/run-index-v1.json"
CACHE_TTL = 600  # seconds
AGENTIC_MODE = "Agentic & coding"
DIAGNOSTIC_MODE = "Speed / diagnostic"
VIEWER_MODES = (AGENTIC_MODE, DIAGNOSTIC_MODE)
DEFAULT_EVIDENCE = "scored"
MIN_AGENTIC_INPUT_TOKENS = 32768
MODEL_CONTEXTS = json.loads(Path(__file__).with_name("model-contexts.json").read_text())["models"]
EXPECTED_COLUMNS = ["timestamp", "suite", "name", "metric", "model", "value"]
FILTER_SPECS = {
    "machine": ("Machine", ("campaign_hardware_machine", "machine")),
    "host": ("Host", ("campaign_hardware_host", "host", "hostname")),
    "accelerator": (
        "Accelerator model",
        ("campaign_hardware_accelerator_model", "accelerator_model", "chip", "gpu"),
    ),
    "engine": ("Engine", ("campaign_software_engine", "engine")),
    "engine_version": (
        "Engine version",
        ("campaign_software_engine_version", "engine_version", "tag_engine_version"),
    ),
    "backend": ("Backend", ("campaign_software_backend", "backend", "tag_backend")),
    "quant_format": (
        "Quant format",
        ("campaign_model_quantization", "quant_format", "quantization", "tag_quant", "quant"),
    ),
    "reasoning_effort": (
        "Reasoning effort",
        ("campaign_run_reasoning_effort", "reasoning_effort", "tag_reasoning_effort", "reasoning_arm"),
    ),
    "thinking": (
        "Thinking arm",
        ("campaign_run_thinking", "thinking", "tag_thinking", "tag_think_value"),
    ),
    "mtp": (
        "MTP",
        ("campaign_run_speculative_mtp", "speculative_mtp", "tag_mtp", "mtp", "tag_speculative_decoding"),
    ),
    "speculative_tokens": (
        "Speculative tokens",
        (
            "campaign_run_num_speculative_tokens",
            "num_speculative_tokens",
            "tag_num_speculative_tokens",
        ),
    ),
    "kv_cache_dtype": (
        "KV cache dtype",
        ("campaign_run_kv_cache_dtype", "kv_cache_dtype", "tag_kv_cache_dtype"),
    ),
    "kv_offload_backend": (
        "KV offload backend",
        ("campaign_run_kv_offload_backend", "kv_offload_backend"),
    ),
    "moe_cpu_offload": (
        "MoE CPU offload",
        ("campaign_run_moe_cpu_offload", "moe_cpu_offload"),
    ),
    "moe_cpu_layers": (
        "MoE CPU layers",
        ("campaign_run_moe_cpu_layers", "moe_cpu_layers"),
    ),
    "kv_offload_cpu_gib": (
        "KV offload CPU (GiB)",
        ("campaign_run_kv_offload_cpu_gib", "kv_offload_cpu_gib"),
    ),
    "host_ram_limit_gib": (
        "Host RAM limit (GiB)",
        ("campaign_hardware_host_ram_limit_gib", "host_ram_limit_gib"),
    ),
    "prefix_caching": (
        "Prefix caching",
        ("campaign_run_prefix_cache", "prefix_cache", "prefix_caching"),
    ),
    "cache_workload": (
        "Cache workload",
        ("campaign_run_cache_workload", "cache_workload"),
    ),
    "power_limit_w": (
        "Power limit (W)",
        ("power_limit_w", "campaign_hardware_power_cap_w", "tag_power_limit_w"),
    ),
    "agents": (
        "Concurrency",
        ("campaign_run_concurrent_agents", "concurrent_agents", "concurrency", "tag_concurrency", "agents"),
    ),
    "context_band": ("Context band", ("context_band", "tag_context_band")),
    "allocated_context": (
        "Allocated context (tokens)",
        (
            "campaign_run_allocated_context_tokens",
            "allocated_context",
            "context_tokens_target",
            "tag_context_tokens_target",
            "configured_window_tokens",
            "tag_configured_window_tokens",
        ),
    ),
    "context_tokens": (
        "Context (tokens)",
        (
            "campaign_run_depth_tokens",
            "context_tokens_actual",
            "tag_context_tokens_actual",
            "context_tokens",
        ),
    ),
    "prompt_tokens": (
        "Prompt (tokens)",
        (
            "campaign_run_prompt_tokens",
            "prompt_tokens",
            "tag_prompt_tokens",
            "total_input_tokens",
            "tag_total_input_tokens",
            "input_tokens",
            "tag_input_tokens",
            "requested_prompt_tokens",
            "tag_requested_prompt_tokens",
        ),
    ),
    "output_budget_tokens": (
        "Output tokens",
        (
            "campaign_run_output_tokens",
            "max_output_tokens",
            "output_token_budget",
            "max_new_tokens",
            "max_gen_toks",
            "tag_max_gen_toks",
            "output_reservation_tokens",
            "tag_output_reservation_tokens",
            "reserved_output_tokens",
            "tag_reserved_output_tokens",
        ),
    ),
    "arrival_pattern": (
        "Arrival pattern",
        ("arrival", "arrival_pattern", "tag_arrival", "tag_arrival_pattern", "load_shape"),
    ),
    "request_rate": (
        "Request rate",
        ("request_rate", "tag_request_rate", "campaign_run_request_rate"),
    ),
    "burstiness": ("Burstiness", ("burstiness", "tag_burstiness", "campaign_run_burstiness")),
}
WORKLOAD_FIELDS = (
    "allocated_context",
    "context_tokens",
    "prompt_tokens",
    "output_budget_tokens",
    "completion_state",
)
PLACEHOLDERS = frozenset(
    {"", "n/a", "na", "nan", "none", "null", "unknown", "unspecified", "unindexed", "unstated"}
)
CSS = """
#title { color: #243746; margin-bottom: 2px; }
#subtitle { color: #647783; max-width: 760px; margin: 0 0 18px; line-height: 1.5; }
#selection-status { border-left: 3px solid #126b72; padding: 10px 14px; background: #e8f0f1; }
.gradio-container { background: #f2f5f6; }
.dark .gradio-container { background: var(--body-background-fill); }
.dark #selection-status { background: var(--background-fill-secondary); }
.gradio-container .js-plotly-plot,
.gradio-container .js-plotly-plot .svg-container { background: transparent !important; }
.gradio-container .js-plotly-plot text,
.gradio-container .js-plotly-plot .annotation-text { fill: var(--body-text-color, #243746) !important; }
body, .gradio-container { font-family: Inter, "Segoe UI", Arial, sans-serif; }
button:focus-visible, input:focus-visible, [role="combobox"]:focus-visible {
    outline: 3px solid #126b72 !important;
    outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
}
"""
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
)


# ── Data loading ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ViewerCache:
    created_at: float
    rows: pd.DataFrame
    modes: dict[str, pd.DataFrame]
    evidence_counts: dict[str, int]


_cache: ViewerCache | None = None
_cache_lock = Lock()
_refresh_lock = Lock()
LOGGER = logging.getLogger(__name__)


def empty_data() -> pd.DataFrame:
    columns = [
        *EXPECTED_COLUMNS,
        "unit",
        "display_unit",
        "model_short",
        "evidence_status",
        "series_key",
        "series_label",
        "workload_complete",
        "viewer_mode",
        "model_card_url",
        "native_context_tokens",
        "yarn_extended_context_tokens",
        "max_recommended_context_tokens",
        "largest_measured_input_tokens",
        "variant",
        "caveat",
        "fan_duty",
        "ambient_temp_f",
        "patches",
        "dedicated",
        *FILTER_SPECS,
        *WORKLOAD_FIELDS,
    ]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def _dataset_path(uri: str) -> str:
    """Return the original shard path represented by a canonical HF URI."""
    path = uri.removeprefix(f"hf://{DATASET}/")
    canonical_prefix = "data/run-canonical-"
    if path.startswith(canonical_prefix):
        return f"data/{path.removeprefix(canonical_prefix)}"
    return path


def _json_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _clean_value(value: object, preferred: tuple[str, ...] = ()) -> str | None:
    """Return a readable row value, omitting nulls, placeholders, and raw objects."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        for key in preferred:
            if key in value:
                return _clean_value(value[key], preferred)
        return None
    if isinstance(value, (list, tuple)):
        return _clean_value(value[0], preferred) if len(value) == 1 else None
    if isinstance(value, Real):
        number = float(value)
        return f"{int(number)}" if number.is_integer() else f"{number:g}"
    if not isinstance(value, str):
        return None

    text = value.strip()
    if text.casefold() in PLACEHOLDERS:
        return None
    if text.startswith(("{", "[")):
        return _clean_value(_json_mapping(text), preferred)
    return text


def _row_value(row: pd.Series, aliases: tuple[str, ...], preferred: tuple[str, ...] = ()) -> str | None:
    """Read a direct field or its optional tags_json projection without inventing values."""
    for alias in aliases:
        if alias in row.index:
            value = _clean_value(row[alias], preferred)
            if value is not None:
                return value

    for container_name in ("tags_json", "extra_json"):
        container = _json_mapping(row.get(container_name))
        if container is None:
            continue
        for alias in aliases:
            key = alias.removeprefix("tag_")
            value = _clean_value(container.get(key), preferred)
            if value is not None:
                return value
    return None


FILTER_ENUM_NONE_FIELDS = frozenset({"kv_offload_backend", "moe_cpu_offload", "cache_workload"})


def _filter_value(value: object, field: str) -> str | None:
    if field in FILTER_ENUM_NONE_FIELDS and isinstance(value, str) and value.strip().casefold() == "none":
        return "none"
    return _clean_value(value)


def _token_count(value: object) -> float | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    match = re.fullmatch(r"([\d,.]+)\s*([kKmM]?)", cleaned)
    if match is None:
        return None
    try:
        number = float(match[1].replace(",", ""))
    except ValueError:
        return None
    multiplier = {"k": 1024, "m": 1024 * 1024}.get(match[2].casefold(), 1)
    return number * multiplier


def _is_throughput_row(row: Mapping[str, object] | pd.Series) -> bool:
    suite = (_clean_value(row.get("suite")) or "").casefold().replace("_", "-")
    task = (_clean_value(row.get("name")) or "").casefold().replace("_", "-")
    metric = (_clean_value(row.get("metric")) or "").casefold().replace("_", "-")
    return (
        any(term in suite for term in ("throughput", "serving", "bench-serve"))
        or any(term in task for term in ("benchmark-serving", "throughput-probe"))
        or any(term in metric for term in ("throughput-", "tokens-per-second", "ttft-", "tpot-", "itl-"))
    )


def _throughput_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column, pattern in (
        ("suite", r"throughput|serving|bench-serve"),
        ("name", r"benchmark-serving|throughput-probe"),
        ("metric", r"throughput-|tokens-per-second|ttft-|tpot-|itl-"),
    ):
        if column in df:
            values = df[column].astype("string").str.casefold().str.replace("_", "-", regex=False)
            mask |= values.str.contains(pattern, regex=True, na=False)
    return mask


def classify_viewer_row(row: Mapping[str, object] | pd.Series) -> str:
    """Keep evaluations and long-input throughput in the decision view."""
    if not _is_throughput_row(row):
        return AGENTIC_MODE

    input_tokens = _token_count(row.get("prompt_tokens"))
    if input_tokens is None:
        input_tokens = _token_count(row.get("context_tokens"))
    return (
        AGENTIC_MODE
        if input_tokens is not None and input_tokens >= MIN_AGENTIC_INPUT_TOKENS
        else DIAGNOSTIC_MODE
    )


def mode_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if "viewer_mode" in df:
        return df[df["viewer_mode"] == mode].copy()
    return df[df.apply(classify_viewer_row, axis="columns") == mode].copy()


def default_context_band(df: pd.DataFrame) -> str | None:
    throughput_mask = (
        df["_is_throughput"].fillna(False).astype(bool) if "_is_throughput" in df else _throughput_mask(df)
    )
    agentic_mask = (
        df["viewer_mode"].eq(AGENTIC_MODE)
        if "viewer_mode" in df
        else df.apply(lambda row: classify_viewer_row(row) == AGENTIC_MODE, axis="columns")
    )
    throughput = df[throughput_mask & agentic_mask]
    candidates = [
        (count, band)
        for band in row_choices(throughput, "context_band")
        if (count := _token_count(band)) is not None and count >= MIN_AGENTIC_INPUT_TOKENS
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (abs(item[0] - 65536), item[0]))[1]


def default_filter_selections(df: pd.DataFrame, mode: str) -> dict[str, str | None]:
    selections: dict[str, str | None] = dict.fromkeys(FILTER_SPECS)
    if mode == AGENTIC_MODE:
        context_band = default_context_band(df)
        if context_band in row_choices(df, "context_band"):
            selections["context_band"] = context_band
    return selections


def current_filter_selections(df: pd.DataFrame, filter_values) -> dict[str, str | None]:
    """Drop selections that no row in this frame carries, such as another mode's context band."""
    selections = dict(zip(FILTER_SPECS, filter_values, strict=False))
    return {
        name: value if value is None or value in row_choices(df, name) else None
        for name, value in selections.items()
    }


def _metric_unit(metric: str, unit: object = None) -> str:
    """Normalize known units and infer only units encoded by the metric contract."""
    explicit = _clean_value(unit)
    if explicit is not None:
        normalized = explicit.casefold()
        return {
            "tokens/s": "tok/s",
            "token/s": "tok/s",
            "seconds": "s",
            "second": "s",
            "ratio": "%",
            "percent": "%",
            "x": "x",
            "round": "rounds",
        }.get(normalized, explicit)

    key = metric.casefold()
    if "requests_per_second" in key or "requests_per_s" in key:
        return "req/s"
    if any(part in key for part in ("tokens_per_second", "tok_s", "throughput")):
        return "tok/s"
    if key.endswith("_ms") or any(part in key for part in ("_p50_ms", "_p90_ms", "_p99_ms")):
        return "ms"
    if key.endswith("_s") or "_duration_" in key or "_latency_" in key:
        return "s"
    if any(part in key for part in ("accuracy", "_rate", "ratio", "pass_at")):
        return "%"
    if "sample_len" in key or "token" in key:
        return "tokens"
    return "unitless"


def _percentage_value(value: float) -> float:
    return value * 100 if abs(value) <= 1 else value


def _chart_value(value: object, unit: str) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return float("nan")
    number = float(number)
    return _percentage_value(number) if unit == "%" else number


def format_value(value: object, unit: str) -> str:
    """Format a measurement and its unit together for every user-facing value."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    number = float(number)
    if unit == "%":
        return f"{_percentage_value(number):.1f}%"
    if unit == "x":
        return f"{number:.2f}x"
    if unit == "tok/s":
        return f"{number:.1f} tok/s"
    if unit == "req/s":
        return f"{number:.2f} req/s"
    if unit in {"ms", "s"}:
        return f"{number:.2f} {unit}"
    if unit == "rounds":
        return f"{number:g} rounds"
    if unit == "tokens":
        return f"{number:,.0f} tokens" if number.is_integer() else f"{number:,.1f} tokens"
    if unit == "bool":
        return "true" if number >= 0.5 else "false"
    if number.is_integer():
        return f"{number:,.0f} {unit}"
    return f"{number:.3g} {unit}"


def humanize(value: object) -> str:
    text = str(value).replace("_", " ").replace("-", " ").strip()
    text = re.sub(r"\s+", " ", text)
    words = []
    for word in text.split():
        lowered = word.casefold()
        words.append(
            {
                "arc": "ARC",
                "gsm8k": "GSM8K",
                "itl": "ITL",
                "ttft": "TTFT",
                "tpot": "TPOT",
                "vllm": "vLLM",
            }.get(lowered, word.capitalize())
        )
    return " ".join(words)


def load_local_run_index(data_dir: Path) -> tuple[dict[str, dict[str, object]], pd.Timestamp | None]:
    """Read the optional run index beside a local or cached Parquet directory."""
    path = data_dir.parent / RUN_INDEX_PATH
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}, None
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return {}, None
    cutoff = payload.get("historical_scored_before")
    cutoff_timestamp = pd.to_datetime(cutoff, utc=True, errors="coerce") if cutoff else None
    return (
        {
            _dataset_path(f"hf://{DATASET}/{entry['path']}"): entry
            for entry in runs
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        },
        cutoff_timestamp if pd.notna(cutoff_timestamp) else None,
    )


RUN_INDEX_PROJECTIONS = {
    "machine": ("machine", "hostname", "campaign_hardware_machine"),
    "engine": ("engine", "campaign_software_engine"),
    "engine_version": ("engine_version", "campaign_software_engine_version"),
    "engine_build": ("campaign_software_engine_build",),
    "engine_commit": ("campaign_software_engine_commit",),
}


def apply_run_index_overrides(df: pd.DataFrame, index: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Project safe per-shard labels and engine metadata without changing Parquet files."""
    if df.empty or "source_path" not in df:
        return df
    df = df.copy()
    for source_path, entry in index.items():
        mask = df["source_path"].eq(source_path)
        if not mask.any():
            continue
        for field, columns in RUN_INDEX_PROJECTIONS.items():
            if field not in entry:
                continue
            for column in columns:
                if column not in df:
                    df[column] = pd.Series(pd.NA, index=df.index, dtype="object")
                elif not pd.api.types.is_object_dtype(df[column].dtype):
                    df[column] = df[column].astype("object")
                df.loc[mask, column] = entry[field]
    return df


MACHINE_LABELS = {
    "nvidia rtx pro 6000 blackwell max-q": "RTX PRO 6000 Max-Q 96GB",
    "nvidia rtx pro 6000 blackwell max-q 96gb": "RTX PRO 6000 Max-Q 96GB",
    "rtx pro 6000 96gb": "RTX PRO 6000 Max-Q 96GB",
    "rtx pro 6000 max-q 96gb": "RTX PRO 6000 Max-Q 96GB",
    "rtx 4080 16gb": "RTX 4080 SUPER 16GB",
    "rtx 4080 super 16gb": "RTX 4080 SUPER 16GB",
    "apple m4 max": "Mac Studio M4 Max 128GB",
    "mac studio": "Mac Studio M4 Max 128GB",
    "mac studio m4 max 128gb": "Mac Studio M4 Max 128GB",
    "2x apple m4 max 128gb (thunderbolt rdma)": "2x Apple M4 Max 128GB (Thunderbolt RDMA)",
}
MACHINE_SHORT_LABELS = {"2x Apple M4 Max 128GB (Thunderbolt RDMA)": "2x Apple M4 Max 128GB · TB5 RDMA"}


def normalize_machine(value: object) -> str | None:
    """Normalize known machine-label variants to the public display names."""
    cleaned = _clean_value(value)
    return MACHINE_LABELS.get(cleaned.casefold(), cleaned) if cleaned else None


def _arrival_pattern(row: pd.Series) -> str | None:
    arrival = _row_value(row, FILTER_SPECS["arrival_pattern"][1])
    normalized = arrival.casefold() if arrival else ""
    if "batch" in normalized or normalized in {"inf", "infinite"}:
        return "Batched"
    if "poisson" in normalized or "stagger" in normalized:
        return "Staggered"
    request_rate = _row_value(row, FILTER_SPECS["request_rate"][1])
    if request_rate is None:
        # Older rows record the arrival only in the run-index variant ("... batched", "... Poisson 0.5 rps").
        variant = (_clean_value(row.get("variant")) or "").casefold()
        if "batched" in variant:
            return "Batched"
        if "poisson" in variant or "stagger" in variant:
            return "Staggered"
        return None
    return "Batched" if request_rate.casefold() in {"inf", "infinite"} else "Staggered"


def _json_text(row: pd.Series, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias not in row.index:
            continue
        value = row[alias]
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return value.strip() or None
        if isinstance(value, (Mapping, list, tuple)):
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        return _clean_value(value)
    return None


def add_evidence_metadata(
    df: pd.DataFrame, index: dict[str, dict[str, object]], historical_cutoff: pd.Timestamp | None
) -> pd.DataFrame:
    """Attach evidence and row-backed dimensions used to compare benchmark series."""
    df = df.copy()

    def run_entry(row: pd.Series) -> dict[str, object]:
        path = _clean_value(row.get("source_path"))
        return index.get(path or "", {})

    def evidence_status(row: pd.Series) -> str:
        entry = run_entry(row)
        if entry:
            return str(entry.get("status", "experimental")).casefold()
        if historical_cutoff is not None and row["timestamp"] < historical_cutoff:
            return "scored"
        return "experimental"

    df["evidence_status"] = df.apply(evidence_status, axis="columns")
    df["variant"] = df.apply(lambda row: _clean_value(run_entry(row).get("variant")), axis="columns")
    df["caveat"] = df.apply(lambda row: _clean_value(run_entry(row).get("caveat")), axis="columns")
    df["context_band"] = df.apply(
        lambda row: (
            _clean_value(run_entry(row).get("context_band"))
            or _row_value(row, FILTER_SPECS["context_band"][1])
        ),
        axis="columns",
    )

    preferred = {
        "accelerator": ("model", "name", "chip", "accelerator"),
        "engine": ("engine", "name", "runtime"),
        "backend": ("backend", "name"),
        "quant_format": ("format", "quantization", "quant"),
    }

    def dimension_value(row: pd.Series, name: str, aliases: tuple[str, ...]) -> str | None:
        if name in FILTER_ENUM_NONE_FIELDS:
            for alias in aliases:
                if alias in row.index and (value := _filter_value(row[alias], name)) is not None:
                    return value
        value = _row_value(row, aliases, preferred.get(name, ()))
        if value is not None or name not in {"engine", "backend"}:
            return value
        serving = _json_mapping(row.get("serving"))
        return _clean_value(serving.get(name), preferred.get(name, ())) if serving else None

    for name, (_, aliases) in FILTER_SPECS.items():
        if name == "arrival_pattern":
            df[name] = df.apply(_arrival_pattern, axis="columns")
        elif name == "machine":

            def machine_value(row: pd.Series, keys: tuple[str, ...] = aliases) -> str | None:
                value = dimension_value(row, "machine", keys)
                if value is None:
                    candidate = _row_value(
                        row,
                        ("campaign_hardware_accelerator_model", "accelerator_model", "chip", "accelerator"),
                    )
                    if candidate and candidate.casefold() in MACHINE_LABELS:
                        value = candidate
                return normalize_machine(value)

            df[name] = df.apply(machine_value, axis="columns")
        elif name == "context_band":
            df[name] = [
                _clean_value(existing) or dimension_value(row, name, aliases)
                for existing, (_, row) in zip(df[name], df.iterrows(), strict=False)
            ]
        elif name == "thinking":

            def thinking_value(row: pd.Series, keys: tuple[str, ...] = aliases) -> str | None:
                value = dimension_value(row, "thinking", keys)
                return {"true": "on", "false": "off"}.get(str(value).casefold(), value)

            df[name] = df.apply(thinking_value, axis="columns")
        else:
            df[name] = df.apply(
                lambda row, key=name, keys=aliases: dimension_value(row, key, keys),
                axis="columns",
            )
    df["output_budget_tokens"] = [
        value if value is not None else _clean_value(run_entry(row).get("max_output_tokens"))
        for (_, row), value in zip(
            df.iterrows(),
            df["output_budget_tokens"],
            strict=False,
        )
    ]

    def completion(row: pd.Series) -> str | None:
        value = _row_value(
            row,
            ("completion_state", "tag_completion_state", "finish_reason", "tag_finish_reasons"),
            ("state", "status", "finish_reason", "reason"),
        )
        if value is not None:
            return value
        truncated = _row_value(row, ("truncated_rate", "tag_truncated_rate"))
        if truncated is not None:
            return "complete" if float(truncated) == 0 else "truncated"
        return _clean_value(run_entry(row).get("completion_state"))

    df["completion_state"] = df.apply(completion, axis="columns")
    df["fan_duty"] = df.apply(
        lambda row: _row_value(
            row,
            (
                "campaign_environment_fan_duty_pct_json",
                "campaign_resource_fan_percent",
                "fan_duty_pct",
                "tag_fan_duty_pct",
            ),
            ("gpu", "chassis", "mac"),
        ),
        axis="columns",
    )
    df["ambient_temp_f"] = df.apply(
        lambda row: _row_value(
            row,
            ("campaign_environment_ambient_temp_f", "ambient_temp_f", "tag_ambient_temp_f"),
        ),
        axis="columns",
    )
    df["patches"] = df.apply(
        lambda row: _json_text(
            row,
            ("campaign_run_patches_json", "patches_json", "patches", "tag_patches"),
        ),
        axis="columns",
    )
    df["dedicated"] = df.apply(
        lambda row: _row_value(row, ("dedicated", "tag_dedicated", "campaign_run_dedicated")),
        axis="columns",
    )
    df["caveat"] = [
        caveat or _row_value(row, ("caveat", "tag_caveat", "tag_caveats"))
        for caveat, (_, row) in zip(df["caveat"], df.iterrows(), strict=False)
    ]
    df["display_unit"] = [
        _metric_unit(str(metric), unit)
        for metric, unit in zip(
            df.get("metric", pd.Series("", index=df.index)),
            df.get("unit", pd.Series(None, index=df.index)),
            strict=False,
        )
    ]
    df["model_short"] = df["model"].map(
        lambda value: short_model(str(value)) if _clean_value(value) is not None else ""
    )
    df["model_card_url"] = df["model"].map(
        lambda value: (MODEL_CONTEXTS.get(_clean_value(value) or "") or {}).get("card_url")
    )
    df["native_context_tokens"] = df["model"].map(
        lambda value: (MODEL_CONTEXTS.get(_clean_value(value) or "") or {}).get("native_context_tokens")
    )
    df["yarn_extended_context_tokens"] = df["model"].map(
        lambda value: (MODEL_CONTEXTS.get(_clean_value(value) or "") or {}).get(
            "yarn_extended_context_tokens"
        )
    )
    df["max_recommended_context_tokens"] = df["model"].map(
        lambda value: (MODEL_CONTEXTS.get(_clean_value(value) or "") or {}).get(
            "max_recommended_context_tokens"
        )
    )
    input_tokens = pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(
        pd.to_numeric(df["context_tokens"], errors="coerce")
    )
    context_rows = pd.DataFrame(
        {
            "model": df["model"].map(_clean_value),
            "machine": df.apply(_context_machine, axis="columns"),
            "input_tokens": input_tokens,
        },
        index=df.index,
    )
    df["largest_measured_input_tokens"] = context_rows.groupby(["model", "machine"], dropna=False)[
        "input_tokens"
    ].transform("max")
    df["viewer_mode"] = df.apply(classify_viewer_row, axis="columns")
    df["workload_complete"] = df.loc[:, list(WORKLOAD_FIELDS)].notna().all(axis="columns")
    efforts = df.get("reasoning_effort", pd.Series(None, index=df.index)).map(_clean_value).fillna("unstated")
    variants = df["variant"].map(_clean_value)
    # Cloud and local records may share a model name; the declared kind,
    # provider, and provider-qualified id keep their series apart. Shards
    # published before the columns existed read them as null and keep keys.
    identity_fields = (
        "model",
        "model_kind",
        "model_provider",
        "model_id",
        "machine",
        "host",
        "accelerator",
        "engine",
        "backend",
        "engine_version",
        "quant_format",
        "reasoning_effort",
        "thinking",
        "mtp",
        "speculative_tokens",
        "kv_cache_dtype",
        "power_limit_w",
        "agents",
        "context_band",
        "allocated_context",
        "context_tokens",
        "prompt_tokens",
        "output_budget_tokens",
        "arrival_pattern",
        "request_rate",
        "burstiness",
        "completion_state",
        "display_unit",
    )

    def comparison_key(row: pd.Series) -> str:
        values = [row.get(field) for field in identity_fields]
        values.append(variants.loc[row.name])
        joined = "|".join("" if _clean_value(value) is None else str(value) for value in values)
        return f"{joined}|{efforts.loc[row.name] or 'unstated'}"

    df["series_key"] = df.apply(comparison_key, axis="columns")

    def series_label(row: pd.Series) -> str:
        model = _clean_value(row.get("model_short"))
        machine = (
            _clean_value(row.get("machine"))
            or _clean_value(row.get("host"))
            or _clean_value(row.get("accelerator"))
        )
        engine = _clean_value(row.get("engine"))
        return (
            " / ".join(dict.fromkeys(value for value in (model, machine, engine) if value)) or "Benchmark run"
        )

    df["series_label"] = df.apply(series_label, axis="columns")
    return df


ITEM_NAME = re.compile(r"test_\d+(?:_assertion_\d+)?\Z", re.IGNORECASE)
AGGREGATE_METRICS = frozenset(
    {"accuracy", "exact_match", "pass_at_1_mean", "pass_at_1_rate", "pass_rate", "score"}
)
THROUGHPUT_OUTPUT_METRIC = "throughput_output_toks_per_s"
BAR_LABEL_LIMIT = 150
LABEL_KNOBS = (
    ("agents", "c"),
    ("context_tokens", "context"),
    ("prompt_tokens", "prompt"),
    ("output_budget_tokens", "output"),
    ("arrival_pattern", "arrival"),
    ("request_rate", "request"),
    ("burstiness", "burst"),
    ("context_band", "context"),
    ("allocated_context", "allocated"),
    ("reasoning_effort", "reasoning"),
    ("thinking", "thinking"),
    ("mtp", "MTP"),
    ("speculative_tokens", "spec"),
    ("power_limit_w", "power"),
    ("backend", "backend"),
    ("quant_format", "quant"),
    ("kv_cache_dtype", "KV"),
)
DIAGNOSTIC_GROUP_FIELDS = (
    "model",
    "machine",
    "engine_version",
    "agents",
    "prompt_tokens",
    "output_budget_tokens",
)


@dataclass
class SeriesLabelEntry:
    key: str
    row: pd.Series
    base: str
    parts: list[str]


def _label_value(value: object) -> str | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    text = re.sub(r"\s+", " ", cleaned)
    return f"{text[:27]}…" if len(text) > 28 else text


def comparable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid evidence and per-item measurements before exposing tasks or charts."""
    if df.empty:
        return df.copy()
    rows = df
    if "evidence_status" in rows:
        rows = rows[rows["evidence_status"].astype("string").str.casefold() != "invalid"]
    if "name" in rows:
        item_names = rows["name"].astype("string").str.fullmatch(ITEM_NAME, na=False)
        rows = rows[~item_names]

    if {"metric", "suite", "name"} <= set(rows.columns):
        metrics = rows["metric"].astype("string").str.casefold()
        item_pass = metrics == "pass_at_1"
        aggregates = metrics.isin(AGGREGATE_METRICS)
        keys = [column for column in ("source_path", "suite", "name", "model") if column in rows]
        if item_pass.any() and aggregates.any() and keys:
            aggregate_keys = pd.MultiIndex.from_frame(rows.loc[aggregates, keys].fillna(""))
            item_keys = pd.MultiIndex.from_frame(rows.loc[item_pass, keys].fillna(""))
            rows = rows.drop(index=rows.loc[item_pass].index[item_keys.isin(aggregate_keys)])
    return rows.copy()


def evidence_view(df: pd.DataFrame, status: str = "scored") -> pd.DataFrame:
    """Return only results appropriate for the selected evidence view."""
    return _evidence_rows(comparable_rows(df), status)


def _evidence_rows(rows: pd.DataFrame, status: str) -> pd.DataFrame:
    normalized = status.casefold()
    aliases = {
        "scored evidence": "scored",
        "experimental / excluded": "experimental",
        "experimental": "experimental",
        "recovered": "recovered",
    }
    if normalized == "invalid":
        return rows.iloc[0:0].copy()
    if normalized in {"all", "all valid", "all-valid"}:
        if "evidence_status" not in rows:
            return rows
        valid_statuses = {"scored", "experimental", "recovered"}
        return rows[rows["evidence_status"].astype("string").str.casefold().isin(valid_statuses)]
    if "evidence_status" not in rows:
        return rows if normalized == "scored" else rows.iloc[0:0].copy()
    return rows[
        rows["evidence_status"].astype("string").str.casefold() == aliases.get(normalized, normalized)
    ]


def _prepare_cache(df: pd.DataFrame) -> ViewerCache:
    """Build reusable mode frames and whole-dataset evidence counts for one load."""
    df = df.copy()
    # Every interaction rebuilds the filter menus, so normalize these fields once per refresh.
    for name in FILTER_SPECS:
        if name in df and pd.api.types.infer_dtype(df[name], skipna=True) not in {"string", "empty"}:
            df[name] = df[name].map(lambda value, field=name: _filter_value(value, field))
    df.attrs["_viewer_filter_values_normalized"] = True
    df["_is_throughput"] = _throughput_mask(df)
    counts = (
        df["evidence_status"].astype("string").str.casefold().value_counts().to_dict()
        if "evidence_status" in df
        else {}
    )
    modes = {mode: mode_rows(df, mode) for mode in VIEWER_MODES}
    return ViewerCache(time.time(), df, modes, {str(key): int(value) for key, value in counts.items()})


def _current_cache() -> ViewerCache:
    load_data()
    with _cache_lock:
        if _cache is None:
            raise RuntimeError("Benchmark data cache was not initialized")
        return _cache


def cached_viewer_rows(mode: str, evidence: str, cache: ViewerCache | None = None) -> pd.DataFrame:
    """Select from the per-load mode frame without reopening any Parquet shard."""
    cache = cache or _current_cache()
    rows = cache.modes.get(mode, empty_data())
    return _evidence_rows(rows, evidence).copy()


def evidence_summary(counts: Mapping[str, int]) -> str:
    totals = "; ".join(
        f"{label}: {int(counts.get(key, 0)):,} rows"
        for key, label in (("scored", "Scored"), ("experimental", "Experimental"), ("recovered", "Recovered"))
    )
    return f"Evidence in this dataset: {totals}."


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce the two historical result layouts and drop non-measurements.

    Two publisher generations flattened results differently. Newer shards write
    ``name`` / ``metric`` / ``value`` / ``unit`` directly; older shards nested
    each result's metric object, so pandas exploded it into
    ``metric_name`` / ``metric_metric`` / ``metric_value`` / ``metric_unit``.
    The viewer only reads the flat columns, so without coalescing here it
    silently ignores most real measurements (e.g. tool-calling, ttft,
    code-accuracy, math-hard, and older throughput runs).

    Rows that were skipped (CI runs with no MLX server) or carry no numeric
    value are failure records, not comparable results — drop them so a suite
    only appears when it actually has data to chart.
    """
    for flat, nested in (
        ("name", "metric_name"),
        ("metric", "metric_metric"),
        ("value", "metric_value"),
        ("unit", "metric_unit"),
    ):
        if flat not in df.columns:
            df[flat] = pd.NA
        if nested in df.columns:
            df[flat] = df[flat].fillna(df[nested])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "skipped" in df.columns:
        df = df[~df["skipped"].fillna(False).astype(bool)]
    return df.dropna(subset=["name", "metric", "value"]).reset_index(drop=True)


def load_data() -> pd.DataFrame:
    global _cache
    with _cache_lock:
        cached = _cache
    if cached is not None:
        if time.time() - cached.created_at >= CACHE_TTL:
            _start_background_refresh()
        return cached.rows

    # The first request has no prior frame to serve, so it loads synchronously.
    with _refresh_lock:
        with _cache_lock:
            if _cache is not None:
                return _cache.rows
        df = _load_dataset()
        cache = _prepare_cache(df)
        with _cache_lock:
            _cache = cache
        return df


def _load_dataset() -> pd.DataFrame:
    local_data_dir = (
        Path(os.environ["MLX_BENCHMARKS_LOCAL_DATA_DIR"]).expanduser()
        if os.getenv("MLX_BENCHMARKS_LOCAL_DATA_DIR")
        else None
    )
    if local_data_dir is None:
        snapshot_dir = Path(
            snapshot_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                token=False,
                allow_patterns=["data/run-canonical-*.parquet", RUN_INDEX_PATH],
            )
        )
        data_dir = snapshot_dir / "data"
    else:
        data_dir = local_data_dir

    paths = sorted(data_dir.glob("run-canonical-*.parquet"))
    if not paths:
        return empty_data()

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_path"] = _dataset_path(f"hf://{DATASET}/data/{path.name}")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    raw_suites = set(df["suite"].dropna().unique())
    run_index, historical_cutoff = load_local_run_index(data_dir)
    df = apply_run_index_overrides(df, run_index)
    df = normalize_rows(df)
    df = comparable_rows(add_evidence_metadata(df, run_index, historical_cutoff))
    df.attrs["awaiting_suites"] = sorted(raw_suites - set(df["suite"].dropna().unique()))
    return df


def _refresh_cache() -> None:
    global _cache
    try:
        cache = _prepare_cache(_load_dataset())
    except Exception:
        LOGGER.exception("Background benchmark dataset refresh failed; serving the cached frame")
    else:
        with _cache_lock:
            _cache = cache
    finally:
        _refresh_lock.release()


def _start_background_refresh() -> None:
    if _refresh_lock.acquire(blocking=False):
        Thread(target=_refresh_cache, daemon=True).start()


def request_data_refresh() -> pd.DataFrame:
    """Refresh cached rows in the background and return the current frame immediately."""
    with _cache_lock:
        cached = _cache
    if cached is None:
        return load_data()
    _start_background_refresh()
    return cached.rows


def short_model(name: str | None) -> str:
    """Strip common prefixes for axis labels."""
    if not isinstance(name, str) or not name.strip():
        return "unknown model"
    name = re.sub(r"^mlx-community/", "", name)
    name = re.sub(r"^openrouter/openai/", "openrouter/", name)
    return name


def short_machine(name: str) -> str:
    """Use compact chart labels for long canonical machine names."""
    return MACHINE_SHORT_LABELS.get(name, name)


def row_choices(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df:
        return []
    if df.attrs.get("_viewer_filter_values_normalized") and column in FILTER_SPECS:
        values = set(df[column].dropna().unique())
    else:
        values = {value for item in df[column] if (value := _filter_value(item, column)) is not None}
    return sorted(values, key=str.casefold)


def filter_rows(df: pd.DataFrame, selections: dict[str, str | None]) -> pd.DataFrame:
    filtered = df
    for column, selection in selections.items():
        selected = _filter_value(selection, column)
        if selected is not None and column in filtered:
            if filtered.attrs.get("_viewer_filter_values_normalized") and column in FILTER_SPECS:
                mask = filtered[column].eq(selected).fillna(False)
            else:
                mask = (
                    filtered[column].map(lambda value, field=column: _filter_value(value, field)) == selected
                )
            filtered = filtered[mask]
    return filtered


def available_filter_choices(df: pd.DataFrame, column: str, selections: dict[str, str | None]) -> list[str]:
    other_filters = {name: value for name, value in selections.items() if name != column}
    return row_choices(filter_rows(df, other_filters), column)


def filter_viewer_rows(df: pd.DataFrame, selections: dict[str, str | None], mode: str) -> pd.DataFrame:
    context_band = selections.get("context_band")
    if mode != AGENTIC_MODE or context_band is None or "context_band" not in df:
        return filter_rows(df, selections)
    without_context = {name: value for name, value in selections.items() if name != "context_band"}
    filtered = filter_rows(df, without_context)
    throughput = (
        filtered["_is_throughput"].fillna(False).astype(bool)
        if "_is_throughput" in filtered
        else _throughput_mask(filtered)
    )
    in_band = filtered["context_band"].astype("string").eq(context_band).fillna(False)
    return filtered[~throughput | in_band].copy()


def available_viewer_filter_choices(
    df: pd.DataFrame, column: str, selections: dict[str, str | None], mode: str
) -> list[str]:
    other_filters = {name: value for name, value in selections.items() if name != column}
    return row_choices(filter_viewer_rows(df, other_filters, mode), column)


def available_viewer_filter_choices_for_all(
    df: pd.DataFrame, selections: dict[str, str | None], mode: str
) -> dict[str, list[str]]:
    """Build cascading choices from shared row masks instead of refiltering the frame per menu."""
    names = tuple(FILTER_SPECS)
    all_rows = pd.Series(True, index=df.index)
    masks = []
    for name in names:
        selected = _filter_value(selections.get(name), name)
        if selected is None or name not in df:
            masks.append(all_rows)
        elif name == "context_band" and mode == AGENTIC_MODE:
            throughput = (
                df["_is_throughput"].fillna(False).astype(bool)
                if "_is_throughput" in df
                else _throughput_mask(df)
            )
            in_band = df[name].astype("string").eq(selected).fillna(False)
            masks.append(~throughput | in_band)
        elif df.attrs.get("_viewer_filter_values_normalized") and name in FILTER_SPECS:
            masks.append(df[name].eq(selected).fillna(False))
        else:
            masks.append(
                df[name]
                .map(lambda value, field=name, choice=selected: _filter_value(value, field) == choice)
                .fillna(False)
            )

    prefix = [all_rows]
    for mask in masks:
        prefix.append(prefix[-1] & mask)
    suffix = [all_rows] * (len(names) + 1)
    for index in range(len(names) - 1, -1, -1):
        suffix[index] = suffix[index + 1] & masks[index]

    return {
        name: row_choices(df.loc[prefix[index] & suffix[index + 1], [name]], name) if name in df else []
        for index, name in enumerate(names)
    }


def suite_choices(df: pd.DataFrame) -> list[str]:
    rows = comparable_rows(df)
    if rows.empty:
        return []
    series_column = "series_key" if "series_key" in rows else "model"
    counts = rows.groupby(["suite", "name", "metric"], dropna=False)[series_column].nunique()
    suites = set(counts[counts >= 2].index.get_level_values("suite"))
    primary = rows[(rows["suite"] == "throughput") & (rows["metric"] == THROUGHPUT_OUTPUT_METRIC)]
    if not primary.empty:
        suites.add("throughput")
    return sorted(suites, key=str.casefold)


def task_choices(df: pd.DataFrame, suite: str | None) -> list[str]:
    if suite is None or df.empty:
        return []
    return row_choices(comparable_rows(df[df["suite"] == suite]), "name")


def metric_choices(df: pd.DataFrame, suite: str | None, task: str | None) -> list[str]:
    if suite is None or task is None or df.empty:
        return []
    matching = comparable_rows(df[(df["suite"] == suite) & (df["name"] == task)])
    return row_choices(matching, "metric")


def unit_choices(df: pd.DataFrame, suite: str | None, task: str | None, metric: str | None) -> list[str]:
    if suite is None or task is None or metric is None or df.empty:
        return []
    matching = comparable_rows(df[(df["suite"] == suite) & (df["name"] == task) & (df["metric"] == metric)])
    return row_choices(matching, "display_unit")


def valid_triples(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    if df.empty:
        return []
    return sorted(
        {
            (str(row.suite), str(row.name), str(row.metric))
            for row in df[["suite", "name", "metric"]].dropna().itertuples(index=False)
        }
    )


def top_task_metric(df: pd.DataFrame, suite: str) -> tuple[str | None, str | None]:
    """Choose the populated task and metric with the most distinct models."""
    sub = df[df["suite"] == suite]
    if sub.empty:
        return (None, None)
    counts = sub.groupby(["name", "metric"])["model"].nunique()
    name, metric = counts.idxmax()
    return (str(name), str(metric))


def top_metric(df: pd.DataFrame, suite: str, task: str) -> str | None:
    sub = df[(df["suite"] == suite) & (df["name"] == task)]
    if sub.empty:
        return None
    if suite == "throughput" and THROUGHPUT_OUTPUT_METRIC in sub["metric"].values:
        return THROUGHPUT_OUTPUT_METRIC
    return str(sub.groupby("metric")["model"].nunique().idxmax())


def best_default(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Prefer the scored output-throughput task with the most rows in this view."""
    rows = comparable_rows(df)
    if "evidence_status" in rows:
        rows = rows[rows["evidence_status"].astype("string").str.casefold() == "scored"]
    if rows.empty:
        return (None, None, None)

    primary = rows[(rows["suite"] == "throughput") & (rows["metric"] == THROUGHPUT_OUTPUT_METRIC)]
    if not primary.empty:
        counts = primary.groupby("name", dropna=True).size()
        task = str(counts.sort_values(ascending=False, kind="stable").index[0])
        return ("throughput", task, THROUGHPUT_OUTPUT_METRIC)

    if "machine" not in rows:
        return (None, None, None)

    candidates = []
    for (suite, task), task_group in rows.groupby(["suite", "name"], dropna=True):
        if suite == "throughput" and THROUGHPUT_OUTPUT_METRIC in task_group["metric"].values:
            task_group = task_group[task_group["metric"] == THROUGHPUT_OUTPUT_METRIC]
        for metric, group in task_group.groupby("metric", dropna=True):
            if group["machine"].dropna().nunique() < 2:
                continue
            values = pd.to_numeric(group["value"], errors="coerce").dropna()
            if values.nunique() < 2:
                continue
            latest = pd.to_datetime(group["timestamp"], utc=True, errors="coerce").max()
            if pd.isna(latest):
                continue
            candidates.append((latest, str(suite), str(task), str(metric)))
    if not candidates:
        return (None, None, None)
    candidates.sort(key=lambda item: (-item[0].value, item[1], item[2], item[3]))
    return candidates[0][1:]


def format_tokens(value: object) -> str | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return cleaned
    return f"{number:,.0f}" if number.is_integer() else f"{number:,.1f}"


def model_context_values(model: object) -> tuple[str, str, str, str] | None:
    context = MODEL_CONTEXTS.get(_clean_value(model) or "")
    if not context:
        return None
    native = format_tokens(context.get("native_context_tokens"))
    extended = format_tokens(context.get("yarn_extended_context_tokens"))
    recommended = format_tokens(context.get("max_recommended_context_tokens"))
    return (
        f"{native} tokens" if native else "",
        f"{extended} tokens" if extended else "Not documented",
        f"{recommended} tokens" if recommended else "",
        str(context.get("card_url") or ""),
    )


def workload_details(row: pd.Series) -> str:
    details = []
    for name, label in (
        ("allocated_context", "Allocated context"),
        ("context_tokens", "Context"),
        ("prompt_tokens", "Prompt"),
        ("output_budget_tokens", "Output budget"),
    ):
        value = format_tokens(row.get(name))
        if value is not None:
            details.append(f"{label}: {value} tokens")
    agents = _clean_value(row.get("agents"))
    if agents is not None:
        details.append(f"Agents: {agents}")
    completion = _clean_value(row.get("completion_state"))
    if completion is not None:
        details.append(f"Completion: {completion}")
    context = model_context_values(row.get("model"))
    if context:
        details.extend(
            (
                f"Native model context: {context[0]}",
                f"YaRN-extended context: {context[1]}",
                f"Max recommended context: {context[2]}",
            )
        )
        if context[3]:
            details.append(f'Model card: <a href="{context[3]}" target="_blank">Hugging Face</a>')
    largest_input = format_tokens(row.get("largest_measured_input_tokens"))
    if largest_input:
        details.append(f"Largest input measured for this model and machine: {largest_input} tokens")
    for field, label in (
        ("backend", "Backend"),
        ("quant_format", "Quant"),
        ("reasoning_effort", "Reasoning"),
        ("thinking", "Thinking"),
        ("mtp", "MTP"),
        ("speculative_tokens", "Speculative tokens"),
        ("kv_cache_dtype", "KV cache"),
        ("engine_version", "Engine version"),
        ("power_limit_w", "Power limit (W)"),
        ("context_band", "Context band"),
        ("arrival_pattern", "Arrival"),
        ("request_rate", "Request rate"),
        ("burstiness", "Burstiness"),
        ("variant", "Variant"),
    ):
        value = _clean_value(row.get(field))
        if value is not None:
            details.append(f"{label}: {value}")
    if not row.get("workload_complete", False):
        details.append("Workload details incomplete")
    return "; ".join(details) or "Workload details were not recorded"


def _distinct_variant_tails(variants: Mapping[str, str]) -> dict[str, str]:
    """Drop the words every variant shares at the start, e.g. "vLLM 0.31.0 128kx11 MTP-on batched" -> "batched"."""
    words = {key: value.split() for key, value in variants.items()}
    shared = 0
    for column in zip(*words.values(), strict=False):
        if len(set(column)) > 1:
            break
        shared += 1
    return {key: " ".join(parts[shared:]) or variants[key] for key, parts in words.items()}


def series_labels(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {}
    latest = df.sort_values("timestamp").drop_duplicates("series_key", keep="last")
    entries: list[SeriesLabelEntry] = []
    for _, row in latest.iterrows():
        model = _clean_value(row.get("model_short")) or _clean_value(row.get("model")) or "Benchmark run"
        machine = (
            _clean_value(row.get("machine")) or _clean_value(row.get("accelerator")) or "machine not recorded"
        )
        machine = short_machine(machine)
        engine = " ".join(
            value
            for value in (_clean_value(row.get("engine")), _clean_value(row.get("engine_version")))
            if value
        )
        base = " · ".join((short_model(model), machine, engine) if engine else (short_model(model), machine))
        entries.append(SeriesLabelEntry(str(row["series_key"]), row, base, []))

    groups: dict[str, list[SeriesLabelEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.base, []).append(entry)
    for group in groups.values():
        if len(group) < 2:
            continue
        varying = [
            (field, label)
            for field, label in LABEL_KNOBS
            if len({_clean_value(entry.row.get(field)) for entry in group}) > 1
        ]
        for entry in group:
            for field, label in varying[:3]:
                if (value := _label_value(entry.row.get(field))) is not None:
                    part = f"{label} {value}"
                    if len(_render_series_label(entry) + " · " + part) <= BAR_LABEL_LIMIT:
                        entry.parts.append(part)

        for field, label in varying[3:]:
            labels_by_text: dict[str, list[SeriesLabelEntry]] = {}
            for entry in group:
                labels_by_text.setdefault(_render_series_label(entry), []).append(entry)
            for duplicates in labels_by_text.values():
                if (
                    len(duplicates) < 2
                    or len({_clean_value(entry.row.get(field)) for entry in duplicates}) < 2
                ):
                    continue
                for entry in duplicates:
                    if len(entry.parts) < 3 and (value := _label_value(entry.row.get(field))) is not None:
                        part = f"{label} {value}"
                        if len(_render_series_label(entry) + " · " + part) <= BAR_LABEL_LIMIT:
                            entry.parts.append(part)

    duplicate_labels: dict[str, list[SeriesLabelEntry]] = {}
    for entry in entries:
        duplicate_labels.setdefault(_render_series_label(entry), []).append(entry)
    for group in duplicate_labels.values():
        if len(group) < 2:
            continue
        knobs_equal = all(
            len({_clean_value(entry.row.get(field)) for entry in group}) == 1 for field, _ in LABEL_KNOBS
        )
        variants = {entry.key: _clean_value(entry.row.get("variant")) for entry in group}
        if all(variants.values()) and len(set(variants.values())) == len(group):
            suffixes = _distinct_variant_tails({key: str(value) for key, value in variants.items()})
        elif knobs_equal:
            dates = {entry.key: _run_timestamp(entry.row, "%Y-%m-%d") for entry in group}
            date_counts = pd.Series(list(dates.values())).value_counts()
            suffixes = {
                entry.key: (
                    f"run {_run_timestamp(entry.row, '%Y-%m-%d %H:%M')}"
                    if date_counts.get(dates[entry.key], 0) > 1
                    else f"run {dates[entry.key]}"
                )
                for entry in group
            }
        else:
            suffixes = {
                entry.key: f"comparison {index}"
                for index, entry in enumerate(sorted(group, key=lambda item: item.key), start=1)
            }
        for entry in group:
            suffix = suffixes[entry.key]
            while len(_render_series_label(entry) + " · " + suffix) > BAR_LABEL_LIMIT and entry.parts:
                entry.parts.pop()
            entry.parts.append(suffix)

    duplicate_labels = {}
    for entry in entries:
        duplicate_labels.setdefault(_render_series_label(entry), []).append(entry)
    for group in duplicate_labels.values():
        if len(group) > 1:
            for index, entry in enumerate(sorted(group, key=lambda item: item.key), start=1):
                entry.parts.append(f"#{index}")

    return {entry.key: _render_series_label(entry) for entry in entries}


def latest_diagnostic_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest workload comparison for the diagnostic bars."""
    if df.empty or not set(DIAGNOSTIC_GROUP_FIELDS) <= set(df.columns):
        return df.copy()
    return df.sort_values("timestamp", kind="stable").drop_duplicates(
        list(DIAGNOSTIC_GROUP_FIELDS), keep="last"
    )


def _render_series_label(entry: SeriesLabelEntry) -> str:
    return " · ".join([entry.base, *entry.parts])


def _run_timestamp(row: pd.Series, format_string: str) -> str:
    timestamp = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
    return timestamp.strftime(format_string) if pd.notna(timestamp) else "unknown"


# ── Chart builders ────────────────────────────────────────────────────────────


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font_size=16,
    )
    fig.update_layout(height=360, margin={"l": 24, "r": 24, "t": 32, "b": 32})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return theme_figure(fig)


def theme_figure(fig: go.Figure) -> go.Figure:
    """Use Plotly's dark palette on dark pages and transparent surfaces for both modes."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f2f5fa",
    )
    return fig


def selected_rows(
    df: pd.DataFrame, suite: str | None, task: str | None, metric: str | None, unit: str | None
) -> pd.DataFrame:
    if suite is None or task is None or metric is None:
        return df.iloc[0:0].copy()
    rows = comparable_rows(df)
    selected = rows[(rows["suite"] == suite) & (rows["name"] == task) & (rows["metric"] == metric)].copy()
    if unit is not None and "display_unit" in selected:
        selected = selected[selected["display_unit"] == unit]
    return selected


def bar_chart(
    df: pd.DataFrame,
    suite: str | None,
    task: str | None,
    metric: str | None,
    unit: str | None = None,
    mode: str = AGENTIC_MODE,
) -> go.Figure:
    """Show the latest row for each fully identified comparison."""
    sub = selected_rows(df, suite, task, metric, unit)
    if sub.empty:
        return empty_figure("No data matches these filters. Clear a filter or choose another task.")

    if mode == DIAGNOSTIC_MODE:
        sub = latest_diagnostic_rows(sub)
    sub = sub.sort_values("timestamp").drop_duplicates("series_key", keep="last")
    labels = series_labels(sub)
    sub["label"] = sub["series_key"].map(labels)
    sub["display_value"] = sub.apply(
        lambda row: format_value(row["value"], row["display_unit"]), axis="columns"
    )
    sub["plot_value"] = sub.apply(lambda row: _chart_value(row["value"], row["display_unit"]), axis="columns")
    sub["workload"] = sub.apply(workload_details, axis="columns")
    sub["run_date"] = pd.to_datetime(sub["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    sub = sub.sort_values("plot_value", ascending=True)
    units = row_choices(sub, "display_unit")
    axis_unit = units[0] if len(units) == 1 else "unit varies by row"
    state = _clean_value(sub.iloc[0].get("evidence_status")) or "experimental"
    color = {"scored": "#126b72", "experimental": "#a95e19", "recovered": "#675493"}.get(state, "#647783")

    fig = px.bar(
        sub,
        x="plot_value",
        y="label",
        orientation="h",
        text="display_value",
        custom_data=["workload", "evidence_status", "run_date"],
        labels={"plot_value": f"{humanize(metric)} ({axis_unit})", "label": "Model and system"},
        title=f"{humanize(task)} — {humanize(metric)}",
        color_discrete_sequence=[color],
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>%{text}<br>%{customdata[0]}"
            "<br>Evidence: %{customdata[1]}<br>Run: %{customdata[2]}<extra></extra>"
        ),
        cliponaxis=False,
    )
    fig.update_layout(
        height=max(360, len(sub) * 48),
        margin={"l": 24, "r": 70, "t": 62, "b": 58},
        yaxis_title="",
        font_size=13,
        showlegend=False,
        barmode="group",
    )
    fig.update_yaxes(automargin=True)
    if axis_unit == "%":
        fig.update_xaxes(range=[0, 100], dtick=25, tickformat=".0f")
    return theme_figure(fig)


def trend_chart(
    df: pd.DataFrame,
    suite: str | None,
    task: str | None,
    metric: str | None,
    models: list[str] | None,
    unit: str | None = None,
) -> go.Figure:
    """Plot only compatible rows from the active suite, task, metric, and filters."""
    sub = selected_rows(df, suite, task, metric, unit)
    if sub.empty:
        return empty_figure("No trend rows match these filters.")
    if models is None:
        counts = sub["series_key"].value_counts().sort_values(ascending=False)
        models = counts.head(6).index.tolist()
    elif not models:
        return empty_figure("Select a comparison to show its trend.")
    sub = sub[sub["series_key"].isin(models)].sort_values("timestamp").copy()
    if sub.empty:
        return empty_figure("Choose a comparison with data for this task and metric.")

    labels = series_labels(sub)
    sub["label"] = sub["series_key"].map(labels)
    sub["display_value"] = sub.apply(
        lambda row: format_value(row["value"], row["display_unit"]), axis="columns"
    )
    sub["plot_value"] = sub.apply(lambda row: _chart_value(row["value"], row["display_unit"]), axis="columns")
    sub["workload"] = sub.apply(workload_details, axis="columns")
    units = row_choices(sub, "display_unit")
    axis_unit = units[0] if len(units) == 1 else "unit varies by row"
    fig = px.line(
        sub,
        x="timestamp",
        y="plot_value",
        color="label",
        markers=True,
        custom_data=["display_value", "workload", "evidence_status"],
        labels={
            "plot_value": f"{humanize(metric)} ({axis_unit})",
            "timestamp": "Run time (UTC)",
            "label": "Model and system",
        },
        title=f"{humanize(task)} — {humanize(metric)} over time",
        color_discrete_sequence=["#126b72", "#a95e19", "#675493", "#4b718e", "#98723b", "#4e7164"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>%{customdata[0]}<br>%{customdata[1]}"
            "<br>Evidence: %{customdata[2]}<br>Run: %{x|%Y-%m-%d %H:%M UTC}<extra></extra>"
        )
    )
    fig.update_layout(height=440, font_size=13, legend_title="", margin={"l": 72, "r": 24, "t": 62, "b": 62})
    if axis_unit == "%":
        fig.update_yaxes(range=[0, 100], dtick=25, tickformat=".0f")
    return theme_figure(fig)


def summary_table(
    df: pd.DataFrame,
    suite: str | None,
    task: str | None,
    metric: str | None,
    unit: str | None = None,
) -> pd.DataFrame:
    """Summarize only the active task and unit with evidence and run context."""
    selected = selected_rows(df, suite, task, metric, unit)
    if selected.empty:
        return pd.DataFrame({"No matching runs": []})

    row_counts = selected.groupby("series_key").size().rename("Runs (count)")
    latest = selected.sort_values("timestamp").drop_duplicates("series_key", keep="last").copy()
    latest["Runs (count)"] = latest["series_key"].map(row_counts)
    labels = series_labels(selected)
    latest["Model and system"] = latest["series_key"].map(labels)
    active_unit = unit or row_choices(latest, "display_unit")[0]
    latest[f"Value ({active_unit})"] = latest["value"].map(lambda value: format_value(value, active_unit))
    latest["Evidence"] = (
        latest["evidence_status"]
        .map({"scored": "Scored", "experimental": "Experimental", "recovered": "Recovered"})
        .fillna("Experimental")
    )
    latest["Run date (UTC)"] = pd.to_datetime(latest["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M")

    columns = ["Model and system"]
    context_values = latest["model"].map(model_context_values)
    if context_values.map(lambda value: value is not None).any():
        latest["Native model context"] = context_values.map(lambda value: value[0] if value else "")
        latest["YaRN-extended context"] = context_values.map(lambda value: value[1] if value else "")
        latest["Max recommended context"] = context_values.map(lambda value: value[2] if value else "")
        latest["Model card URL"] = context_values.map(lambda value: value[3] if value else "")
        columns.extend(
            ["Native model context", "YaRN-extended context", "Max recommended context", "Model card URL"]
        )
    if "largest_measured_input_tokens" in latest:
        latest["Largest input measured"] = latest["largest_measured_input_tokens"].map(
            lambda value: f"{format_tokens(value)} tokens" if _clean_value(value) is not None else ""
        )
        if latest["Largest input measured"].ne("").any():
            columns.append("Largest input measured")

    if not latest.empty and _is_throughput_row(latest.iloc[0]):
        metric_details = (
            ("Output tok/s", ("throughput_output_toks_per_s", "output_tokens_per_second")),
            ("TTFT p50 (ms)", ("ttft_p50_ms",)),
            ("TTFT p99 (ms)", ("ttft_p99_ms",)),
        )
        metrics = tuple(alias for _, aliases in metric_details for alias in aliases)
        metric_lookup = _run_metric_lookup(df, metrics)
        for label, aliases in metric_details:
            latest[label] = latest.apply(
                lambda row, metrics=aliases: _run_metric_value(df, row, metrics, metric_lookup),
                axis="columns",
            )
            columns.append(label)

    for field, aliases in (
        ("prefix_cache_hit_rate", ("campaign_speed_prefix_cache_hit_rate", "prefix_cache_hit_rate")),
        (
            "external_prefix_cache_hit_rate",
            ("campaign_speed_external_prefix_cache_hit_rate", "external_prefix_cache_hit_rate"),
        ),
    ):
        latest[field] = latest.apply(lambda row, keys=aliases: _row_value(row, keys), axis="columns")

    row_backed_fields = (
        ("machine", "Machine"),
        ("accelerator", "Accelerator model"),
        ("engine", "Engine"),
        ("engine_version", "Engine version"),
        ("backend", "Backend"),
        ("quant_format", "Quant format"),
        ("reasoning_effort", "Reasoning effort"),
        ("thinking", "Thinking arm"),
        ("mtp", "MTP"),
        ("speculative_tokens", "Speculative tokens"),
        ("kv_cache_dtype", "KV cache dtype"),
        ("prefix_cache_hit_rate", "Prefix cache hit rate"),
        ("external_prefix_cache_hit_rate", "External prefix cache hit rate"),
        ("power_limit_w", "Power limit (W)"),
        ("agents", "Agents (count)"),
        ("context_band", "Context band"),
        ("allocated_context", "Allocated context (tokens)"),
        ("context_tokens", "Context (tokens)"),
        ("prompt_tokens", "Prompt (tokens)"),
        ("output_budget_tokens", "Output budget (tokens)"),
        ("arrival_pattern", "Arrival pattern"),
        ("request_rate", "Request rate"),
        ("burstiness", "Burstiness"),
        ("completion_state", "Completion"),
        ("fan_duty", "Fan duty"),
        ("ambient_temp_f", "Ambient (°F)"),
        ("patches", "Patches"),
        ("dedicated", "Dedicated"),
        ("variant", "Variant"),
        ("caveat", "Caveat"),
    )

    def detail_value(raw: object, field: str) -> str:
        if field == "patches":
            return str(raw) if isinstance(raw, str) else ""
        value = _clean_value(raw)
        if value is None:
            return ""
        if field in {
            "agents",
            "allocated_context",
            "context_tokens",
            "prompt_tokens",
            "output_budget_tokens",
        }:
            suffix = "agent" if field == "agents" else "tokens"
            return f"{format_tokens(value)} {suffix}"
        if field in {"prefix_cache_hit_rate", "external_prefix_cache_hit_rate"}:
            return format_value(value, "%")
        suffix = {
            "power_limit_w": " W",
            "ambient_temp_f": " °F",
            "fan_duty": "%",
            "request_rate": " req/s",
        }.get(field, "")
        return f"{value}{suffix}"

    for field, label in row_backed_fields:
        if field not in latest:
            continue
        values = latest[field].map(lambda raw, key=field: detail_value(raw, key))
        if values.ne("").any():
            latest[label] = values
            columns.append(label)

    sample_value = latest.apply(
        lambda row: _row_value(row, ("n_samples", "sample_count", "n_limit", "tag_n_limit")), axis="columns"
    )
    repeat_value = latest.apply(
        lambda row: _row_value(
            row, ("measured_repetitions", "tag_measured_repetitions", "repetitions", "tag_repetitions")
        ),
        axis="columns",
    )
    if sample_value.map(_clean_value).notna().any():
        latest["Samples (count)"] = sample_value
        columns.append("Samples (count)")
    if repeat_value.map(_clean_value).notna().any():
        latest["Repeats (count)"] = repeat_value
        columns.append("Repeats (count)")

    columns.extend([f"Value ({active_unit})", "Evidence", "Runs (count)", "Run date (UTC)"])
    return latest[columns].reset_index(drop=True)


def _context_machine(row: pd.Series) -> str | None:
    return (
        _clean_value(row.get("machine"))
        or _clean_value(row.get("accelerator"))
        or _clean_value(row.get("host"))
    )


def _run_metric_lookup(df: pd.DataFrame, aliases: tuple[str, ...]):
    """Index latest companion metrics once instead of filtering the full frame per row."""
    if not {"metric", "value", "timestamp"} <= set(df.columns):
        return None

    key_fields = tuple(
        field for field in ("suite", "name", "model", "machine", "source_path") if field in df.columns
    )
    candidates = df[df["metric"].isin(aliases)]
    columns = [*key_fields, "metric", "value"]
    for field in ("display_unit", "unit"):
        if field in candidates.columns:
            columns.append(field)
        else:
            candidates = candidates.assign(**{field: pd.NA})
            columns.append(field)
    columns.append("timestamp")
    candidates = candidates[columns].copy()
    for field in key_fields:
        candidates[field] = candidates[field].map(_clean_value)
    candidates = candidates.sort_values("timestamp", kind="stable").drop_duplicates(
        [*key_fields, "metric"], keep="last"
    )

    key_width = len(key_fields) + 1
    lookup = {
        record[:key_width]: record[key_width : key_width + 3]
        for record in candidates.itertuples(index=False, name=None)
    }
    return key_fields, lookup


def _run_metric_value(
    df: pd.DataFrame,
    row: pd.Series,
    aliases: tuple[str, ...],
    lookup=None,
) -> str:
    if not {"metric", "value"} <= set(df.columns):
        return ""
    if lookup is not None:
        key_fields, values = lookup
        key = tuple(_clean_value(row.get(field)) for field in key_fields)
        if all(value is not None for value in key):
            for alias in aliases:
                result = values.get((*key, alias))
                if result is not None:
                    value, display_unit, unit = result
                    unit = _clean_value(display_unit) or _metric_unit(alias, unit)
                    return format_value(value, unit)
            return ""

    matches = df[df["metric"].isin(aliases)]
    for field in ("suite", "name", "model", "machine", "source_path"):
        value = _clean_value(row.get(field))
        if field in df and value is not None:
            matches = matches[matches[field].map(_clean_value) == value]
    if matches.empty:
        return ""
    metric = next((name for name in aliases if name in set(matches["metric"])), None)
    if metric is None:
        return ""
    value_row = matches[matches["metric"] == metric].sort_values("timestamp").iloc[-1]
    unit = _clean_value(value_row.get("display_unit")) or _metric_unit(metric, value_row.get("unit"))
    return format_value(value_row["value"], unit)


# ── Gradio UI ─────────────────────────────────────────────────────────────────


def filter_option_label(name: str, value: str) -> str:
    if name == "agents":
        return f"{value} agent" if value == "1" else f"{value} agents"
    if name in {"allocated_context", "context_tokens", "prompt_tokens", "output_budget_tokens"}:
        return f"{format_tokens(value)} tokens"
    return value


def filter_dropdown(values: list[str], label: str, value: str | None, name: str):
    # Every viewer dropdown accepts stale values: a second input event can arrive before the browser applies the
    # previous update, and synchronize() resets any value outside the current choices.
    return gr.Dropdown(
        choices=[(filter_option_label(name, item), item) for item in values],
        value=value if value in values else None,
        label=label,
        interactive=bool(values),
        visible=True,
        allow_custom_value=True,
        elem_id=f"filter-{name}",
    )


def build_ui():
    cache = _current_cache()
    initial_mode = AGENTIC_MODE
    evidence_options = [
        ("All valid", "all"),
        ("Scored", "scored"),
        ("Experimental", "experimental"),
        ("Recovered", "recovered"),
    ]
    initial_evidence = DEFAULT_EVIDENCE
    initial_rows = cached_viewer_rows(initial_mode, initial_evidence, cache)
    initial_filter_values = default_filter_selections(initial_rows, initial_mode)
    filtered_rows = filter_viewer_rows(initial_rows, initial_filter_values, initial_mode)
    default_suite, default_task, default_metric = best_default(filtered_rows)
    suites = suite_choices(filtered_rows)
    if default_suite not in suites:
        default_suite, default_task, default_metric = best_default(filtered_rows)
    tasks = task_choices(filtered_rows, default_suite)
    if default_task not in tasks and default_suite is not None:
        default_task, _ = top_task_metric(filtered_rows, default_suite)
    metrics = metric_choices(filtered_rows, default_suite, default_task)
    if default_metric not in metrics and default_suite is not None and default_task is not None:
        default_metric = top_metric(filtered_rows, default_suite, default_task)
    units = unit_choices(filtered_rows, default_suite, default_task, default_metric)
    initial_unit = units[0] if units else None
    active_rows = selected_rows(filtered_rows, default_suite, default_task, default_metric, initial_unit)
    initial_labels = series_labels(active_rows)
    initial_models = active_rows["series_key"].value_counts().head(6).index.tolist()

    def metric_options(data: pd.DataFrame, suite: str | None, task: str | None):
        options = []
        for metric_value in metric_choices(data, suite, task):
            metric_units = unit_choices(data, suite, task, metric_value)
            unit_text = f" ({', '.join(metric_units)})" if metric_units else ""
            options.append((f"{humanize(metric_value)}{unit_text}", metric_value))
        return options

    def dropdown(
        values: list[str],
        label: str,
        value: str | None = None,
        *,
        human_names: bool = False,
        interactive: bool | None = None,
        visible: bool | None = None,
        elem_id: str | None = None,
    ):
        choices = [
            (
                humanize(item) if human_names else item,
                item,
            )
            for item in values
        ]
        enabled = bool(values) if interactive is None else interactive
        return gr.Dropdown(
            choices=choices,
            value=value if value in values else None,
            label=label,
            interactive=enabled,
            visible=bool(values) if visible is None else visible,
            allow_custom_value=True,
            elem_id=elem_id,
        )

    def selection_status(
        counts: Mapping[str, int],
        rows: pd.DataFrame,
        evidence: str,
        suite: str | None,
        task: str | None,
        metric: str | None,
        unit: str | None,
    ) -> str:
        evidence_totals = evidence_summary(counts)
        if rows.empty:
            message = (
                f"No {evidence.capitalize()} rows match these filters. Clear a filter or choose another task."
            )
            return f"{message}  \n{evidence_totals}"
        model_count = rows["model"].map(_clean_value).nunique() if "model" in rows else 0
        comparison_count = rows["series_key"].nunique() if "series_key" in rows else 0
        comparison = " / ".join(humanize(value) for value in (suite, task, metric) if value)
        message = (
            f"**{len(rows):,} results** across **{comparison_count:,} comparisons** and "
            f"**{model_count:,} models**. {comparison} ({unit or 'unit not recorded'})."
        )
        missing = [label for name, (label, _) in FILTER_SPECS.items() if not row_choices(rows, name)]
        if missing:
            message += "  \nNot recorded for these rows: " + ", ".join(missing) + "."
        incomplete = (
            int((~rows["workload_complete"].fillna(False)).sum()) if "workload_complete" in rows else 0
        )
        if incomplete:
            message += f"  \nWorkload details are incomplete on {incomplete:,} rows."
        repeat_counts = rows.groupby("series_key").size() if "series_key" in rows else pd.Series(dtype=int)
        if not repeat_counts.gt(1).any():
            message += "  \nTrend shows individual results until comparable runs repeat over time."
        return f"{message}  \n{evidence_totals}"

    def mode_description(mode: str) -> str:
        if mode == DIAGNOSTIC_MODE:
            return (
                "Decode-speed diagnostics — short prompts, not used for model decisions. "
                "Includes throughput rows below 32k or with unknown input length."
            )
        return "Throughput comparisons require at least 32k input tokens; all evaluation suites are included."

    def default_models(rows: pd.DataFrame) -> list[str]:
        if rows.empty:
            return []
        return rows["series_key"].value_counts().head(6).index.tolist()

    def synchronize(
        mode, evidence, suite, task, metric, unit, selected_models, pick_default_context, *filter_values
    ):
        cache = _current_cache()
        data = cached_viewer_rows(str(mode), str(evidence), cache)
        selections = current_filter_selections(data, filter_values)
        available_choices = available_viewer_filter_choices_for_all(data, selections, str(mode))
        filter_updates = []
        for name, (label, _) in FILTER_SPECS.items():
            choices = available_choices[name]
            selected = selections.get(name)
            if name == "context_band" and pick_default_context and selected is None and mode == AGENTIC_MODE:
                selected = default_context_band(data)
            if selected not in choices:
                selected = None
            selections[name] = selected
            filter_updates.append(filter_dropdown(choices, label, selected, name))

        filtered = filter_viewer_rows(data, selections, str(mode))
        available_suites = suite_choices(filtered)
        if suite not in available_suites:
            suite, task, metric = best_default(filtered)
            # best_default skips single-machine and unscored views; a filter can leave only those.
            suite = suite or next(iter(available_suites), None)
        available_tasks = task_choices(filtered, suite)
        if task not in available_tasks and suite is not None:
            task, _ = top_task_metric(filtered, suite)
        available_metrics = metric_choices(filtered, suite, task)
        if metric not in available_metrics and suite is not None and task is not None:
            metric = top_metric(filtered, suite, task)
        available_units = unit_choices(filtered, suite, task, metric)
        if unit not in available_units:
            matching = selected_rows(filtered, suite, task, metric, None)
            counts = matching["display_unit"].value_counts() if not matching.empty else {}
            unit = max(available_units, key=lambda item: counts.get(item, 0)) if available_units else None

        active = selected_rows(filtered, suite, task, metric, unit)
        labels = series_labels(active)
        valid_models = list(labels)
        chosen_models = [value for value in (selected_models or []) if value in valid_models]
        if not chosen_models:
            chosen_models = default_models(active)
        model_choices = [(labels[key], key) for key in valid_models]

        suite_update = dropdown(available_suites, "Suite", suite, human_names=True)
        task_update = dropdown(available_tasks, "Task", task, human_names=True)
        metric_update = gr.Dropdown(
            choices=metric_options(filtered, suite, task),
            value=metric if metric in available_metrics else None,
            label="Metric" if available_metrics else "Metric — no matching rows",
            interactive=bool(available_metrics),
            allow_custom_value=True,
            elem_id="metric",
        )
        unit_update = gr.Dropdown(
            choices=[(value, value) for value in available_units],
            value=unit,
            label="Unit" if available_units else "Unit — no matching rows",
            interactive=len(available_units) > 1,
            allow_custom_value=True,
            elem_id="unit",
        )
        bar = bar_chart(filtered, suite, task, metric, unit, str(mode))
        trend = trend_chart(filtered, suite, task, metric, chosen_models, unit)
        table = summary_table(filtered, suite, task, metric, unit)
        status = selection_status(cache.evidence_counts, active, str(evidence), suite, task, metric, unit)
        model_update = gr.CheckboxGroup(
            choices=model_choices,
            value=chosen_models,
            label="Compatible comparisons with data",
            elem_id="trend-models",
        )
        return (
            suite_update,
            task_update,
            metric_update,
            unit_update,
            mode_description(str(mode)),
            *filter_updates,
            bar,
            model_update,
            trend,
            table,
            status,
        )

    def update_trend_from_models(mode, evidence, suite, task, metric, unit, *values):
        selected_models = values[-1] if values else []
        filter_values = values[:-1]
        rows = cached_viewer_rows(str(mode), str(evidence))
        data = filter_viewer_rows(rows, current_filter_selections(rows, filter_values), str(mode))
        return trend_chart(data, suite, task, metric, selected_models, unit)

    def synchronize_mode(mode, evidence, *filter_values):
        return synchronize(mode, evidence, None, None, None, None, None, True, *filter_values)

    def synchronize_evidence(mode, evidence, *filter_values):
        return synchronize(mode, evidence, None, None, None, None, None, True, *filter_values)

    def synchronize_suite(mode, evidence, suite, *filter_values):
        return synchronize(mode, evidence, suite, None, None, None, None, False, *filter_values)

    def synchronize_task(mode, evidence, suite, task, *filter_values):
        return synchronize(mode, evidence, suite, task, None, None, None, False, *filter_values)

    def synchronize_metric(mode, evidence, suite, task, metric, *filter_values):
        return synchronize(mode, evidence, suite, task, metric, None, None, False, *filter_values)

    def synchronize_unit(mode, evidence, suite, task, metric, unit, *filter_values):
        return synchronize(mode, evidence, suite, task, metric, unit, None, False, *filter_values)

    def synchronize_filters(mode, evidence, suite, task, metric, unit, *filter_values):
        return synchronize(mode, evidence, suite, task, metric, unit, None, False, *filter_values)

    def refresh(mode, evidence, suite, task, metric, unit, *filter_values):
        request_data_refresh()
        return synchronize(mode, evidence, suite, task, metric, unit, None, True, *filter_values)

    with gr.Blocks(title="MLX Benchmarks") as demo:
        gr.Markdown("# MLX Benchmarks", elem_id="title")
        gr.Markdown(
            "Compare one task and metric at a time. Workload, machine, and evidence stay attached to every run.  \n"
            "Dataset: [MLX Benchmarks](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks)",
            elem_id="subtitle",
        )
        with gr.Row():
            mode_dd = gr.Radio(
                choices=[(AGENTIC_MODE, AGENTIC_MODE), (DIAGNOSTIC_MODE, DIAGNOSTIC_MODE)],
                value=initial_mode,
                label="Benchmark focus",
                scale=1,
                elem_id="mode",
            )
        mode_help = gr.Markdown(mode_description(initial_mode), elem_id="mode-description")
        with gr.Row():
            evidence_dd = gr.Radio(
                choices=evidence_options,
                value=initial_evidence,
                label="Evidence",
                scale=1,
                elem_id="evidence",
            )
        with gr.Row():
            suite_dd = dropdown(suites, "Suite", default_suite, human_names=True, elem_id="suite")
            task_dd = dropdown(tasks, "Task", default_task, human_names=True, elem_id="task")
            metric_dd = gr.Dropdown(
                choices=metric_options(filtered_rows, default_suite, default_task),
                value=default_metric,
                label="Metric",
                interactive=bool(metrics),
                allow_custom_value=True,
                elem_id="metric",
            )
            unit_dd = gr.Dropdown(
                choices=[(value, value) for value in units],
                value=initial_unit,
                label="Unit",
                interactive=len(units) > 1,
                allow_custom_value=True,
                elem_id="unit",
            )
            refresh_btn = gr.Button("Refresh data", scale=0)

        with gr.Accordion("Filter by machine and workload", open=True):
            filter_controls = []
            filter_items = list(FILTER_SPECS.items())
            for start in range(0, len(filter_items), 5):
                with gr.Row():
                    for name, (label, _) in filter_items[start : start + 5]:
                        choices = row_choices(initial_rows, name)
                        filter_controls.append(
                            filter_dropdown(choices, label, initial_filter_values[name], name)
                        )

        status = gr.Markdown(
            selection_status(
                cache.evidence_counts,
                active_rows,
                initial_evidence,
                default_suite,
                default_task,
                default_metric,
                initial_unit,
            ),
            elem_id="selection-status",
        )
        with gr.Tabs():
            with gr.Tab("Latest"):
                bar_plot = gr.Plot(
                    value=bar_chart(
                        filtered_rows, default_suite, default_task, default_metric, initial_unit, initial_mode
                    ),
                    label="Latest comparable run",
                )
            with gr.Tab("Trend"):
                model_select = gr.CheckboxGroup(
                    choices=[(initial_labels[key], key) for key in initial_models],
                    value=initial_models,
                    label="Compatible comparisons with data",
                    elem_id="trend-models",
                )
                trend_plot = gr.Plot(
                    value=trend_chart(
                        filtered_rows,
                        default_suite,
                        default_task,
                        default_metric,
                        initial_models,
                        initial_unit,
                    ),
                    label="Results over time",
                )
            with gr.Tab("Summary"):
                table_out = gr.DataFrame(
                    value=summary_table(
                        filtered_rows, default_suite, default_task, default_metric, initial_unit
                    ),
                    interactive=False,
                    label="Latest run per comparison",
                )

        filter_controls = tuple(filter_controls)
        sync_outputs = [
            suite_dd,
            task_dd,
            metric_dd,
            unit_dd,
            mode_help,
            *filter_controls,
            bar_plot,
            model_select,
            trend_plot,
            table_out,
            status,
        ]
        mode_dd.input(
            synchronize_mode,
            inputs=[mode_dd, evidence_dd, *filter_controls],
            outputs=sync_outputs,
        )
        evidence_dd.input(
            synchronize_evidence,
            inputs=[mode_dd, evidence_dd, *filter_controls],
            outputs=sync_outputs,
        )
        suite_dd.input(
            synchronize_suite,
            inputs=[mode_dd, evidence_dd, suite_dd, *filter_controls],
            outputs=sync_outputs,
        )
        task_dd.input(
            synchronize_task,
            inputs=[mode_dd, evidence_dd, suite_dd, task_dd, *filter_controls],
            outputs=sync_outputs,
        )
        metric_dd.input(
            synchronize_metric,
            inputs=[mode_dd, evidence_dd, suite_dd, task_dd, metric_dd, *filter_controls],
            outputs=sync_outputs,
        )
        unit_dd.input(
            synchronize_unit,
            inputs=[mode_dd, evidence_dd, suite_dd, task_dd, metric_dd, unit_dd, *filter_controls],
            outputs=sync_outputs,
        )
        for control in filter_controls:
            control.input(
                synchronize_filters,
                inputs=[mode_dd, evidence_dd, suite_dd, task_dd, metric_dd, unit_dd, *filter_controls],
                outputs=sync_outputs,
            )

        model_select.input(
            update_trend_from_models,
            inputs=[
                mode_dd,
                evidence_dd,
                suite_dd,
                task_dd,
                metric_dd,
                unit_dd,
                *filter_controls,
                model_select,
            ],
            outputs=[trend_plot],
        )
        refresh_btn.click(
            refresh,
            inputs=[mode_dd, evidence_dd, suite_dd, task_dd, metric_dd, unit_dd, *filter_controls],
            outputs=sync_outputs,
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(css=CSS, theme=THEME)
