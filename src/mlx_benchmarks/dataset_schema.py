"""Stable flat Parquet schema shared by every published run shard."""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

import pyarrow as pa

TAG_KEYS = (
    "aggregation",
    "answered_rate",
    "arm",
    "backend",
    "cache_busting",
    "campaign",
    "campaign_id",
    "caveat",
    "caveat2",
    "caveats",
    "cell",
    "cell_id",
    "check",
    "check_rc",
    "clean_reset",
    "clustered",
    "completed_requests",
    "concurrency",
    "configured_window_tokens",
    "context",
    "context_len",
    "context_tokens_actual",
    "context_tokens_target",
    "cumulative_tok_s_max",
    "cumulative_tok_s_min",
    "decode_tok_s_max",
    "decode_tok_s_min",
    "dedicated",
    "degraded",
    "domain",
    "experiment",
    "failure_bad_json_args",
    "failure_bad_response_body",
    "failure_empty_function_name",
    "failure_http_error",
    "failure_no_tool_call",
    "failure_stream_truncated",
    "failure_timeout",
    "failure_unknown_tool",
    "filter",
    "finish_reasons",
    "initial_model_state",
    "lm_eval_key",
    "maturity",
    "max_gen_toks",
    "max_tokens",
    "measured_repetitions",
    "method",
    "mtp",
    "mtp_runtime_commit",
    "n_err",
    "n_limit",
    "n_ok",
    "n_requests",
    "n_runs",
    "n_tasks",
    "output_reservation_tokens",
    "overlap",
    "phase",
    "prefill_tok_s_max",
    "prefill_tok_s_min",
    "prefix_cache_tokens",
    "profile",
    "prompt",
    "prompt_set",
    "prompt_tokens",
    "provisional",
    "published_from",
    "quant",
    "quiesce_window",
    "reasoning_effort",
    "recovered_partial",
    "repetition_penalty",
    "repetitions",
    "repo",
    "requested_prompt_tokens",
    "reserved_output_tokens",
    "rounds",
    "run",
    "run_tag",
    "scoring",
    "serving",
    "speculative_decoding",
    "spread_tok_s",
    "stream",
    "swap_baseline_mb",
    "swap_growth_mb",
    "task",
    "temperature",
    "think_kwarg",
    "think_kwarg_sent",
    "think_value",
    "thinking",
    "total_eval_time_s",
    "total_input_tokens",
    "total_output_tokens",
    "total_s_max",
    "total_s_min",
    "track",
    "truncated_rate",
    "ttft_s_max",
    "ttft_s_min",
    "valid_rounds",
    "validated",
    "wall_s",
    "warmup_runs",
    "width",
)

OPTIONAL_RESULT_COLUMNS = (
    "duration_seconds",
    "prompt_tokens_per_second",
    "decode_tokens_per_second",
    "total_tokens_per_second",
    "first_token_latency_ms",
    "peak_rss_mb",
    "output_tokens_mean",
    "output_tokens_median",
    "output_tokens_p90",
    "reasoning_tokens_mean",
    "tokens_per_correct",
    "truncation_rate",
)

OPTIONAL_RESULT_STRING_COLUMNS = ("reasoning_arm",)

SYSTEM_COLUMNS = (
    "os",
    "chip",
    "memory_gb",
    "hostname",
    "vllm_mlx_version",
    "runner",
    "python_version",
    "mlx_version",
    "lm_eval_version",
    "kernel",
    "gpu",
    "engine",
    "power_limit_w",
    "container",
    "topology",
)

DERIVED_THROUGHPUT_SPEED_FIELDS = (
    "measured_mean_concurrency",
    "potential_output_tok_s_p50",
    "potential_output_tok_s_p90",
    "potential_output_tok_s_p99",
    "potential_output_tok_s_p50_at_mean_concurrency",
    "potential_output_tok_s_p90_at_mean_concurrency",
    "potential_output_tok_s_p99_at_mean_concurrency",
    "peak_output_tok_s_1s_max",
    "peak_output_tok_s_1s_p99",
    "potential_efficiency",
)
THROUGHPUT_SUITES = frozenset({"throughput", "throughput-serving"})

CAMPAIGN_DIMENSION_TYPES = {
    "environment": {
        "ambient_temp_f": pa.float64(),
        "fan_duty_pct": pa.string(),
    },
    "hardware": {
        "machine": pa.string(),
        "interconnect": pa.string(),
        "world_size": pa.int64(),
        "accelerator_model": pa.string(),
        "accelerator_memory_gb": pa.float64(),
        "power_source": pa.string(),
        "power_mode": pa.string(),
        "memory_bandwidth_gbps": pa.float64(),
        "host_cpu": pa.string(),
        "host_ram_gb": pa.float64(),
        "host_ram_limit_gib": pa.float64(),
        "host_ram_speed_mts": pa.float64(),
        "pcie_generation": pa.float64(),
        "pcie_width_lanes": pa.int64(),
        "power_cap_w": pa.float64(),
        "ups_circuit": pa.string(),
        "chassis_container": pa.string(),
        "nautobot_device_id": pa.string(),
    },
    "software": {
        "operating_system": pa.string(),
        "macos_version": pa.string(),
        "kernel": pa.string(),
        "driver_version": pa.string(),
        "accelerator_runtime": pa.string(),
        "engine": pa.string(),
        "engine_build": pa.string(),
        "engine_is_latest_stable": pa.bool_(),
        "engine_latest_stable_at_run": pa.string(),
        "engine_release_date": pa.string(),
        "engine_version": pa.string(),
        "engine_commit": pa.string(),
        "backend": pa.string(),
        "gpu_architectures": pa.string(),
        "build_flags": pa.string(),
        "flash_attention": pa.bool_(),
        "cuda_toolkit_version": pa.string(),
    },
    "model": {
        "family": pa.string(),
        "id": pa.string(),
        "hf_repo": pa.string(),
        "revision_sha": pa.string(),
        "total_parameters": pa.int64(),
        "active_parameters": pa.int64(),
        "architecture": pa.string(),
        "experts_total": pa.int64(),
        "experts_active": pa.int64(),
        "quantization": pa.string(),
        "bits_per_weight": pa.float64(),
        "file_size_gb": pa.float64(),
        "license": pa.string(),
        "pipeline_tag": pa.string(),
        "native_max_context_tokens": pa.int64(),
        "file_sha256": pa.string(),
    },
    "run": {
        "allocated_context_tokens": pa.int64(),
        "kv_cache_dtype": pa.string(),
        "prompt_tokens": pa.int64(),
        "depth_tokens": pa.int64(),
        "output_tokens": pa.int64(),
        "measured_output_tokens": pa.int64(),
        "concurrent_agents": pa.int64(),
        "load_phase": pa.string(),
        "phase_duration_seconds": pa.int64(),
        "closed_loop": pa.bool_(),
        "streaming": pa.bool_(),
        "embedding_items_per_request": pa.int64(),
        "embedding_input_tokens_per_request": pa.int64(),
        "batch_size": pa.int64(),
        "ubatch_size": pa.int64(),
        "parallel_slots": pa.int64(),
        "prefix_cache": pa.bool_(),
        "kv_offload_backend": pa.string(),
        "kv_offload_cpu_gib": pa.float64(),
        "kv_offload_disk_gib": pa.float64(),
        "weight_cpu_offload_gib": pa.float64(),
        "moe_cpu_offload": pa.string(),
        "moe_cpu_layers": pa.int64(),
        "llamacpp_cache_ram_mib": pa.int64(),
        "gpu_memory_utilization": pa.float64(),
        "max_num_batched_tokens": pa.int64(),
        "cache_workload": pa.string(),
        "num_users": pa.int64(),
        "num_rounds": pa.int64(),
        "shared_prompt_tokens": pa.int64(),
        "user_history_tokens": pa.int64(),
        "request_rate": pa.float64(),
        "speculative_mtp": pa.bool_(),
        "draft_model": pa.string(),
        "temperature": pa.float64(),
        "thinking": pa.bool_(),
        "chat_template": pa.string(),
        "seed": pa.int64(),
        "repeats": pa.int64(),
        "sample_slice_index": pa.int64(),
        "warm_cold": pa.string(),
        "started_at": pa.string(),
        "ended_at": pa.string(),
        "patches": pa.string(),
        "knob_case": pa.string(),
        "knobs": pa.string(),
        "sampling": pa.string(),
        "num_speculative_tokens": pa.int64(),
        "kv_pool_tokens": pa.int64(),
        "max_running_requests": pa.int64(),
        "ple_offload_target": pa.string(),
        "load_time_s": pa.float64(),
    },
    "speed": {
        "ttft_p50_ms": pa.float64(),
        "ttft_p90_ms": pa.float64(),
        "ttft_p95_ms": pa.float64(),
        "ttft_p99_ms": pa.float64(),
        "request_latency_p50_ms": pa.float64(),
        "request_latency_p95_ms": pa.float64(),
        "request_latency_p99_ms": pa.float64(),
        "requests_per_second": pa.float64(),
        "embeddings_per_second": pa.float64(),
        "embedding_tokens_per_second": pa.float64(),
        "embedding_latency_p50_ms": pa.float64(),
        "embedding_latency_p95_ms": pa.float64(),
        "embedding_latency_p99_ms": pa.float64(),
        "prefill_tokens_per_second": pa.float64(),
        "decode_tokens_per_second_per_agent": pa.float64(),
        "aggregate_output_tokens_per_second": pa.float64(),
        "total_tokens_per_second": pa.float64(),
        "measured_mean_concurrency": pa.float64(),
        "potential_output_tok_s_p50": pa.float64(),
        "potential_output_tok_s_p90": pa.float64(),
        "potential_output_tok_s_p99": pa.float64(),
        "potential_output_tok_s_p50_at_mean_concurrency": pa.float64(),
        "potential_output_tok_s_p90_at_mean_concurrency": pa.float64(),
        "potential_output_tok_s_p99_at_mean_concurrency": pa.float64(),
        "peak_output_tok_s_1s_max": pa.float64(),
        "peak_output_tok_s_1s_p99": pa.float64(),
        "potential_efficiency": pa.float64(),
        "tpot_ms": pa.float64(),
        "itl_p50_ms": pa.float64(),
        "itl_p99_ms": pa.float64(),
        "mtp_acceptance_rate": pa.float64(),
        "request_success_rate": pa.float64(),
        "mtp_accept_length": pa.float64(),
        "kv_cache_usage_peak_pct": pa.float64(),
        "prefix_cache_hit_rate": pa.float64(),
        "external_prefix_cache_hit_rate": pa.float64(),
        "kv_block_idle_before_evict_p50_s": pa.float64(),
        "ttft_first_round_p50_ms": pa.float64(),
        "ttft_later_rounds_p50_ms": pa.float64(),
        "ttft_later_rounds_p99_ms": pa.float64(),
        "queue_time_p50_ms": pa.float64(),
        "queue_time_p99_ms": pa.float64(),
        "goodput_pct": pa.float64(),
        "c_at_p10_35tps": pa.int64(),
    },
    "resource": {
        "accelerator_memory_peak_gb": pa.float64(),
        "host_ram_peak_gb": pa.float64(),
        "host_ram_peak_gib": pa.float64(),
        "preemptions": pa.int64(),
        "cpu_offload_layers": pa.int64(),
        "ple_bytes": pa.int64(),
        "gpu_utilization_avg_percent": pa.float64(),
        "gpu_utilization_max_percent": pa.float64(),
        "cpu_performance_utilization_avg_percent": pa.float64(),
        "cpu_efficiency_utilization_avg_percent": pa.float64(),
        "power_avg_w": pa.float64(),
        "power_max_w": pa.float64(),
        "system_power_avg_w": pa.float64(),
        "system_power_max_w": pa.float64(),
        "soc_power_avg_w": pa.float64(),
        "soc_power_max_w": pa.float64(),
        "cpu_power_avg_w": pa.float64(),
        "cpu_power_max_w": pa.float64(),
        "memory_power_avg_w": pa.float64(),
        "memory_power_max_w": pa.float64(),
        "accelerator_memory_power_avg_w": pa.float64(),
        "accelerator_memory_power_max_w": pa.float64(),
        "ane_power_avg_w": pa.float64(),
        "ane_power_max_w": pa.float64(),
        "energy_j_per_output_token": pa.float64(),
        "tokens_per_second_per_watt": pa.float64(),
        "temperature_max_c": pa.float64(),
        "average_clocks_mhz": pa.float64(),
        "cpu_performance_clock_avg_mhz": pa.float64(),
        "cpu_efficiency_clock_avg_mhz": pa.float64(),
        "throttle_reasons": pa.string(),
        "fan_percent": pa.float64(),
    },
    "quality": {
        "benchmark_name": pa.string(),
        "benchmark_version": pa.string(),
        "subset": pa.string(),
        "sample_count": pa.int64(),
        "score": pa.float64(),
        "standard_error": pa.float64(),
        "confidence_interval_low": pa.float64(),
        "confidence_interval_high": pa.float64(),
        "judge_model": pa.string(),
        "contamination_note": pa.string(),
    },
    "provenance": {
        "timestamp_utc": pa.string(),
        "run_id": pa.string(),
        "config_name": pa.string(),
        "config_git_sha": pa.string(),
        "operator_agent": pa.string(),
        "raw_output_path": pa.string(),
        "schema_version": pa.string(),
    },
    "cost": {
        "kwh_per_million_output_tokens": pa.float64(),
        "local_electricity_usd_per_million_tokens": pa.float64(),
        "median_rental_usd_per_million_tokens": pa.float64(),
        "median_api_usd_per_million_tokens": pa.float64(),
    },
}

# Standard cache-field names map onto these stable campaign dimensions. Existing
# dimensions are reused when their meaning and units match the shared contract.
STANDARD_CACHE_FIELD_DIMENSIONS = {
    "host_ram_limit_gib": ("hardware", "host_ram_limit_gib"),
    "prefix_caching": ("run", "prefix_cache"),
    "kv_offload_backend": ("run", "kv_offload_backend"),
    "kv_offload_cpu_gib": ("run", "kv_offload_cpu_gib"),
    "kv_offload_disk_gib": ("run", "kv_offload_disk_gib"),
    "weight_cpu_offload_gib": ("run", "weight_cpu_offload_gib"),
    "llamacpp_cache_ram_mib": ("run", "llamacpp_cache_ram_mib"),
    "gpu_memory_utilization": ("run", "gpu_memory_utilization"),
    "max_num_batched_tokens": ("run", "max_num_batched_tokens"),
    "gpu_kv_capacity_tokens": ("run", "kv_pool_tokens"),
    "cache_workload": ("run", "cache_workload"),
    "num_users": ("run", "num_users"),
    "num_rounds": ("run", "num_rounds"),
    "shared_prompt_tokens": ("run", "shared_prompt_tokens"),
    "user_history_tokens": ("run", "user_history_tokens"),
    "answer_tokens": ("run", "output_tokens"),
    "request_rate": ("run", "request_rate"),
    "prefix_cache_hit_rate": ("speed", "prefix_cache_hit_rate"),
    "external_prefix_cache_hit_rate": ("speed", "external_prefix_cache_hit_rate"),
    "kv_cache_usage_peak": ("speed", "kv_cache_usage_peak_pct"),
    "kv_block_idle_before_evict_p50_s": ("speed", "kv_block_idle_before_evict_p50_s"),
    "host_ram_peak_gib": ("resource", "host_ram_peak_gib"),
    "preemptions": ("resource", "preemptions"),
    "ttft_first_round_p50_ms": ("speed", "ttft_first_round_p50_ms"),
    "ttft_later_rounds_p50_ms": ("speed", "ttft_later_rounds_p50_ms"),
    "ttft_later_rounds_p99_ms": ("speed", "ttft_later_rounds_p99_ms"),
}
CACHE_CONFIGURATION_FIELDS = (
    "host_ram_limit_gib",
    "prefix_caching",
    "kv_offload_backend",
    "kv_offload_cpu_gib",
    "kv_offload_disk_gib",
    "weight_cpu_offload_gib",
    "llamacpp_cache_ram_mib",
    "gpu_memory_utilization",
    "max_num_batched_tokens",
    "gpu_kv_capacity_tokens",
)
STANDARD_CACHE_FIELD_TYPES = {
    name: CAMPAIGN_DIMENSION_TYPES[category][field]
    for name, (category, field) in STANDARD_CACHE_FIELD_DIMENSIONS.items()
}
STANDARD_CACHE_FIELD_DESCRIPTIONS = {
    "host_ram_limit_gib": "Memory limit available to the serving host, in GiB.",
    "prefix_caching": "Whether prefix caching was enabled.",
    "kv_offload_backend": "KV-cache offload backend: none, native, lmcache, or llamacpp_cache_ram.",
    "kv_offload_cpu_gib": "Configured KV-cache offload capacity in host RAM, in GiB.",
    "kv_offload_disk_gib": "Configured KV-cache offload capacity on disk, in GiB.",
    "weight_cpu_offload_gib": "Model-weight offload capacity in host RAM, in GiB.",
    "llamacpp_cache_ram_mib": "llama.cpp cache-ram setting in MiB.",
    "gpu_memory_utilization": "Fraction of GPU memory reserved by the serving engine, from 0 to 1.",
    "max_num_batched_tokens": "Maximum number of tokens processed in one serving batch.",
    "gpu_kv_capacity_tokens": "Configured total token capacity of the serving KV pool, from the engine startup log.",
    "cache_workload": "Cache pressure class: fits_gpu, spills_to_ram, exceeds_all, or none.",
    "num_users": "Number of simulated users in the cache workload.",
    "num_rounds": "Number of interaction rounds per simulated user.",
    "shared_prompt_tokens": "Token count in the prompt shared across users.",
    "user_history_tokens": "History token count retained for each simulated user.",
    "answer_tokens": "Requested output token count per answer.",
    "request_rate": "Configured request arrival rate in requests per second.",
    "prefix_cache_hit_rate": "Prefix-cache hit ratio as a fraction of eligible lookups.",
    "external_prefix_cache_hit_rate": "External prefix-cache hit ratio as a fraction of eligible lookups.",
    "kv_cache_usage_peak": "Peak KV-cache occupancy as a percentage of configured capacity.",
    "kv_block_idle_before_evict_p50_s": "Median idle time before a KV-cache block is evicted, in seconds.",
    "host_ram_peak_gib": "Peak host RAM used during the run, in GiB.",
    "preemptions": "Number of requests preempted by the serving engine.",
    "ttft_first_round_p50_ms": "Median time to first token in the first interaction round, in milliseconds.",
    "ttft_later_rounds_p50_ms": "Median time to first token after the first interaction round, in milliseconds.",
    "ttft_later_rounds_p99_ms": "99th percentile time to first token after the first interaction round, in milliseconds.",
}
CACHE_OFFLOAD_BACKENDS = frozenset({"none", "native", "lmcache", "llamacpp_cache_ram"})
CACHE_WORKLOADS = frozenset({"fits_gpu", "spills_to_ram", "exceeds_all", "none"})

CAMPAIGN_DIMENSION_JSON_FIELDS = frozenset(
    {
        ("environment", "fan_duty_pct"),
        ("software", "gpu_architectures"),
        ("software", "build_flags"),
        ("resource", "throttle_reasons"),
        ("run", "patches"),
        ("run", "knobs"),
        ("run", "sampling"),
    }
)


def campaign_dimension_column(category: str, field: str) -> str:
    """Return the fixed Parquet column name for one campaign dimension."""
    suffix = "_json" if (category, field) in CAMPAIGN_DIMENSION_JSON_FIELDS else ""
    return f"campaign_{category}_{field}{suffix}"


def _apply_campaign_dimension_projection(
    row: dict[str, Any],
    campaign_dimensions: Mapping[str, Any] | None,
    dimension_null_reasons: Mapping[str, str] | None,
) -> None:
    """Apply validated per-shard dimensions to a normalized row."""
    if campaign_dimensions is not None:
        if not isinstance(campaign_dimensions, Mapping):
            raise ValueError("run-index campaign_dimensions must be an object")
        for category, values in campaign_dimensions.items():
            fields = CAMPAIGN_DIMENSION_TYPES.get(category)
            if fields is None or not isinstance(values, Mapping):
                raise ValueError(f"invalid run-index campaign_dimensions category: {category!r}")
            for field, value in values.items():
                if field not in fields:
                    raise ValueError(f"unknown run-index campaign dimension: {category}.{field}")
                if value is not None and (category, field) in CAMPAIGN_DIMENSION_JSON_FIELDS:
                    value = json.dumps(value, sort_keys=True)
                row[campaign_dimension_column(category, field)] = value

    if dimension_null_reasons is None:
        return
    if not isinstance(dimension_null_reasons, Mapping) or not all(
        isinstance(field, str) and isinstance(reason, str) for field, reason in dimension_null_reasons.items()
    ):
        raise ValueError("run-index dimension_null_reasons must map field paths to strings")
    current = row.get("campaign_dimension_null_reasons_json")
    if current is None:
        reasons: dict[str, str] = {}
    else:
        decoded = json.loads(current) if isinstance(current, str) else current
        if not isinstance(decoded, dict):
            raise ValueError("historical campaign dimension null reasons must contain a JSON object")
        reasons = {str(field): str(reason) for field, reason in decoded.items()}
    reasons.update(dimension_null_reasons)
    row["campaign_dimension_null_reasons_json"] = json.dumps(reasons, sort_keys=True)


JSON_STRING_COLUMNS = frozenset({"serving", "gpu", "system_engine", "container", "topology"})
PUBLISHED_JSON_COLUMNS = frozenset({"base_model", "dtype", "hardware", "dimension_null_reasons"})
APPLE_MACHINE_FALLBACK = "Apple M4 Max"
_MACHINE_LABEL_PATTERNS = (
    re.compile(r"Mac Studio M4 Max 128GB\Z"),
    re.compile(r"MacBook Pro M4 Max [1-9][0-9]*GB\Z"),
    re.compile(r"RTX PRO 6000 Max-Q 96GB\Z"),
    re.compile(r"RTX 4080 SUPER 16GB\Z"),
    re.compile(r"2x Apple M4 Max 128GB \(Thunderbolt RDMA\)\Z"),
)
_MACHINE_ID_FIELDS = (
    "campaign_hardware_machine",
    "machine",
    "hostname",
    "host",
    "published_from",
)
_NESTED_MACHINE_ID_FIELDS = frozenset(
    {"campaign_hardware_machine", "machine", "hostname", "host", "node_name"}
)

PARQUET_ROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("timestamp", pa.string()),
        ("git_sha", pa.string()),
        ("trigger", pa.string()),
        ("suite", pa.string()),
        ("model", pa.string()),
        ("model_task", pa.string()),
        ("model_task_source", pa.string()),
        ("model_revision", pa.string()),
        ("quantization", pa.string()),
        ("seed", pa.int64()),
        ("env_class", pa.string()),
        ("concurrency", pa.int64()),
        ("reasoning_effort", pa.string()),
        ("serving", pa.string()),
        ("os", pa.string()),
        ("chip", pa.string()),
        ("memory_gb", pa.int64()),
        ("hostname", pa.string()),
        ("vllm_mlx_version", pa.string()),
        ("runner", pa.string()),
        ("python_version", pa.string()),
        ("mlx_version", pa.string()),
        ("lm_eval_version", pa.string()),
        ("kernel", pa.string()),
        ("gpu", pa.string()),
        ("system_engine", pa.string()),
        ("engine", pa.string()),
        ("power_limit_w", pa.float64()),
        ("container", pa.string()),
        ("topology", pa.string()),
        ("system_extra", pa.string()),
        *[
            (campaign_dimension_column(category, field), column_type)
            for category, fields in CAMPAIGN_DIMENSION_TYPES.items()
            for field, column_type in fields.items()
        ],
        ("campaign_dimension_null_reasons_json", pa.string()),
        ("name", pa.string()),
        ("metric", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        *[(column, pa.float64()) for column in OPTIONAL_RESULT_COLUMNS],
        *[(column, pa.string()) for column in OPTIONAL_RESULT_STRING_COLUMNS],
        *[(f"tag_{key}", pa.string()) for key in TAG_KEYS],
        ("tags_json", pa.string()),
        ("model_id", pa.string()),
        ("pipeline_tag", pa.string()),
        ("library_name", pa.string()),
        ("license", pa.string()),
        ("license_name", pa.string()),
        ("license_link", pa.string()),
        ("base_model", pa.string()),
        ("base_model_relation", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("parameters_total", pa.int64()),
        ("dtype", pa.string()),
        ("quant", pa.string()),
        ("architectures", pa.list_(pa.string())),
        ("model_type", pa.string()),
        ("context_length", pa.int64()),
        ("gated", pa.string()),
        ("dataset_id", pa.string()),
        ("dataset_task_id", pa.string()),
        ("dataset_revision", pa.string()),
        ("date", pa.string()),
        ("source_url", pa.string()),
        ("notes", pa.string()),
        ("evaluation_framework", pa.string()),
        ("config", pa.string()),
        ("split", pa.string()),
        ("hardware", pa.string()),
        ("engine_version", pa.string()),
        ("profile", pa.string()),
        ("ctx_per_slot", pa.int64()),
        ("kv_cache_dtype", pa.string()),
        ("thinking", pa.bool_()),
        ("temperature", pa.float64()),
        ("max_tokens", pa.int64()),
        ("prompt_chars", pa.int64()),
        ("prompt_tokens", pa.int64()),
        ("system_prompt_chars", pa.int64()),
        ("system_prompt_tokens", pa.int64()),
        ("harness", pa.string()),
        ("router_key_alias", pa.string()),
        ("run_id", pa.string()),
        ("start_utc", pa.string()),
        ("end_utc", pa.string()),
        ("dimension_null_reasons", pa.string()),
        ("extra_json", pa.string()),
    ]
)


def empty_parquet_row() -> dict[str, Any]:
    """Return one row with every canonical column present and unset."""
    return dict.fromkeys(PARQUET_ROW_SCHEMA.names)


def _machine_label_is_valid(value: object) -> bool:
    return value == APPLE_MACHINE_FALLBACK or (
        isinstance(value, str) and any(pattern.fullmatch(value) for pattern in _MACHINE_LABEL_PATTERNS)
    )


def parse_machine_labels(value: str | None) -> dict[str, str]:
    """Parse and validate the optional hostname-to-label Actions secret."""
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("machine label configuration must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("machine label configuration contains an invalid entry")
    labels: dict[str, str] = {}
    for host, label in decoded.items():
        if not isinstance(host, str) or not host.strip() or not _machine_label_is_valid(label):
            raise ValueError("machine label configuration contains an invalid entry")
        labels[host] = label
    return labels


def _machine_label_hardware_text(row: Mapping[str, Any]) -> str:
    values = [row.get("chip"), row.get("gpu"), row.get("campaign_hardware_accelerator_model")]
    return " ".join(
        json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value or "")
        for value in values
    ).casefold()


def _nested_machine_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        found = [
            nested
            for key, item in value.items()
            if str(key).casefold() in _NESTED_MACHINE_ID_FIELDS
            for nested in ([item] if isinstance(item, str) else [])
        ]
        return found + [nested for item in value.values() for nested in _nested_machine_ids(item)]
    if isinstance(value, list):
        return [nested for item in value for nested in _nested_machine_ids(item)]
    return []


def _machine_identity_values(
    row: Mapping[str, Any],
    tags: Mapping[str, Any],
    extras: Mapping[str, Any],
) -> list[Any]:
    direct = [row.get(field) for field in _MACHINE_ID_FIELDS] + [
        container.get(field.removeprefix("tag_"))
        for container in (tags, extras)
        for field in _MACHINE_ID_FIELDS
    ]
    nested = [
        identity for field in ("topology", "system_extra") for identity in _nested_machine_ids(row.get(field))
    ]
    return direct + nested


def _scrub_machine_identifiers(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for machine_id, label in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            value = value.replace(machine_id, label)
        return value
    if isinstance(value, dict):
        return {
            _scrub_machine_identifiers(key, replacements): _scrub_machine_identifiers(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub_machine_identifiers(item, replacements) for item in value]
    return value


def _machine_label_for_row(
    row: Mapping[str, Any],
    tags: Mapping[str, Any],
    extras: Mapping[str, Any],
    labels: Mapping[str, str],
) -> str | None:
    identity_values = _machine_identity_values(row, tags, extras)
    for value in identity_values:
        if isinstance(value, str) and value in labels:
            label = labels[value]
            if not _machine_label_is_valid(label):
                raise ValueError("machine label configuration contains an invalid entry")
            break
    else:
        valid_labels = [
            value for value in identity_values if isinstance(value, str) and _machine_label_is_valid(value)
        ]
        label = next(
            (value for value in valid_labels if value != APPLE_MACHINE_FALLBACK),
            valid_labels[0] if valid_labels else "",
        )

    hardware = _machine_label_hardware_text(row)
    if not label:
        if "m4 max" in hardware:
            label = APPLE_MACHINE_FALLBACK
        elif "rtx pro 6000" in hardware:
            label = "RTX PRO 6000 Max-Q 96GB"
        elif "rtx 4080 super" in hardware:
            label = "RTX 4080 SUPER 16GB"
    if not label:
        if any(value not in (None, "") for value in identity_values):
            raise ValueError("a machine identifier has no safe hardware label")
        return None

    if "m4 max" in label.casefold() and "m4 max" not in hardware:
        raise ValueError("a machine label does not match the row hardware")
    if label.startswith("MacBook Pro M4 Max "):
        memory = row.get("memory_gb")
        if memory is None or not label.endswith(f"{int(memory)}GB"):
            raise ValueError("a machine label does not match the row memory")
    if label.startswith("Mac Studio M4 Max ") and row.get("memory_gb") != 128:
        raise ValueError("a machine label does not match the row memory")
    if label.startswith("RTX PRO 6000") and "rtx pro 6000" not in hardware:
        raise ValueError("a machine label does not match the row hardware")
    if label.startswith("RTX 4080 SUPER") and "rtx 4080 super" not in hardware:
        raise ValueError("a machine label does not match the row hardware")
    accelerator_memory = row.get("campaign_hardware_accelerator_memory_gb")
    expected_accelerator_memory = (
        int(label.removesuffix("GB").split()[-1])
        if label.startswith(("RTX PRO 6000", "RTX 4080 SUPER"))
        else None
    )
    if (
        accelerator_memory is not None
        and expected_accelerator_memory is not None
        and float(accelerator_memory) != expected_accelerator_memory
    ):
        raise ValueError("a machine label does not match the row accelerator memory")
    return label


def normalize_legacy_rows(
    rows: list[dict[str, Any]],
    *,
    machine_labels: Mapping[str, str] | None = None,
    campaign_dimensions: Mapping[str, Any] | None = None,
    dimension_null_reasons: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Pad historical rows, retain dynamic fields, and project safe machine labels."""
    normalized_rows: list[dict[str, Any]] = []
    schema_names = set(PARQUET_ROW_SCHEMA.names)
    for legacy in rows:
        tags_json = legacy.get("tags_json")
        if tags_json is None:
            tags: dict[str, Any] = {}
        else:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
            if not isinstance(tags, dict):
                raise ValueError("historical tags_json must contain a JSON object")
            tags = dict(tags)
        tags.update(
            {
                key.removeprefix("tag_"): value
                for key, value in legacy.items()
                if key.startswith("tag_") and value is not None
            }
        )
        extras = {
            key: value
            for key, value in legacy.items()
            if key not in schema_names and not key.startswith("tag_") and value is not None
        }
        extra_json = legacy.get("extra_json")
        if extra_json is not None:
            decoded_extras = json.loads(extra_json) if isinstance(extra_json, str) else extra_json
            if not isinstance(decoded_extras, dict):
                raise ValueError("historical extra_json must contain a JSON object")
            extras = {**decoded_extras, **extras}
        row = {name: legacy.get(name) for name in PARQUET_ROW_SCHEMA.names}
        for column in JSON_STRING_COLUMNS:
            value = row[column]
            if value is not None and not isinstance(value, str):
                row[column] = json.dumps(value, sort_keys=True)
        system_extra = legacy.get("system_extra")
        row["system_extra"] = (
            system_extra
            if isinstance(system_extra, str)
            else json.dumps(system_extra, sort_keys=True)
            if system_extra is not None
            else "{}"
        )
        hardware_dimensions = (
            campaign_dimensions.get("hardware") if isinstance(campaign_dimensions, Mapping) else None
        )
        if isinstance(hardware_dimensions, Mapping) and hardware_dimensions.get("machine") is not None:
            machine = hardware_dimensions["machine"]
            if not _machine_label_is_valid(machine):
                raise ValueError("run-index machine must use a canonical hardware label")
            row["campaign_hardware_machine"] = machine
        label = _machine_label_for_row(row, tags, extras, machine_labels or {})
        if label:
            row["hostname"] = label
            for field in _MACHINE_ID_FIELDS:
                if row.get(field) is not None:
                    row[field] = label
                tag = field.removeprefix("tag_")
                tag_field = f"tag_{tag}"
                if row.get(tag_field) is not None:
                    row[tag_field] = label
                if tag in tags:
                    tags[tag] = label
                if tag in extras:
                    extras[tag] = label
            row["tags_json"] = json.dumps(tags, sort_keys=True)
            row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)
        row["tags_json"] = json.dumps(tags, sort_keys=True)
        row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)
        raw_machine_ids = {
            value
            for value in _machine_identity_values(row, tags, extras)
            if isinstance(value, str) and value and not _machine_label_is_valid(value)
        }
        if raw_machine_ids:
            if label is None:
                raise ValueError("a machine identifier has no safe hardware label")
            replacements = {
                machine_id: (machine_labels or {}).get(machine_id, label) for machine_id in raw_machine_ids
            }
            row = _scrub_machine_identifiers(row, replacements)
        _apply_campaign_dimension_projection(row, campaign_dimensions, dimension_null_reasons)
        software_dimensions = (
            campaign_dimensions.get("software") if isinstance(campaign_dimensions, Mapping) else None
        )
        if isinstance(software_dimensions, Mapping) and software_dimensions.get("engine_version") is not None:
            row["engine_version"] = software_dimensions["engine_version"]
        hardware = row.get("hardware")
        if isinstance(hardware, str):
            try:
                hardware = json.loads(hardware)
            except json.JSONDecodeError:
                hardware = None
        if isinstance(hardware, Mapping):
            machine = hardware.get("machine")
            if row["campaign_hardware_machine"] is None and machine is not None:
                if not _machine_label_is_valid(machine):
                    raise ValueError("hardware machine must use a canonical hardware label")
                row["campaign_hardware_machine"] = machine
            for field in ("accelerator_model", "accelerator_memory_gb"):
                column = campaign_dimension_column("hardware", field)
                if row[column] is None and hardware.get(field) is not None:
                    row[column] = hardware[field]
        normalized_rows.append(row)
    return normalized_rows


def canonical_shard_needs_refresh(
    existing: pa.Table,
    expected: pa.Table,
    *,
    machine_labels: Mapping[str, str] | None = None,
    prior_projection: pa.Table | None = None,
    campaign_dimensions: Mapping[str, Any] | None = None,
    dimension_null_reasons: Mapping[str, str] | None = None,
) -> bool:
    """Check whether a canonical shard can be refreshed from its retained source."""
    if not expected.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False):
        raise ValueError("expected table does not use the canonical schema")
    if existing.num_rows != expected.num_rows:
        raise RuntimeError("canonical shard row count differs from its source")
    if (
        existing.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
        and existing.to_pylist() == expected.to_pylist()
    ):
        return False
    if prior_projection is not None:
        if not prior_projection.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False):
            raise ValueError("prior projection does not use the canonical schema")
        if prior_projection.num_rows != expected.num_rows:
            raise RuntimeError("prior projection row count differs from its source")
        prior_rows = normalize_legacy_rows(existing.to_pylist(), machine_labels=machine_labels)
        prior_table = pa.Table.from_pylist(prior_rows, schema=PARQUET_ROW_SCHEMA)
        if prior_table.to_pylist() == prior_projection.to_pylist():
            return True

    refreshed = pa.Table.from_pylist(
        normalize_legacy_rows(
            existing.to_pylist(),
            machine_labels=machine_labels,
            campaign_dimensions=campaign_dimensions,
            dimension_null_reasons=dimension_null_reasons,
        ),
        schema=PARQUET_ROW_SCHEMA,
    )
    if refreshed.to_pylist() != expected.to_pylist():
        raise RuntimeError("canonical shard data differs from its normalized source")
    return True


# The dataset card's default configuration reads this one file. It sits below `data/`
# so the top-level shard globs and the original-shard filter never match it.
RECENT_SHARD_PATH = "data/recent/latest.parquet"
RECENT_WINDOW = datetime.timedelta(days=30)
_STAMPED_CANONICAL_SHARD = re.compile(
    r"data/run-canonical-run-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-[^/]+\.parquet"
)


def recent_shard_paths(
    paths: Iterable[str],
    *,
    window: datetime.timedelta = RECENT_WINDOW,
    excluded_paths: Iterable[str] = (),
) -> list[str]:
    """Timestamped canonical shards within ``window`` of the newest one, oldest first.

    The window ends at the newest shard rather than at the clock, so it names no calendar
    month and is never empty while any timestamped shard exists. Shards without a run
    timestamp in their name (rescue and bulk-export shards) are left out.
    """
    excluded = set(excluded_paths)
    stamped = {
        path: datetime.datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S")
        for path in paths
        if path not in excluded and (match := _STAMPED_CANONICAL_SHARD.fullmatch(path))
    }
    if not stamped:
        return []
    newest = max(stamped.values())
    kept = [path for path, moment in stamped.items() if newest - moment <= window]
    return sorted(kept, key=lambda path: (stamped[path], path))


def _conform_to_row_schema(table: pa.Table) -> pa.Table:
    unexpected = set(table.schema.names) - set(PARQUET_ROW_SCHEMA.names)
    if unexpected:
        raise ValueError(f"shard has columns outside the canonical schema: {', '.join(sorted(unexpected))}")
    if table.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False):
        return table
    return pa.Table.from_pylist(table.to_pylist(), schema=PARQUET_ROW_SCHEMA)


def recent_shard_table(tables: Iterable[pa.Table]) -> pa.Table:
    """Combine canonical shard tables into one table, newest run first.

    Rows are kept as published; a shard written before a column existed gains nulls for it.
    The sort is stable, so rows of one run keep their published order.
    """
    conformed = [_conform_to_row_schema(table) for table in tables]
    if not conformed:
        raise ValueError("no canonical shards to combine")
    return pa.concat_tables(conformed).sort_by([("timestamp", "descending")])
