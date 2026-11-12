"""Verify schema.json is itself valid and that canonical examples pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from mlx_benchmarks.dataset_schema import (
    CAMPAIGN_DIMENSION_TYPES,
    PARQUET_ROW_SCHEMA,
    STANDARD_CACHE_FIELD_DESCRIPTIONS,
    STANDARD_CACHE_FIELD_DIMENSIONS,
    STANDARD_CACHE_FIELD_TYPES,
    campaign_dimension_column,
)
from mlx_benchmarks.envelope import (
    EnvelopeValidationError,
    load_schema,
    validate_envelope,
)


def test_schema_is_valid_draft7() -> None:
    Draft7Validator.check_schema(load_schema())


def test_schema_declares_id_and_examples() -> None:
    schema = load_schema()
    assert schema.get("$id"), "schema must declare $id so consumers can pin to a canonical URL"
    assert schema.get("examples"), "schema should ship at least one example"


def test_standard_cache_fields_have_typed_campaign_dimensions_and_descriptions() -> None:
    assert set(STANDARD_CACHE_FIELD_TYPES) == set(STANDARD_CACHE_FIELD_DIMENSIONS)
    assert set(STANDARD_CACHE_FIELD_TYPES) == set(STANDARD_CACHE_FIELD_DESCRIPTIONS)
    assert len(STANDARD_CACHE_FIELD_TYPES) == 26

    for name, (category, field) in STANDARD_CACHE_FIELD_DIMENSIONS.items():
        assert field in CAMPAIGN_DIMENSION_TYPES[category], name
        assert STANDARD_CACHE_FIELD_TYPES[name] == CAMPAIGN_DIMENSION_TYPES[category][field]
        assert STANDARD_CACHE_FIELD_DESCRIPTIONS[name]


def test_moe_offload_dimensions_validate_mode_and_nonnegative_layer_count(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "campaign_dimensions": {"run": {"moe_cpu_offload": "ncmoe", "moe_cpu_layers": 15}},
    }
    validate_envelope(env)

    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "campaign_dimensions": {"run": {"moe_cpu_offload": "all"}}})
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "campaign_dimensions": {"run": {"moe_cpu_layers": -1}}})


def test_valid_envelope_passes(valid_envelope: dict) -> None:
    validate_envelope(valid_envelope)


@pytest.mark.parametrize("suite", ["spl", "tool-queries", "agentic", "triage", "homelab-access", "repo-qa"])
def test_domain_eval_suites_validate(valid_envelope: dict, suite: str) -> None:
    validate_envelope({**valid_envelope, "suite": suite})


def test_repo_qa_example_validates() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "envelope.repo-qa.json"
    validate_envelope(json.loads(example_path.read_text()))


def test_unknown_runtime_git_sha_can_be_null(valid_envelope: dict) -> None:
    validate_envelope({**valid_envelope, "git_sha": None})


def test_invalid_envelope_fails(invalid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError) as excinfo:
        validate_envelope(invalid_envelope)
    message = str(excinfo.value)
    # Known problems in the fixture: bad schema_version, bad timestamp, bad git_sha, bad trigger, bad suite, short system.
    assert "schema_version" in message or "suite" in message
    # Multiple errors collected, not just the first
    assert len(excinfo.value.errors) >= 2


def test_cluster_envelope_validates(cluster_envelope: dict) -> None:
    # A full two-node TB5 pipeline envelope: env_class + concurrency + serving +
    # system.topology all populated.
    validate_envelope(cluster_envelope)


def test_cluster_hardware_dimensions_validate(valid_envelope: dict) -> None:
    validate_envelope(
        {
            **valid_envelope,
            "campaign_dimensions": {
                "hardware": {
                    "machine": "2x Apple M4 Max 128GB (Thunderbolt RDMA)",
                    "interconnect": "Thunderbolt 5 RDMA",
                    "world_size": 2,
                    "host_ram_limit_gib": 187.890625,
                },
                "run": {
                    "kv_offload_backend": "none",
                    "moe_cpu_offload": "none",
                    "cache_workload": "none",
                },
            },
        }
    )


def test_new_optional_top_level_fields_validate(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "env_class": "under-load",
        "concurrency": 8,
        "serving": {"stack": "vllm-mlx", "endpoint_port": 8000, "served_model": "m"},
    }
    validate_envelope(env)


def test_campaign_dimensions_are_optional_and_accept_explicit_nulls(valid_envelope: dict) -> None:
    # Legacy envelopes remain valid without campaign-specific dimensions.
    validate_envelope(valid_envelope)

    env = {
        **valid_envelope,
        "campaign_dimensions": {
            "hardware": {"machine": "Mac Studio", "pcie_generation": None},
            "software": {
                "backend": "Metal",
                "gpu_architectures": None,
                "build_flags": {"ENABLE_METAL": True},
            },
            "speed": {"ttft_p50_ms": 14.5, "itl_p99_ms": None},
        },
        "dimension_null_reasons": {
            "hardware.pcie_generation": "N/A_SOC",
            "software.gpu_architectures": "N/A_MLX",
            "speed.itl_p99_ms": "NOT_REPORTED",
        },
    }
    validate_envelope(env)


def test_wave_campaign_dimensions_and_direct_serving_validate(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "suite": "throughput-serving",
        "serving": {"stack": "direct"},
        "results": [
            {
                **valid_envelope["results"][0],
                "tags": {"domain": "agentic"},
            }
        ],
        "campaign_dimensions": {
            "environment": {"ambient_temp_f": 72.5, "fan_duty_pct": {"gpu": 100, "mac": None}},
            "hardware": {"nautobot_device_id": None},
            "software": {"cuda_toolkit_version": "13.4.3"},
            "run": {
                "started_at": "2026-10-16T12:00:00Z",
                "ended_at": "2026-10-16T12:01:00Z",
                "patches": [{"id": "mtp", "revision": "abc1234", "enabled": True}],
                "knob_case": "wave2-mtp-on",
                "knobs": {
                    "num_speculative_tokens": 3,
                    "request_shape": {"prompt_tokens": 1000, "output_tokens": 2048},
                },
                "sampling": {"mode": "greedy"},
                "num_speculative_tokens": 3,
                "kv_pool_tokens": 32768,
                "max_running_requests": 8,
                "ple_offload_target": "ram",
                "load_time_s": 12.4,
            },
            "speed": {
                "mtp_accept_length": 2.5,
                "kv_cache_usage_peak_pct": 90,
                "prefix_cache_hit_rate": 0.4,
                "queue_time_p50_ms": 3.2,
                "queue_time_p99_ms": 10.5,
                "goodput_pct": 99,
                "c_at_p10_35tps": 8,
            },
            "resource": {"ple_bytes": 1024, "power_avg_w": None},
        },
        "dimension_null_reasons": {"resource.power_avg_w": "METRICS_URL_UNSET"},
    }

    validate_envelope(env)


def test_framework_eval_domain_rejects_unknown_value(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "results": [{**valid_envelope["results"][0], "tags": {"domain": "other"}}],
    }
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(env)


def test_stage0_system_load_phase_contract_validates(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "campaign": {"profile": "stage0-system-load"},
        "campaign_dimensions": {
            "hardware": {
                "machine": "Mac Studio M4 Max 128GB",
                "power_source": "ac",
                "power_mode": "automatic",
            },
            "software": {"macos_version": "26.6.2"},
            "run": {
                "concurrent_agents": 1,
                "load_phase": "serial",
                "phase_duration_seconds": 180,
                "closed_loop": True,
                "streaming": False,
                "embedding_items_per_request": 1,
                "embedding_input_tokens_per_request": 128,
            },
            "model": {"pipeline_tag": "feature-extraction"},
            "speed": {
                "embeddings_per_second": 2.5,
                "embedding_tokens_per_second": 320,
                "embedding_latency_p50_ms": 12,
                "embedding_latency_p95_ms": 18,
                "embedding_latency_p99_ms": 20,
            },
        },
    }

    validate_envelope(env)

    missing_power_mode = {
        **env,
        "campaign_dimensions": {
            **env["campaign_dimensions"],
            "hardware": {"machine": "Mac Studio M4 Max 128GB", "power_source": "ac"},
        },
    }
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(missing_power_mode)

    missing_pipeline_tag = {
        **env,
        "campaign_dimensions": {
            **env["campaign_dimensions"],
            "model": {},
        },
    }
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(missing_pipeline_tag)

    decision = {
        **env,
        "campaign_dimensions": {
            **env["campaign_dimensions"],
            "run": {
                "concurrent_agents": 4,
                "load_phase": "high_parallel",
                "phase_duration_seconds": 180,
                "closed_loop": True,
                "streaming": False,
            },
            "speed": {
                "requests_per_second": 2.5,
                "request_latency_p50_ms": 410,
                "request_latency_p95_ms": 620,
                "request_latency_p99_ms": 710,
            },
        },
    }
    validate_envelope(decision)


def test_campaign_dimensions_reject_unknown_fields(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "campaign_dimensions": {"hardware": {"invented_field": "value"}},
    }
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(env)


def test_blocked_run_can_preserve_unknown_required_system_values_as_null(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "cell_status": "aborted",
        "system": {"os": None, "chip": None, "memory_gb": None},
    }
    validate_envelope(env)


def test_campaign_dimension_fields_match_the_fixed_parquet_schema() -> None:
    dimension_groups = load_schema()["properties"]["campaign_dimensions"]["properties"]
    assert set(dimension_groups) == set(CAMPAIGN_DIMENSION_TYPES)
    parquet_columns = set(PARQUET_ROW_SCHEMA.names)
    for group, fields in CAMPAIGN_DIMENSION_TYPES.items():
        properties = dimension_groups[group]["properties"]
        assert set(properties) == set(fields)
        for field in fields:
            assert "null" in properties[field]["type"]
            assert campaign_dimension_column(group, field) in parquet_columns


def test_env_class_enum_is_enforced(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "env_class": "chaotic"})


def test_concurrency_minimum_is_enforced(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "concurrency": 0})


def test_reasoning_effort_accepts_any_family_naming(valid_envelope: dict) -> None:
    """Not an enum on purpose: families name their own levels, and normalising
    one family's name into another's merges arms that are not the same arm. The
    schema's job is to carry the value, not to referee the vocabulary."""
    for level in ("low", "medium", "high", "xhigh", "max", "off", "unstated", "think-harder"):
        validate_envelope({**valid_envelope, "reasoning_effort": level})


def test_reasoning_effort_must_be_a_string(valid_envelope: dict) -> None:
    # A bare boolean is how effort got collapsed in the agentic harness; the
    # schema refuses it so that shape cannot reach a shard.
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "reasoning_effort": True})


def test_serving_rejects_unknown_key(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "serving": {"stack": "x", "bogus": 1}})


def test_cloud_endpoint_envelope_validates(cloud_endpoint_envelope: dict) -> None:
    validate_envelope(cloud_endpoint_envelope)


def test_cloud_endpoint_envelope_requires_provider(valid_envelope: dict) -> None:
    envelope = {**valid_envelope, "model_kind": "cloud-endpoint"}
    envelope.pop("model_revision")
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(envelope)


def test_cloud_endpoint_envelope_rejects_hub_revision(valid_envelope: dict) -> None:
    envelope = {
        **valid_envelope,
        "model_kind": "cloud-endpoint",
        "model_provider": "example-cloud",
    }
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(envelope)


def test_model_provider_requires_cloud_endpoint_kind(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "model_provider": "example-cloud"})
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "model_kind": "local", "model_provider": "example-cloud"})


def test_model_kind_rejects_unknown_value(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "model_kind": "hosted"})


def test_topology_parallelism_enum_is_enforced(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "system": {**valid_envelope["system"], "topology": {"parallelism": "quantum"}},
    }
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(env)


def test_format_checker_rejects_non_iso_timestamp(valid_envelope: dict) -> None:
    """Targeted test: without format_checker=, jsonschema accepts any string
    for ``format: date-time``. The publisher contract — and the viewer's
    ``pd.to_datetime`` — requires real ISO-8601, so the validator must enforce it."""
    bad = dict(valid_envelope)
    bad["timestamp"] = "not-an-iso-date"
    with pytest.raises(EnvelopeValidationError) as excinfo:
        validate_envelope(bad)
    # Error must be scoped to the timestamp field specifically.
    paths = [list(e.absolute_path) for e in excinfo.value.errors]
    assert ["timestamp"] in paths
