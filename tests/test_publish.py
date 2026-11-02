"""Cover publish.py without touching the network."""

from __future__ import annotations

import io
import json
import logging

import pytest

import mlx_benchmarks.publish as publish_module
from mlx_benchmarks.dataset_schema import (
    CAMPAIGN_DIMENSION_TYPES,
    PARQUET_ROW_SCHEMA,
    campaign_dimension_column,
)
from mlx_benchmarks.envelope import EnvelopeValidationError
from mlx_benchmarks.publish import (
    PublishError,
    canonical_shard_path,
    envelope_to_rows,
    publish,
    rows_to_parquet,
    slugify,
    target_path,
)


def test_slugify_strips_special_chars() -> None:
    assert slugify("mlx-community/Qwen3.5-9B-MLX-4bit") == "mlx-community-qwen3-5-9b-mlx-4bit"
    assert slugify("openrouter/openai/gpt-5-mini") == "openrouter-openai-gpt-5-mini"


def test_target_path_format(valid_envelope: dict) -> None:
    path = target_path(valid_envelope)
    assert path.startswith("data/run-canonical-")
    assert path.endswith(".parquet")
    # No colons (filesystem-hostile on Windows/CI, and HF commit paths)
    assert ":" not in path
    # Contains git_sha and suite
    assert valid_envelope["git_sha"] in path
    assert valid_envelope["suite"] in path


def test_canonical_shard_path_covers_run_and_aggregate_files() -> None:
    assert canonical_shard_path("data/run-2026-07-01.parquet") == (
        "data/run-canonical-run-2026-07-01.parquet"
    )
    assert canonical_shard_path("data/train-00000-of-00001.parquet") == (
        "data/run-canonical-train-00000-of-00001.parquet"
    )


def test_target_path_includes_payload_hash(valid_envelope: dict) -> None:
    bare = target_path(valid_envelope)
    stamped_a = target_path(valid_envelope, payload=b"alpha")
    stamped_b = target_path(valid_envelope, payload=b"beta")
    # Same prefix, distinct suffixes: re-runs producing different bytes in the
    # same second MUST NOT collide.
    assert stamped_a != stamped_b
    assert stamped_a != bare
    assert stamped_b != bare
    assert stamped_a.startswith(bare.removesuffix(".parquet"))


def test_sample_slice_cells_publish_to_distinct_paths(valid_envelope: dict) -> None:
    paths = []
    for sample_slice_index in (0, 1):
        envelope = {
            **valid_envelope,
            "campaign_dimensions": {"run": {"sample_slice_index": sample_slice_index}},
        }
        payload = rows_to_parquet(envelope_to_rows(envelope))
        paths.append(target_path(envelope, payload=payload))

    assert paths[0] != paths[1]


def test_envelope_to_rows_explodes_results(valid_envelope: dict) -> None:
    rows = envelope_to_rows(valid_envelope)
    assert len(rows) == len(valid_envelope["results"])
    row = rows[0]
    # Base envelope fields propagated
    assert row["suite"] == "reasoning"
    assert row["model"] == "mlx-community/Qwen3.5-9B-MLX-4bit"
    # Tags exploded with tag_ prefix
    assert row["tag_lm_eval_key"] == "exact_match,flexible-extract"
    # Duration promoted to top-level column
    assert row["duration_seconds"] == 123.4


def test_model_task_fields_reach_every_parquet_row(valid_envelope: dict) -> None:
    envelope = {
        **valid_envelope,
        "model_task": "feature-extraction",
        "model_task_source": "model_card",
    }

    rows = envelope_to_rows(envelope)

    assert all(row["model_task"] == "feature-extraction" for row in rows)
    assert all(row["model_task_source"] == "model_card" for row in rows)


def test_rows_to_parquet_roundtrip(valid_envelope: dict) -> None:
    import pyarrow.parquet as pq

    rows = envelope_to_rows(valid_envelope)
    parquet_bytes = rows_to_parquet(rows)
    assert parquet_bytes[:4] == b"PAR1"  # parquet magic
    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert table.num_rows == len(rows)
    assert table.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)


def test_campaign_dimensions_keep_a_fixed_nullable_parquet_schema(valid_envelope: dict) -> None:
    import pyarrow.parquet as pq

    legacy_rows = envelope_to_rows(valid_envelope)
    enriched = {
        **valid_envelope,
        "campaign_dimensions": {
            "hardware": {
                "machine": "2x Apple M4 Max 128GB (Thunderbolt RDMA)",
                "interconnect": "Thunderbolt 5 RDMA",
                "world_size": 2,
                "pcie_generation": None,
            },
            "software": {
                "gpu_architectures": ["sm_120"],
                "build_flags": {"GGML_CUDA": True, "CMAKE_BUILD_TYPE": "Release"},
                "flash_attention": True,
            },
            "model": {
                "file_sha256": "a" * 64,
            },
            "speed": {"ttft_p50_ms": 14.5},
        },
        "dimension_null_reasons": {"hardware.pcie_generation": "N/A_SOC"},
    }
    enriched_rows = envelope_to_rows(enriched)
    legacy_table = pq.read_table(io.BytesIO(rows_to_parquet(legacy_rows)))
    enriched_table = pq.read_table(io.BytesIO(rows_to_parquet(enriched_rows)))

    assert legacy_table.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
    assert enriched_table.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
    legacy = legacy_table.to_pylist()[0]
    row = enriched_table.to_pylist()[0]
    dimension_columns = {
        campaign_dimension_column(category, field)
        for category, fields in CAMPAIGN_DIMENSION_TYPES.items()
        for field in fields
    }
    assert dimension_columns <= set(legacy)
    assert dimension_columns <= set(row)
    assert all(legacy[column] is None for column in dimension_columns)
    assert row["campaign_hardware_machine"] == "2x Apple M4 Max 128GB (Thunderbolt RDMA)"
    assert row["campaign_hardware_interconnect"] == "Thunderbolt 5 RDMA"
    assert row["campaign_hardware_world_size"] == 2
    assert legacy["campaign_hardware_interconnect"] is None
    assert legacy["campaign_hardware_world_size"] is None
    assert row["campaign_hardware_pcie_generation"] is None
    assert json.loads(row["campaign_software_gpu_architectures_json"]) == ["sm_120"]
    assert json.loads(row["campaign_software_build_flags_json"]) == {
        "CMAKE_BUILD_TYPE": "Release",
        "GGML_CUDA": True,
    }
    assert row["campaign_model_file_sha256"] == "a" * 64
    assert json.loads(row["campaign_dimension_null_reasons_json"]) == {"hardware.pcie_generation": "N/A_SOC"}


def test_cache_configuration_campaign_dimensions_reach_parquet(valid_envelope: dict) -> None:
    import pyarrow.parquet as pq

    envelope = {
        **valid_envelope,
        "campaign_dimensions": {
            "hardware": {"host_ram_limit_gib": 48.0},
            "run": {
                "prefix_cache": True,
                "kv_offload_backend": "none",
                "kv_offload_cpu_gib": 0.0,
                "kv_offload_disk_gib": 0.0,
                "weight_cpu_offload_gib": 0.0,
                "moe_cpu_offload": "cpumoe",
                "moe_cpu_layers": 40,
                "llamacpp_cache_ram_mib": 8192,
                "gpu_memory_utilization": 0.88,
                "max_num_batched_tokens": 8192,
                "kv_pool_tokens": 123456,
                "output_tokens": 512,
            },
        },
    }

    table = pq.read_table(io.BytesIO(rows_to_parquet(envelope_to_rows(envelope))))
    row = table.to_pylist()[0]

    assert row[campaign_dimension_column("hardware", "host_ram_limit_gib")] == 48.0
    assert row[campaign_dimension_column("run", "prefix_cache")] is True
    assert row[campaign_dimension_column("run", "kv_offload_backend")] == "none"
    assert row[campaign_dimension_column("run", "moe_cpu_offload")] == "cpumoe"
    assert row[campaign_dimension_column("run", "moe_cpu_layers")] == 40
    assert row[campaign_dimension_column("run", "kv_offload_cpu_gib")] == 0.0
    assert row[campaign_dimension_column("run", "kv_offload_disk_gib")] == 0.0
    assert row[campaign_dimension_column("run", "weight_cpu_offload_gib")] == 0.0
    assert row[campaign_dimension_column("run", "llamacpp_cache_ram_mib")] == 8192
    assert row[campaign_dimension_column("run", "gpu_memory_utilization")] == 0.88
    assert row[campaign_dimension_column("run", "max_num_batched_tokens")] == 8192
    assert row[campaign_dimension_column("run", "kv_pool_tokens")] == 123456
    assert row[campaign_dimension_column("run", "output_tokens")] == 512


def _stage0_envelope(valid_envelope: dict, pipeline_tag: str) -> dict:
    return {
        **valid_envelope,
        "campaign": {"profile": "stage0-system-load"},
        "campaign_dimensions": {
            "hardware": {
                "machine": "MacBook Pro M4 Max 128GB",
                "power_source": "ac",
                "power_mode": "automatic",
            },
            "software": {"macos_version": "26.6.2"},
            "model": {"pipeline_tag": pipeline_tag},
            "run": {
                "concurrent_agents": 4,
                "load_phase": "high_parallel",
                "phase_duration_seconds": 180,
                "closed_loop": True,
                "streaming": False,
                "embedding_items_per_request": 1,
                "embedding_input_tokens_per_request": 256,
            },
            "speed": {
                "embeddings_per_second": 125,
                "embedding_tokens_per_second": 32000,
                "embedding_latency_p50_ms": 7.5,
                "embedding_latency_p95_ms": 12.0,
                "embedding_latency_p99_ms": 15.0,
            },
        },
        "results": [
            {
                "name": "embedding-throughput",
                "metric": "embeddings_per_second",
                "value": 125,
                "unit": "vectors/s",
            },
            {
                "name": "embedding-tokens",
                "metric": "embedding_tokens_per_second",
                "value": 32000,
                "unit": "tokens/s",
            },
        ],
    }


def test_stage0_phase_dimensions_repeat_on_each_published_result_row(
    valid_envelope: dict,
    published_metadata: dict,
    mock_hf_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    phase = _stage0_envelope(valid_envelope, "text-generation")

    rows = envelope_to_rows(phase)
    table = pq.read_table(io.BytesIO(rows_to_parquet(rows)))

    assert table.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
    assert table.num_rows == 2
    for row in table.to_pylist():
        assert row["os"] == phase["system"]["os"]
        assert row["chip"] == phase["system"]["chip"]
        assert row["memory_gb"] == phase["system"]["memory_gb"]
        assert row["campaign_hardware_machine"] == "MacBook Pro M4 Max 128GB"
        assert row["campaign_hardware_power_source"] == "ac"
        assert row["campaign_hardware_power_mode"] == "automatic"
        assert row["campaign_software_macos_version"] == "26.6.2"
        assert row["campaign_run_load_phase"] == "high_parallel"
        assert row["campaign_run_phase_duration_seconds"] == 180
        assert row["campaign_run_closed_loop"] is True
        assert row["campaign_run_embedding_input_tokens_per_request"] == 256
        assert row["campaign_model_pipeline_tag"] == "text-generation"
        assert row["campaign_speed_embedding_tokens_per_second"] == 32000
        assert row["campaign_speed_embedding_latency_p95_ms"] == 12.0

    published_rows: list[dict] = []
    original = publish_module.rows_to_parquet

    def capture(rows: list[dict]) -> bytes:
        published_rows.extend(rows)
        return original(rows)

    monkeypatch.setattr(publish_module, "rows_to_parquet", capture)
    publish(phase, published_metadata=published_metadata, dry_run=True)
    assert len(published_rows) == 2
    assert all(row["pipeline_tag"] == "text-generation" for row in published_rows)
    assert all(row["model_task"] == "text-generation" for row in published_rows)


def test_stage0_publish_rejects_catalog_pipeline_tag_mismatch(
    valid_envelope: dict,
    published_metadata: dict,
    mock_hf_registry: None,
) -> None:
    phase = _stage0_envelope(valid_envelope, "feature-extraction")

    with pytest.raises(PublishError, match="must equal the selected catalog pipeline_tag"):
        publish(phase, published_metadata=published_metadata, dry_run=True)


def test_rows_to_parquet_rejects_empty() -> None:
    with pytest.raises(PublishError, match="No result rows"):
        rows_to_parquet([])


def test_publish_dry_run_returns_path(
    valid_envelope: dict, published_metadata: dict, mock_hf_registry: None
) -> None:
    path = publish(valid_envelope, published_metadata=published_metadata, dry_run=True)
    # The returned path carries a content-addressed suffix when payload is real.
    assert path.startswith(target_path(valid_envelope).removesuffix(".parquet"))


def test_publish_writes_hub_fields_as_named_parquet_columns(
    valid_envelope: dict,
    published_metadata: dict,
    mock_hf_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written_rows: list[dict] = []
    original = publish_module.rows_to_parquet

    def capture(rows: list[dict]) -> bytes:
        written_rows.extend(rows)
        return original(rows)

    monkeypatch.setattr(publish_module, "rows_to_parquet", capture)
    publish(valid_envelope, published_metadata=published_metadata, dry_run=True)

    [row] = written_rows
    assert row["model_id"] == valid_envelope["model"]
    assert row["model_revision"] == valid_envelope["model_revision"]
    assert row["pipeline_tag"] == "text-generation"
    assert row["model_task"] == row["pipeline_tag"]
    assert row["model_task_source"] == "model_card"
    assert json.loads(row["dtype"]) == {"F16": 1000}
    assert row["dataset_id"] == published_metadata["dataset_id"]
    assert row["engine"] == published_metadata["engine"]
    assert "published_result_json" not in row


def test_publish_performance_only_rows_do_not_claim_an_eval_result(
    valid_envelope: dict,
    published_metadata: dict,
    mock_hf_registry: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    dataset_fields = (
        "dataset_id",
        "dataset_task_id",
        "dataset_revision",
        "evaluation_framework",
        "config",
        "split",
    )
    metadata = {
        **published_metadata,
        **dict.fromkeys(dataset_fields),
        "dimension_null_reasons": {
            **published_metadata["dimension_null_reasons"],
            **dict.fromkeys(dataset_fields, "not_applicable"),
        },
    }

    publish(valid_envelope, published_metadata=metadata, dry_run=True)

    assert "0 HF eval result record(s)" in caplog.text


def test_publish_refuses_invalid_envelope(invalid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        publish(invalid_envelope, dry_run=True)


def test_public_publish_refuses_serving_endpoint_port(valid_envelope: dict, mock_hf_registry: None) -> None:
    envelope = {**valid_envelope, "serving": {"endpoint_port": "legacy field"}}
    with pytest.raises(PublishError, match="retained for legacy rows"):
        publish(envelope, dry_run=True)


def test_envelope_requires_model_task_source_with_task(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        publish({**valid_envelope, "model_task": "feature-extraction"}, dry_run=True)


def test_publish_requires_metadata(valid_envelope: dict) -> None:
    with pytest.raises(PublishError, match="--published-metadata is required"):
        publish(valid_envelope, dry_run=True)


def test_publish_refuses_a_silent_null_score_field(
    valid_envelope: dict,
    published_metadata: dict,
    mock_hf_registry: None,
) -> None:
    metadata = {**published_metadata, "prompt_chars": None}
    with pytest.raises(PublishError, match="prompt_chars is null"):
        publish(valid_envelope, published_metadata=metadata, dry_run=True)


def test_publish_refuses_nested_host_identifiers(
    valid_envelope: dict,
    published_metadata: dict,
    mock_hf_registry: None,
) -> None:
    metadata = {**published_metadata, "hardware": {"system": {"node_name": "REDACTED"}}}
    with pytest.raises(PublishError, match="must not include host or network identifiers"):
        publish(valid_envelope, published_metadata=metadata, dry_run=True)


def test_envelope_to_rows_flattens_all_system_fields(valid_envelope: dict) -> None:
    # Every system field becomes its own column, not just os/chip/memory_gb.
    [row] = envelope_to_rows(valid_envelope)
    assert row["kernel"] == "25.4.0"
    assert row["lm_eval_version"] == "0.4.11"
    assert row["python_version"] == "3.11.9"
    assert row["hostname"] is None


def test_envelope_to_rows_flattens_topology_and_new_top_level_fields(cluster_envelope: dict) -> None:
    cluster_envelope["system"]["topology"] = {
        "world_size": 2,
        "nodes": [{"hostname": "REDACTED", "host_ip": "REDACTED", "chip": "GPU"}],
    }
    [row] = envelope_to_rows(cluster_envelope)
    # Nested topology can't be a scalar cell, so it rides as a JSON string.
    topology = json.loads(row["topology"])
    assert topology["world_size"] == 2
    assert topology["nodes"] == [{"chip": "GPU"}]
    # New top-level fields land as their own columns.
    assert row["env_class"] == "isolated"
    assert row["concurrency"] == 4
    # An envelope field that publish.py does not list becomes a column nowhere:
    # the envelope carries it and every reader of the dataset misses it.
    assert row["reasoning_effort"] == "xhigh"
    assert json.loads(row["serving"])["endpoint_port"] == 8080
    # The JSON-string columns still serialize to parquet.
    assert rows_to_parquet([row])[:4] == b"PAR1"


def test_envelope_to_rows_promotes_tok_per_sec_fields(valid_envelope: dict) -> None:
    """The new optional throughput fields land as top-level parquet columns next
    to ``duration_seconds`` so dashboards can graph tok/s without crawling tags."""
    rows = envelope_to_rows(valid_envelope)
    row = rows[0]
    assert row["prompt_tokens_per_second"] == 1850.0
    assert row["decode_tokens_per_second"] == 52.3
    assert row["total_tokens_per_second"] == 1902.3


def test_envelope_to_rows_keeps_absent_optional_fields_as_null(valid_envelope: dict) -> None:
    """Every optional field remains in each row so shard schemas stay stable."""
    no_throughput = {
        **valid_envelope,
        "results": [
            {
                "name": "lol",
                "metric": "exact_match_flexible",
                "value": 0.5,
                "unit": "ratio",
            }
        ],
    }
    [row] = envelope_to_rows(no_throughput)
    assert row["decode_tokens_per_second"] is None
    assert row["prompt_tokens_per_second"] is None
    assert row["total_tokens_per_second"] is None
    assert row["first_token_latency_ms"] is None
    assert row["peak_rss_mb"] is None


def test_envelope_to_rows_normalizes_columns_across_mixed_rows(valid_envelope: dict) -> None:
    """When SOME results carry throughput but others don't, every row must
    still have the throughput columns (with None where absent). Otherwise
    PyArrow's first-row schema inference would silently drop the column when
    the first row happens to lack it."""
    mixed = {
        **valid_envelope,
        "results": [
            {  # no throughput
                "name": "a",
                "metric": "exact_match_flexible",
                "value": 0.5,
                "unit": "ratio",
            },
            {  # throughput present
                "name": "b",
                "metric": "exact_match_flexible",
                "value": 0.6,
                "unit": "ratio",
                "duration_seconds": 10.0,
                "decode_tokens_per_second": 42.0,
            },
        ],
    }
    rows = envelope_to_rows(mixed)
    assert len(rows) == 2
    assert rows[0].keys() == rows[1].keys()
    assert set(rows[0]) == set(PARQUET_ROW_SCHEMA.names)
    for row in rows:
        assert "decode_tokens_per_second" in row
        assert "duration_seconds" in row
    assert rows[0]["decode_tokens_per_second"] is None
    assert rows[1]["decode_tokens_per_second"] == 42.0


def test_unlisted_tag_keys_do_not_change_the_parquet_columns(valid_envelope: dict) -> None:
    envelope = {
        **valid_envelope,
        "results": [
            {
                **valid_envelope["results"][0],
                "tags": {"future_dimension": "kept"},
            }
        ],
    }
    [row] = envelope_to_rows(envelope)
    assert "tag_future_dimension" not in row
    assert json.loads(row["tags_json"]) == {"future_dimension": "kept"}
