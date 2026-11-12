"""Chart builders return Plotly figures, never raise, even on empty data."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Add the space/ directory to sys.path before resolving app via importlib so
# ruff's top-of-file import rule (E402) stays satisfied without a suppression.
SPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_ROOT))
app = importlib.import_module("app")

SAMPLE_ROWS = [
    {
        "timestamp": "2026-04-24T18:30:00Z",
        "suite": "reasoning",
        "name": "gsm8k_cot_zeroshot",
        "metric": "exact_match_flexible",
        "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "value": 0.8,
    },
    {
        "timestamp": "2026-04-24T19:00:00Z",
        "suite": "reasoning",
        "name": "gsm8k_cot_zeroshot",
        "metric": "exact_match_flexible",
        "model": "mlx-community/gemma-4-e4b-it-4bit",
        "value": 0.6,
    },
]


def _sample_df() -> pd.DataFrame:
    df = pd.DataFrame(SAMPLE_ROWS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["source_path"] = ["data/scored-a.parquet", "data/scored-b.parquet"]
    df = app.add_evidence_metadata(df, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    df["model_short"] = df["model"].apply(app.short_model)
    return df


def test_host_and_engine_filters_are_derived_from_row_values() -> None:
    rows = pd.DataFrame(
        [
            {
                "timestamp": "2026-10-16T12:00:00Z",
                "suite": "throughput-serving",
                "model": "m/a",
                "host": "host-a",
                "engine": "vLLM",
                "allocated_context": None,
                "context_tokens": None,
                "prompt_tokens": None,
                "output_budget_tokens": None,
                "completion_state": None,
            },
            {
                "timestamp": "2026-10-16T12:01:00Z",
                "suite": "throughput-serving",
                "model": "m/b",
                "host": "host-b",
                "engine": "SGLang",
                "allocated_context": None,
                "context_tokens": None,
                "prompt_tokens": None,
                "output_budget_tokens": None,
                "completion_state": None,
            },
        ]
    )
    rows = app.add_evidence_metadata(rows, {}, None)

    assert app.row_choices(rows, "host") == ["host-a", "host-b"]
    assert app.row_choices(rows, "engine") == ["SGLang", "vLLM"]


def test_short_model_handles_nulls() -> None:
    assert app.short_model(None) == "unknown model"
    assert app.short_model("") == "unknown model"


def test_canonical_paths_resolve_to_original_run_index_paths() -> None:
    base = f"hf://{app.DATASET}/"
    assert app._dataset_path(f"{base}data/run-canonical-run-2026-07-01.parquet") == (
        "data/run-2026-07-01.parquet"
    )
    assert app._dataset_path(f"{base}data/run-canonical-train-00000-of-00001.parquet") == (
        "data/train-00000-of-00001.parquet"
    )


def test_empty_data_returns_annotated_figure() -> None:
    fig = app.bar_chart(app.empty_data(), "reasoning", "gsm8k_cot_zeroshot", "exact_match_flexible")
    assert isinstance(fig, go.Figure)
    # Annotation present when no data
    assert len(fig.layout.annotations) == 1
    assert "No data" in fig.layout.annotations[0].text


def test_bar_chart_renders_with_rows() -> None:
    df = _sample_df()
    fig = app.bar_chart(df, "reasoning", "gsm8k_cot_zeroshot", "exact_match_flexible")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # single trace
    # Two models => two bars
    bar = fig.data[0]
    assert len(bar.y) == 2


def test_trend_chart_renders_with_rows() -> None:
    df = _sample_df()
    fig = app.trend_chart(
        df,
        "reasoning",
        "gsm8k_cot_zeroshot",
        "exact_match_flexible",
        models=df["series_key"].tolist(),
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_summary_table_returns_dataframe() -> None:
    df = _sample_df()
    pivot = app.summary_table(df, "reasoning", "gsm8k_cot_zeroshot", "exact_match_flexible")
    assert isinstance(pivot, pd.DataFrame)
    assert "Evidence" in pivot.columns or pivot.empty


def test_run_metric_lookup_matches_latest_companion_metrics_for_exact_run() -> None:
    base = {
        "suite": "throughput",
        "name": "benchmark_serving",
        "model": "model-a",
        "machine": "RTX PRO 6000",
        "source_path": "data/pro.parquet",
    }
    records = []
    for timestamp, metric, value, unit, machine, source_path in (
        ("2026-10-01T00:00:00Z", "throughput_output_toks_per_s", 125.0, "tok/s", None, None),
        ("2026-10-03T00:00:00Z", "output_tokens_per_second", 900.0, "tok/s", None, None),
        ("2026-10-01T00:00:00Z", "ttft_p50_ms", 240.0, "ms", None, None),
        ("2026-10-02T00:00:00Z", "ttft_p50_ms", 200.0, "ms", None, None),
        ("2026-10-04T00:00:00Z", "ttft_p50_ms", 5.0, "ms", "RTX 4080", None),
        ("2026-10-04T00:00:00Z", "throughput_output_toks_per_s", 1100.0, "tok/s", None, "data/other.parquet"),
    ):
        row = {
            **base,
            "timestamp": timestamp,
            "metric": metric,
            "value": value,
            "unit": unit,
            "display_unit": unit,
        }
        if machine is not None:
            row["machine"] = machine
        if source_path is not None:
            row["source_path"] = source_path
        records.append(row)

    rows = pd.DataFrame(records)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    target = rows.iloc[0]
    lookup = app._run_metric_lookup(
        rows,
        ("throughput_output_toks_per_s", "output_tokens_per_second", "ttft_p50_ms"),
    )

    output_aliases = ("throughput_output_toks_per_s", "output_tokens_per_second")
    ttft_aliases = ("ttft_p50_ms",)
    assert (
        app._run_metric_value(rows, target, output_aliases, lookup)
        == app._run_metric_value(rows, target, output_aliases)
        == "125.0 tok/s"
    )
    assert (
        app._run_metric_value(rows, target, ttft_aliases, lookup)
        == app._run_metric_value(rows, target, ttft_aliases)
        == "200.00 ms"
    )


def test_short_model_strips_common_prefixes() -> None:
    assert app.short_model("mlx-community/Qwen3.5-9B-MLX-4bit") == "Qwen3.5-9B-MLX-4bit"
    assert app.short_model("openrouter/openai/gpt-5-mini") == "openrouter/gpt-5-mini"
    assert app.short_model("plain-name") == "plain-name"


def test_normalize_rows_coalesces_layouts_and_drops_non_measurements() -> None:
    raw = pd.DataFrame(
        [
            # Flat layout, real measurement — kept as-is.
            {"suite": "reasoning", "model": "m/a", "name": "gsm8k", "metric": "exact_match", "value": 0.7},
            # Nested layout (older shards) — must be surfaced via coalescing.
            {
                "suite": "tool-calling",
                "model": "m/b",
                "metric_name": "should-call-tool",
                "metric_metric": "accuracy",
                "metric_value": 0.9,
            },
            # Skipped CI run (no MLX server) — dropped.
            {"suite": "code-accuracy", "model": "m/c", "skipped": True, "metric_value": None},
            # No measurement in either layout — dropped.
            {"suite": "framework-eval", "model": "m/d", "name": None, "value": None},
        ]
    )
    out = app.normalize_rows(raw)
    assert set(out["suite"]) == {"reasoning", "tool-calling"}
    surfaced = out[out["suite"] == "tool-calling"].iloc[0]
    assert surfaced["name"] == "should-call-tool"
    assert surfaced["metric"] == "accuracy"
    assert surfaced["value"] == 0.9


def test_unindexed_current_rows_are_experimental() -> None:
    df = pd.DataFrame(
        [{"timestamp": "2026-08-25T00:00:00Z", "model": "m/a", "source_path": "data/new.parquet"}]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = app.add_evidence_metadata(df, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert out.loc[0, "evidence_status"] == "experimental"
    assert app.evidence_view(out).empty


def test_mtp_run_index_retains_all_historical_shards_as_non_scored() -> None:
    index_path = SPACE_ROOT.parent / "metadata" / "run-index-v1.json"
    entries = json.loads(index_path.read_text())["runs"]
    # The shard files live in the published dataset, not this repo, so the
    # index is checked for non-empty, one entry per shard path, and the shard name shape.
    paths = [entry["path"] for entry in entries]
    assert paths
    assert len(set(paths)) == len(paths)
    assert all(path.startswith("data/run-") and path.endswith(".parquet") for path in paths)
    historical = [entry for entry in entries if entry["path"] < "data/run-2026-09"]
    assert {entry["status"] for entry in historical} <= {"experimental", "recovered"}
    assert all(entry["caveat"] for entry in historical)
    assert {entry["status"] for entry in entries} <= {"scored", "experimental", "invalid", "recovered"}
    assert all(entry["caveat"] for entry in entries if entry["status"] == "invalid")
    assert {entry["context_band"] for entry in entries} >= {"64k", "128k"}


def test_bar_chart_keeps_variants_in_separate_series() -> None:
    df = _sample_df()
    duplicate = df.iloc[[0]].copy()
    duplicate["variant"] = "MTP default"
    duplicate["series_key"] = " | ".join(
        [
            str(duplicate.iloc[0]["model"]),
            "unknown",
            "unknown",
            "MTP default",
            "unspecified",
            "unspecified",
            "unstated",
        ]
    )
    fig = app.bar_chart(
        pd.concat([df, duplicate], ignore_index=True),
        "reasoning",
        "gsm8k_cot_zeroshot",
        "exact_match_flexible",
    )
    assert len(fig.data[0].y) == 3


def test_two_reasoning_efforts_are_two_series() -> None:
    """Effort is part of the arm, so the same weights at `high` and at `xhigh`
    must not collapse into one bar — `groupby("series_key").last()` would keep
    only whichever ran later and silently drop the comparison."""
    rows = pd.DataFrame([SAMPLE_ROWS[0], SAMPLE_ROWS[0]])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["source_path"] = ["data/a.parquet", "data/b.parquet"]
    rows["reasoning_effort"] = ["high", "xhigh"]

    out = app.add_evidence_metadata(rows, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert out["series_key"].nunique() == 2
    assert all("high" in key for key in out["series_key"])


def test_rows_without_the_column_still_get_a_series_key() -> None:
    """Every shard published before the field existed lacks the column entirely;
    reading it must default rather than raise, and label as unstated rather than
    claiming an effort nobody recorded."""
    rows = pd.DataFrame(SAMPLE_ROWS)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["source_path"] = ["data/a.parquet", "data/b.parquet"]
    assert "reasoning_effort" not in rows.columns

    out = app.add_evidence_metadata(rows, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert all(key.endswith("unstated") for key in out["series_key"])


def test_cloud_and_local_rows_with_the_same_model_name_stay_two_series() -> None:
    """A cloud endpoint and a local build can serve the same model name; the
    declared kind, provider, and provider-qualified model id must keep them as
    two series, so the summary's latest-row-per-key keeps both rows."""
    rows = pd.DataFrame([SAMPLE_ROWS[0], SAMPLE_ROWS[0]])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["source_path"] = ["data/cloud.parquet", "data/local.parquet"]
    rows["model_kind"] = ["cloud-endpoint", None]
    rows["model_provider"] = ["together", None]
    rows["model_id"] = ["together/Qwen3.5-9B-MLX-4bit", "mlx-community/Qwen3.5-9B-MLX-4bit"]

    out = app.add_evidence_metadata(rows, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert out["series_key"].nunique() == 2

    pivot = app.summary_table(out, "reasoning", SAMPLE_ROWS[0]["name"], SAMPLE_ROWS[0]["metric"])
    assert len(pivot) == 2


def test_old_shard_rows_without_cloud_columns_keep_their_series_key() -> None:
    """Shards published before cloud identity existed lack the columns entirely;
    missing must read as null — the same key an explicit-null row gets — so
    every pre-cloud series keeps its previous composition."""
    rows = pd.DataFrame([SAMPLE_ROWS[0], SAMPLE_ROWS[0]])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["source_path"] = ["data/a.parquet", "data/b.parquet"]
    rows["model_kind"] = None
    rows["model_provider"] = None
    rows["model_id"] = None
    with_nulls = app.add_evidence_metadata(rows, {}, pd.Timestamp("2026-08-25T00:00:00Z"))

    without = rows.drop(columns=["model_kind", "model_provider", "model_id"])
    without_columns = app.add_evidence_metadata(without, {}, pd.Timestamp("2026-08-25T00:00:00Z"))

    assert without_columns["series_key"].tolist() == with_nulls["series_key"].tolist()
    assert with_nulls["series_key"].nunique() == 1


def _workload_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                **SAMPLE_ROWS[0],
                "unit": "tok/s",
                "hostname": "Machine A",
                "chip": "Accelerator A",
                "engine": "MLX",
                "tag_backend": "Metal",
                "quantization": "4-bit",
                "concurrency": 2,
                "tag_configured_window_tokens": 8192,
                "tag_context_tokens_actual": 4096,
                "tag_prompt_tokens": 1024,
                "tag_max_gen_toks": 128,
                "tag_truncated_rate": 0,
                "source_path": "data/first.parquet",
                "_row_id": "first:0",
            },
            {
                **SAMPLE_ROWS[0],
                "unit": "tok/s",
                "hostname": "Machine A",
                "chip": "Accelerator A",
                "engine": "MLX",
                "tag_backend": "Metal",
                "quantization": "4-bit",
                "concurrency": 2,
                "tag_configured_window_tokens": 8192,
                "tag_context_tokens_actual": 16384,
                "tag_prompt_tokens": 8192,
                "tag_max_gen_toks": 256,
                "tag_truncated_rate": 0,
                "source_path": "data/second.parquet",
                "_row_id": "second:0",
            },
        ]
    )
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    return rows


def test_workload_budgets_are_part_of_comparison_identity() -> None:
    rows = app.add_evidence_metadata(
        _workload_rows(),
        {},
        pd.Timestamp("2026-08-25T00:00:00Z"),
    )
    assert rows["workload_complete"].all()
    assert rows["series_key"].nunique() == 2
    assert rows["prompt_tokens"].tolist() == ["1024", "8192"]
    assert rows["output_budget_tokens"].tolist() == ["128", "256"]


def test_incomplete_rows_share_series_without_placeholders() -> None:
    raw = _workload_rows().drop(
        columns=[
            "tag_configured_window_tokens",
            "tag_context_tokens_actual",
            "tag_prompt_tokens",
            "tag_max_gen_toks",
            "tag_truncated_rate",
        ]
    )
    rows = app.add_evidence_metadata(raw, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert not rows["workload_complete"].any()
    assert rows["series_key"].nunique() == 1
    label = rows.iloc[0]["series_label"]
    assert "unknown" not in label.casefold()
    assert "unspecified" not in label.casefold()
    assert "{" not in label
    assert "Machine A" in label
    assert "Accelerator A" not in label
    assert "Metal" not in label
    visible_labels = app.series_labels(rows)
    assert len(set(visible_labels.values())) == 1
    assert all("(run " not in value for value in visible_labels.values())


def test_viewer_labels_show_machine_engine_and_varying_workload_once() -> None:
    raw = _workload_rows()
    raw["hostname"] = "Apple M4 Max"
    raw["chip"] = "Apple M4 Max"
    rows = app.add_evidence_metadata(raw, {}, pd.Timestamp("2026-08-25T00:00:00Z"))

    labels = app.series_labels(rows)

    assert len(set(labels.values())) == 2
    assert all(
        label.startswith("Qwen3.5-9B-MLX-4bit · Mac Studio M4 Max 128GB · MLX") for label in labels.values()
    )
    assert any(
        "context 4096" in label and "prompt 1024" in label and "output 128" in label
        for label in labels.values()
    )
    assert any(
        "context 16384" in label and "prompt 8192" in label and "output 256" in label
        for label in labels.values()
    )
    assert all("None" not in label for label in labels.values())
    assert all(label.count("Mac Studio M4 Max 128GB") == 1 for label in labels.values())
    assert list(app.series_labels(rows.iloc[[0]]).values()) == [
        "Qwen3.5-9B-MLX-4bit · Mac Studio M4 Max 128GB · MLX"
    ]


def test_cascade_options_are_only_row_backed_and_summary_uses_task_and_unit() -> None:
    df = _sample_df()
    other = df.iloc[[0]].copy()
    other["suite"] = "throughput"
    other["name"] = "short-50"
    other["metric"] = "throughput"
    other["value"] = 63.4
    other["unit"] = "tok/s"
    other["display_unit"] = "tok/s"
    rows = pd.concat([df, other], ignore_index=True)

    assert app.suite_choices(rows) == ["reasoning"]
    assert app.task_choices(rows, "throughput") == ["short-50"]
    assert app.metric_choices(rows, "throughput", "short-50") == ["throughput"]
    assert app.valid_triples(rows) == [
        ("reasoning", "gsm8k_cot_zeroshot", "exact_match_flexible"),
        ("throughput", "short-50", "throughput"),
    ]
    table = app.summary_table(rows, "throughput", "short-50", "throughput", "tok/s")
    assert table["Value (tok/s)"].tolist() == ["63.4 tok/s"]
    assert "unitless" not in table.to_string().casefold()


def test_evidence_views_and_unit_formatting_are_distinct() -> None:
    rows = _workload_rows()
    paths = ["data/scored.parquet", "data/experimental.parquet"]
    rows = pd.concat([rows.iloc[[0]], rows.iloc[[1]]], ignore_index=True)
    rows["source_path"] = paths
    index = {
        "data/scored.parquet": {"status": "scored"},
        "data/experimental.parquet": {"status": "experimental"},
    }
    recovered = rows.iloc[[1]].copy()
    recovered["source_path"] = "data/recovered.parquet"
    rows = pd.concat([rows, recovered], ignore_index=True)
    index["data/recovered.parquet"] = {"status": "recovered"}
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    enriched = app.add_evidence_metadata(rows, index, None)

    assert len(app.evidence_view(enriched, "scored")) == 1
    assert len(app.evidence_view(enriched, "experimental")) == 1
    assert len(app.evidence_view(enriched, "recovered")) == 1
    assert app.format_value(0.875, "%") == "87.5%"
    assert app.format_value(63.4, "tok/s") == "63.4 tok/s"


def test_campaign_dimension_columns_drive_filters_and_comparison_identity() -> None:
    rows = _workload_rows()
    rows["campaign_hardware_machine"] = "Benchmark workstation"
    rows["campaign_hardware_accelerator_model"] = "RTX PRO 6000"
    rows["campaign_software_engine"] = "llama.cpp"
    rows["campaign_software_backend"] = "CUDA"
    rows["campaign_model_quantization"] = "Q4_K_M"
    rows["campaign_run_concurrent_agents"] = 4
    rows["campaign_run_allocated_context_tokens"] = 32768
    rows["campaign_run_depth_tokens"] = [4096, 8192]
    rows["campaign_run_prompt_tokens"] = 4096
    rows["campaign_run_output_tokens"] = [128, 256]

    enriched = app.add_evidence_metadata(
        rows,
        {},
        pd.Timestamp("2026-08-25T00:00:00Z"),
    )

    assert app.row_choices(enriched, "machine") == ["Benchmark workstation"]
    assert app.row_choices(enriched, "accelerator") == ["RTX PRO 6000"]
    assert app.row_choices(enriched, "engine") == ["llama.cpp"]
    assert app.row_choices(enriched, "backend") == ["CUDA"]
    assert app.row_choices(enriched, "quant_format") == ["Q4_K_M"]
    assert app.row_choices(enriched, "agents") == ["4"]
    assert app.row_choices(enriched, "allocated_context") == ["32768"]
    assert app.row_choices(enriched, "prompt_tokens") == ["4096"]
    assert enriched["context_tokens"].tolist() == ["4096", "8192"]
    assert enriched["output_budget_tokens"].tolist() == ["128", "256"]
    assert enriched["series_key"].nunique() == 2


def test_request_rate_metric_keeps_request_units() -> None:
    assert app._metric_unit("throughput_requests_per_s") == "req/s"
