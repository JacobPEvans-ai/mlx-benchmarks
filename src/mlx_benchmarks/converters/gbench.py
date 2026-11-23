"""Converter for Google DeepMind ``gbench`` benchmark outputs."""

from __future__ import annotations

import datetime
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.envelope import Envelope, Result

_ISO_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")

_TAG_KEYS: tuple[str, ...] = (
    "benchmark_type",
    "format",
    "model_short",
    "served_model_id",
    "subject_id",
    "scaffold_id",
    "eval_preset",
    "campaign",
    "workload_dataset",
    "multimodal",
    "certified",
    "gbench_commit",
)

_SERVING_METRICS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "throughput",
        "output_tokens_per_second",
        "tok/s",
        ("output_tokens_per_second_mean", "output_tokens_per_second", "throughput_output_toks_per_s"),
    ),
    (
        "request_throughput",
        "requests_per_second",
        "req/s",
        ("request_throughput_mean", "request_throughput", "requests_per_s"),
    ),
    ("ttft_p50", "latency_ms", "ms", ("ttft_p50_ms", "median_ttft_ms", "mean_ttft_ms")),
    ("ttft_p99", "latency_ms", "ms", ("ttft_p99_ms",)),
    ("tpot_p50", "latency_ms", "ms", ("tpot_p50_ms", "median_tpot_ms", "mean_tpot_ms")),
    ("itl_p50", "latency_ms", "ms", ("itl_p50_ms", "median_itl_ms", "mean_itl_ms")),
    ("itl_p99", "latency_ms", "ms", ("itl_p99_ms",)),
)

_STRESS_METRICS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("stress_test", "max_sustainable_qps", "req/s", ("max_sustainable_qps",)),
    ("stress_test_concurrency", "max_concurrent_users", "requests", ("little_law_n_users",)),
    ("stress_test_e2e_latency", "mean_latency_ms", "ms", ("mean_e2e_ms",)),
    (
        "stress_test_single_stream_ttft",
        "latency_ms",
        "ms",
        ("preflight_single_stream_ttft_ms",),
    ),
    (
        "stress_test_single_stream_tpot",
        "latency_ms",
        "ms",
        ("preflight_single_stream_per_token_ms",),
    ),
)


def _as_finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = float(value)
        if math.isfinite(num):
            return num
    return None


def _pick_number(record: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        num = _as_finite(record.get(key))
        if num is not None:
            return num
    return None


def _as_ratio(percent: float) -> float:
    return round(percent / 100.0, 6)


def _iso_timestamp(raw_ts: Any) -> str | None:
    if not isinstance(raw_ts, str) or not _ISO_TIMESTAMP_RE.match(raw_ts):
        return None
    try:
        parsed = datetime.datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stringify_tag(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text or None


def _record_tags(record: Mapping[str, Any], btype: str, ctx_tags: Mapping[str, str]) -> dict[str, str]:
    tags: dict[str, str] = {"gbench_type": btype}
    for key in _TAG_KEYS:
        tag_val = _stringify_tag(record.get(key))
        if tag_val is not None and key != "benchmark_type":
            tags[key] = tag_val
    certified = _stringify_tag(record.get("max_sustainable_qps_certified"))
    if certified is not None:
        tags.setdefault("certified", certified)
    for k, v in ctx_tags.items():
        tags[k] = v
    return tags


def _infer_benchmark_type(record: Mapping[str, Any]) -> str:
    btype = record.get("benchmark_type")
    if isinstance(btype, str) and btype.strip():
        return btype.strip()
    if "eval_name" in record:
        return "eval"
    if "task_results" in record or "passed_cases" in record or "accuracy_percent" in record:
        return "golden"
    if "max_sustainable_qps" in record or record.get("stress_test"):
        return "stress_test_multimodal" if record.get("multimodal") else "stress_test"
    if "passed_scenarios" in record and "total_scenarios" in record:
        return "quality"
    if any(k in record for k in ("ttft_p50_ms", "median_ttft_ms", "tpot_p50_ms", "median_tpot_ms")):
        return "serving_multimodal" if record.get("multimodal") else "serving"
    return "unknown"


def _apply_duration(result: Result, record: Mapping[str, Any]) -> None:
    dur = _pick_number(record, "duration_seconds", "elapsed_s", "duration_s")
    if dur is not None and dur >= 0:
        result["duration_seconds"] = dur


def _failure_message(record: Mapping[str, Any], btype: str) -> str | None:
    status = str(record.get("status") or "").lower()
    failed = bool(record.get("failed")) or status in {"failed", "error", "contaminated"}
    err = record.get("error") or record.get("no_rate_reason")
    if not failed and not err:
        return None
    label = (
        record.get("eval_name")
        or record.get("model_short")
        or record.get("model")
        or record.get("requested_model")
        or btype
    )
    detail = str(err or status or "benchmark failed").strip()
    return f"{btype} ({label}): {detail}"


def _extract_golden(
    record: Mapping[str, Any], btype: str, ctx_tags: Mapping[str, str]
) -> tuple[list[Result], list[str]]:
    tags = _record_tags(record, btype, ctx_tags)
    results: list[Result] = []
    errors = [str(e) for e in (record.get("harness_errors") or []) if str(e).strip()]

    acc_pct = _pick_number(record, "accuracy_percent", "accuracy")
    if acc_pct is None:
        passed = _pick_number(record, "passed_cases")
        denom = _pick_number(record, "applicable_tasks", "total_tasks")
        if passed is not None and denom is not None and denom > 0:
            acc_pct = (passed / denom) * 100.0
    if acc_pct is not None:
        row: Result = {
            "name": "golden",
            "metric": "pass_rate",
            "value": _as_ratio(acc_pct),
            "unit": "ratio",
            "tags": tags,
            "raw": dict(record),
        }
        _apply_duration(row, record)
        results.append(row)

    task_results = record.get("task_results")
    if isinstance(task_results, list):
        by_cat: dict[str, list[bool]] = {}
        for task in task_results:
            if not isinstance(task, Mapping):
                continue
            status = str(task.get("status") or "").lower()
            if status == "not_applicable":
                continue
            cat = str(task.get("category") or "general").strip() or "general"
            by_cat.setdefault(cat, []).append(status == "passed")
        if len(by_cat) > 1:
            for cat, outcomes in sorted(by_cat.items()):
                if not outcomes:
                    continue
                results.append(
                    {
                        "name": f"golden/{cat}",
                        "metric": "pass_rate",
                        "value": round(sum(outcomes) / len(outcomes), 6),
                        "unit": "ratio",
                        "tags": {**tags, "category": cat},
                    }
                )
    return results, errors


def _extract_eval(
    record: Mapping[str, Any], btype: str, ctx_tags: Mapping[str, str]
) -> tuple[list[Result], list[str]]:
    eval_name = str(record.get("eval_name") or "eval").strip() or "eval"
    tags = _record_tags(record, btype, ctx_tags)
    errors: list[str] = []
    results: list[Result] = []

    reasoning_arm: str | None = None
    if record.get("thinking") is True:
        reasoning_arm = "thinking_on"
    elif record.get("thinking") is False:
        reasoning_arm = "thinking_off"

    trunc_rate: float | None = None
    cut_off = _pick_number(record, "replies_cut_off_by_budget")
    total_q = _pick_number(record, "effective_n", "total_questions")
    if cut_off is not None and total_q is not None and total_q > 0:
        trunc_rate = round(cut_off / total_q, 6)

    acc_pct = _pick_number(record, "accuracy")
    if acc_pct is not None:
        raw_payload = {k: v for k, v in record.items() if k != "sample_traces"}
        row: Result = {
            "name": eval_name,
            "metric": "accuracy",
            "value": _as_ratio(acc_pct),
            "unit": "ratio",
            "tags": tags,
            "raw": raw_payload,
        }
        if reasoning_arm is not None:
            row["reasoning_arm"] = reasoning_arm
        if trunc_rate is not None:
            row["truncation_rate"] = trunc_rate
        _apply_duration(row, record)
        results.append(row)

    cats = record.get("category_accuracy")
    if isinstance(cats, Mapping):
        for cat, cstats in sorted(cats.items()):
            if not isinstance(cstats, Mapping):
                continue
            cat_acc = _pick_number(cstats, "accuracy")
            if cat_acc is None:
                continue
            cat_row: Result = {
                "name": f"{eval_name}/{cat}",
                "metric": "accuracy",
                "value": _as_ratio(cat_acc),
                "unit": "ratio",
                "tags": {**tags, "category": str(cat)},
            }
            if reasoning_arm is not None:
                cat_row["reasoning_arm"] = reasoning_arm
            results.append(cat_row)

    return results, errors


def _extract_metric_table(
    record: Mapping[str, Any],
    btype: str,
    ctx_tags: Mapping[str, str],
    specs: tuple[tuple[str, str, str, tuple[str, ...]], ...],
    extra_errors: list[str] | None = None,
) -> tuple[list[Result], list[str]]:
    tags = _record_tags(record, btype, ctx_tags)
    results: list[Result] = []
    ttft_p50 = _pick_number(record, "ttft_p50_ms", "median_ttft_ms", "mean_ttft_ms")
    decode_tps = _pick_number(
        record, "output_tokens_per_second_mean", "output_tokens_per_second", "throughput_output_toks_per_s"
    )
    prefix = f"{btype}/" if btype.endswith("_multimodal") else ""

    for idx, (name, metric, unit, keys) in enumerate(specs):
        val = _pick_number(record, *keys)
        if val is None:
            continue
        row: Result = {
            "name": f"{prefix}{name}",
            "metric": metric,
            "value": val,
            "unit": unit,
            "tags": tags,
        }
        if idx == 0:
            row["raw"] = dict(record)
            _apply_duration(row, record)
        if ttft_p50 is not None:
            row["first_token_latency_ms"] = ttft_p50
        if decode_tps is not None:
            row["decode_tokens_per_second"] = decode_tps
        results.append(row)

    return results, extra_errors or []


def _extract_serving(
    record: Mapping[str, Any], btype: str, ctx_tags: Mapping[str, str]
) -> tuple[list[Result], list[str]]:
    return _extract_metric_table(record, btype, ctx_tags, _SERVING_METRICS)


def _extract_stress(
    record: Mapping[str, Any], btype: str, ctx_tags: Mapping[str, str]
) -> tuple[list[Result], list[str]]:
    errors: list[str] = []
    reason = record.get("no_rate_reason")
    if reason:
        errors.append(f"{btype}: {reason}")
    return _extract_metric_table(record, btype, ctx_tags, _STRESS_METRICS, extra_errors=errors)


def _extract_quality(
    record: Mapping[str, Any], btype: str, ctx_tags: Mapping[str, str]
) -> tuple[list[Result], list[str]]:
    tags = _record_tags(record, btype, ctx_tags)
    errors: list[str] = []
    withheld = record.get("pass_rate_withheld_reason")
    if withheld:
        errors.append(f"quality withheld: {withheld}")
    pass_rate = _pick_number(record, "pass_rate")
    if pass_rate is None:
        return [], errors
    row: Result = {
        "name": "quality",
        "metric": "pass_rate",
        "value": _as_ratio(pass_rate),
        "unit": "ratio",
        "tags": tags,
        "raw": dict(record),
    }
    _apply_duration(row, record)
    return [row], errors


_ExtractorFn = Callable[[Mapping[str, Any], str, Mapping[str, str]], tuple[list[Result], list[str]]]

_EXTRACTORS: dict[str, _ExtractorFn] = {
    "golden": _extract_golden,
    "eval": _extract_eval,
    "serving": _extract_serving,
    "serving_multimodal": _extract_serving,
    "stress_test": _extract_stress,
    "stress_test_multimodal": _extract_stress,
    "quality": _extract_quality,
}


def _collect_records(raw: Any) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], str | None]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)], [], None
    if not isinstance(raw, Mapping):
        raise ValueError("gbench converter expects a JSON object or array of result records")
    if "models" in raw or "failures" in raw:
        models = [item for item in (raw.get("models") or []) if isinstance(item, Mapping)]
        failures = [item for item in (raw.get("failures") or []) if isinstance(item, Mapping)]
        return models, failures, _iso_timestamp(raw.get("timestamp"))
    return [raw], [], _iso_timestamp(raw.get("timestamp"))


class GbenchConverter:
    """Parse ``gbench`` ``summary.json`` or single-pillar JSON artifacts."""

    kind = "gbench"

    def build_envelope(self, raw: Any, ctx: ConverterContext) -> Envelope:
        """Build an envelope v1 payload from ``gbench`` results."""
        records, failures, summary_ts = _collect_records(raw)
        results: list[Result] = []
        errors: list[str] = []

        for failure in failures:
            msg = _failure_message(failure, _infer_benchmark_type(failure))
            if msg:
                errors.append(msg)

        for record in records:
            btype = _infer_benchmark_type(record)
            fail_msg = _failure_message(record, btype)
            if fail_msg and fail_msg not in errors:
                errors.append(fail_msg)
            extractor = _EXTRACTORS.get(btype)
            if extractor is None:
                continue
            part_results, part_errors = extractor(record, btype, ctx.extra_tags)
            results.extend(part_results)
            for err in part_errors:
                if err not in errors:
                    errors.append(err)

        if not results:
            detail = "; ".join(errors) if errors else "gbench payload contains no benchmark results"
            raise ValueError(detail)

        return simple_envelope(ctx, results, errors=errors, timestamp=summary_ts)
