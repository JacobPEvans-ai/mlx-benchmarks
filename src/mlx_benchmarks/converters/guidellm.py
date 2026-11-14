"""GuideLLM ``benchmarks.json`` report -> envelope v1 converter.

GuideLLM (https://github.com/vllm-project/guidellm, v0.8.1) writes a
``GenerativeBenchmarksReport`` — one JSON object with a ``benchmarks`` list,
one entry per load profile it ran. The sweep runs GuideLLM once per
(concurrency level, repeat) pair, so a convertible file carries exactly one
benchmark; more than one means the file must be split first.

Metric names follow the repo's serving-throughput vocabulary (the ``--kind
vllm`` converter's): ``throughput_output_toks_per_s`` etc. Two latency
families are kept apart on purpose — GuideLLM's ``time_to_first_token_ms``
covers the first token of any kind, while ``time_to_first_output_token_ms``
is the first *content* (non-reasoning) token, i.e. the first answer token of
a thinking-on run. Published here as ``ttft_*`` and ``ttfa_*``.

Failure counting: GuideLLM does not retain the server's ``finish_reason``, so
a completed response counts as a failed answer when its output tokens reach
the request's max_tokens cap (the ``finish_reason == "length"`` equivalent —
cap read per request from ``request_args``, falling back to the backend
config) or when its answer text is empty. Requests the scheduler cancelled at
the measurement boundary are counted separately as ``requests_incomplete``;
failed requests as ``request_errors``.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
from typing import Any, cast

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields
from mlx_benchmarks.dataset_schema import THROUGHPUT_SUITES
from mlx_benchmarks.envelope import CampaignDimensions, Envelope, Result

log = logging.getLogger(__name__)

# (metrics path, metric name, unit, summary key). Rate metrics use the
# duration-weighted ``mean`` of GuideLLM's rate distribution (total tokens /
# measured seconds); latency families use percentiles of completed requests.
# Missing metrics are skipped so the converter stays forward-compatible.
_METRIC_ROWS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (("output_tokens_per_second",), "throughput_output_toks_per_s", "tok/s", "mean"),
    (("tokens_per_second",), "throughput_total_toks_per_s", "tok/s", "mean"),
    (("prompt_tokens_per_second",), "throughput_prompt_toks_per_s", "tok/s", "mean"),
    (("requests_per_second",), "requests_per_s", "req/s", "mean"),
    (("request_concurrency",), "measured_concurrency_mean", "count", "mean"),
    (("time_to_first_token_ms",), "ttft_p50_ms", "ms", "percentiles.p50"),
    (("time_to_first_token_ms",), "ttft_p99_ms", "ms", "percentiles.p99"),
    (("time_to_first_output_token_ms",), "ttfa_p50_ms", "ms", "percentiles.p50"),
    (("time_to_first_output_token_ms",), "ttfa_p99_ms", "ms", "percentiles.p99"),
    (("time_per_output_token_ms",), "tpot_p50_ms", "ms", "percentiles.p50"),
    (("inter_token_latency_ms",), "itl_p50_ms", "ms", "percentiles.p50"),
)

_REQUEST_COUNT_ROWS: tuple[tuple[str, str], ...] = (
    ("request_errors", "errored"),
    ("requests_incomplete", "incomplete"),
    ("requests_successful", "successful"),
    ("requests_total", "total"),
)


class GuidellmConverter:
    kind = "guidellm"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        benchmark = _single_benchmark(raw)
        config = _dict(benchmark.get("config"))
        metrics = _dict(benchmark.get("metrics"))
        scheduler_metrics = _dict(benchmark.get("scheduler_metrics"))
        backend = _dict(config.get("backend"))
        strategy = _dict(config.get("strategy"))

        streams = _positive_int(strategy.get("streams"))
        if streams is None:
            raise ValueError(
                "GuideLLM benchmark has no concurrent-profile streams count; run the "
                "sweep with --profile kind=concurrent,streams=<n> "
                f"(strategy: {strategy.get('type_')!r})"
            )

        guidellm_version = _dict(raw.get("metadata")).get("guidellm_version")
        max_tokens = _configured_max_tokens(backend)
        failures = _failed_answers(_request_list(benchmark, "successful"), max_tokens)
        errors = [str(warning) for warning in benchmark.get("warnings") or []]
        if max_tokens is None:
            errors.append(
                "length-truncation detection unavailable: no max_tokens in the "
                "GuideLLM backend config or request args"
            )

        total_requests = _number(_dict(metrics.get("request_totals")).get("total"))
        if total_requests is not None and total_requests <= 0:
            raise ValueError("GuideLLM benchmark recorded no requests")

        duration = _measured_duration(scheduler_metrics)
        base_tags = self._base_tags(config, metrics, streams, guidellm_version, max_tokens)
        request_totals = _dict(metrics.get("request_totals"))

        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": _timestamp(scheduler_metrics, ctx.timestamp_override),
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": self._model(ctx, backend),
            "system": ctx.system or {},  # type: ignore[assignment]
            "results": self._results(metrics, request_totals, failures, duration, base_tags, max_tokens),
            "errors": errors,
        }
        apply_optional_fields(envelope, ctx)
        envelope["concurrency"] = streams  # the file is the ground truth for the level

        if ctx.suite in THROUGHPUT_SUITES:
            dimensions = _campaign_dimensions(metrics, max_tokens, streams)
            if dimensions is not None:
                # Merge onto (not replace) caller-supplied dimensions: the CLI's
                # --campaign-dimensions file may already carry hardware/software
                # groups; this converter only owns the run/speed facts.
                merged = cast("CampaignDimensions", dict(envelope.get("campaign_dimensions") or {}))
                merged.update(dimensions)
                envelope["campaign_dimensions"] = merged
        return envelope

    def _model(self, ctx: ConverterContext, backend: dict[str, Any]) -> str:
        if ctx.model not in (None, "", "unknown"):
            return ctx.model
        served = backend.get("model")
        if isinstance(served, str) and served:
            return served
        raise ValueError("no model id: pass --model or serve with a named backend model")

    def _base_tags(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any],
        streams: int,
        guidellm_version: Any,
        max_tokens: int | None,
    ) -> dict[str, str]:
        tags: dict[str, str] = {
            "thinking": "on",
            "concurrency": str(streams),
            "guidellm_version": str(guidellm_version),
            "benchmark_id": str(config.get("id_")),
        }
        served = _dict(config.get("backend")).get("model")
        if isinstance(served, str) and served:
            tags["served_model"] = served
        prompt_mean = _summary_number(metrics, ("prompt_token_count",), "successful", "mean")
        if prompt_mean is not None:
            tags["prompt_tokens"] = str(round(prompt_mean))
        if max_tokens is not None:
            tags["max_tokens"] = str(max_tokens)
        return tags

    def _results(
        self,
        metrics: dict[str, Any],
        request_totals: dict[str, Any],
        failures: dict[str, int],
        duration: float | None,
        base_tags: dict[str, str],
        max_tokens: int | None,
    ) -> list[Result]:
        results: list[Result] = []

        for path, metric, unit, summary_key in _METRIC_ROWS:
            value = _summary_number(metrics, path, "successful", summary_key)
            if value is None:
                log.debug("GuideLLM metrics missing %s — skipping %s", "/".join(path), metric)
                continue
            results.append(_row(metric, value, unit, duration, base_tags))

        per_request = _summary_number(metrics, ("text", "usage", "output_per_second"), "successful", "median")
        if per_request is not None:
            results.append(_row("output_toks_per_s_per_request", per_request, "tok/s", duration, base_tags))

        for metric, group in _REQUEST_COUNT_ROWS:
            value = _number(request_totals.get(group))
            if value is not None:
                row = _row(metric, value, "count", None, base_tags)
                if metric == "requests_incomplete" and value > 0:
                    row.setdefault("tags", {})["note"] = "cancelled at the measurement boundary"
                results.append(row)

        # Completed responses that ended at the max_tokens cap (finish_reason
        # "length" equivalent) or carry an empty answer: failures, not successes.
        failed = failures["truncated_or_empty"]
        successful = _number(request_totals.get("successful"))
        if failed > 0 or max_tokens is not None:
            row = _row("truncated_or_empty_answers", float(failed), "count", None, base_tags)
            row_tags = row.setdefault("tags", {})
            row_tags["failed_truncated"] = str(failures["truncated"])
            row_tags["failed_empty_answers"] = str(failures["empty"])
            if failed > 0 and successful is not None and successful > 0:
                row["truncation_rate"] = failed / successful
            results.append(row)

        if results:
            headline = next(
                (row for row in results if row.get("metric") == "throughput_output_toks_per_s"),
                results[0],
            )
            for field, path, summary_key in (
                ("output_tokens_mean", ("output_token_count",), "mean"),
                ("output_tokens_median", ("output_token_count",), "median"),
                ("output_tokens_p90", ("output_token_count",), "percentiles.p90"),
                ("reasoning_tokens_mean", ("text", "usage", "reasoning"), "mean"),
            ):
                value = _summary_number(metrics, path, "successful", summary_key)
                if value is not None:
                    headline[field] = value
        return results


def _row(metric: str, value: float, unit: str, duration: float | None, tags: dict[str, str]) -> Result:
    result: Result = {"name": "guidellm", "metric": metric, "value": value, "unit": unit, "tags": dict(tags)}
    if duration is not None:
        result["duration_seconds"] = duration
    return result


def _single_benchmark(raw: dict[str, Any]) -> dict[str, Any]:
    benchmarks = raw.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError(
            "not a GuideLLM benchmarks.json report: missing or empty 'benchmarks' list "
            "(expected GenerativeBenchmarksReport with one benchmark per file)"
        )
    if len(benchmarks) != 1:
        raise ValueError(
            f"GuideLLM report holds {len(benchmarks)} benchmarks; the sweep writes one "
            "(concurrency level, repeat) per file — split it first"
        )
    benchmark = benchmarks[0]
    if not isinstance(benchmark, dict):
        raise ValueError("GuideLLM benchmark entry is not a JSON object")
    return benchmark


def _request_list(benchmark: dict[str, Any], group: str) -> list[Any]:
    requests = _dict(benchmark.get("requests")).get(group)
    return requests if isinstance(requests, list) else []


def _failed_answers(successful_requests: list[Any], config_cap: int | None) -> dict[str, int]:
    """Classify completed responses as truncated (output reached the request's
    max_tokens cap — GuideLLM does not retain the server ``finish_reason``) or
    empty (no answer text; reasoning text alone does not count). A response
    that is both is counted once in ``truncated_or_empty``."""
    truncated = 0
    empty = 0
    failed = 0
    for request in successful_requests:
        if not isinstance(request, dict):
            continue
        output_tokens = _positive_int(request.get("output_tokens"))
        answer = request.get("output")
        has_answer = isinstance(answer, str) and answer.strip() != ""
        cap = _request_max_tokens(request, config_cap)
        is_truncated = cap is not None and output_tokens is not None and output_tokens >= cap
        is_empty = not has_answer
        if is_truncated:
            truncated += 1
        if is_empty:
            empty += 1
        if is_truncated or is_empty:
            failed += 1
    return {"truncated_or_empty": failed, "truncated": truncated, "empty": empty}


def _request_max_tokens(request: dict[str, Any], fallback: int | None) -> int | None:
    """The request's own output cap: its serialized request args first, then the
    backend config the sweep passed."""
    raw_args = request.get("request_args")
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            args = json.loads(raw_args)
        except ValueError:
            log.debug("GuideLLM request_args is not JSON; using the configured max_tokens")
            return fallback
        if isinstance(args, dict):
            for source in (args, _dict(args.get("body"))):
                value = _positive_int(source.get("max_tokens")) or _positive_int(
                    source.get("max_completion_tokens")
                )
                if value is not None:
                    return value
    return fallback


def _configured_max_tokens(backend: dict[str, Any]) -> int | None:
    body = _dict(_dict(backend.get("extras")).get("body"))
    return _positive_int(body.get("max_tokens")) or _positive_int(body.get("max_completion_tokens"))


def _campaign_dimensions(
    metrics: dict[str, Any], max_tokens: int | None, streams: int
) -> CampaignDimensions | None:
    """Typed run/speed dimensions for the throughput suites. Only measured
    values are set — no explicit nulls, so no dimension_null_reasons here."""
    run: dict[str, Any] = {"thinking": True, "closed_loop": True, "concurrent_agents": streams}
    if max_tokens is not None:
        run["output_tokens"] = max_tokens
    for field, path in (
        ("prompt_tokens", ("prompt_token_count",)),
        ("measured_output_tokens", ("output_token_count",)),
    ):
        mean = _summary_number(metrics, path, "successful", "mean")
        if mean is not None:
            run[field] = round(mean)

    speed: dict[str, Any] = {}
    for path, field, summary_key in (
        (("time_to_first_token_ms",), "ttft_p50_ms", "percentiles.p50"),
        (("time_to_first_token_ms",), "ttft_p99_ms", "percentiles.p99"),
        (("requests_per_second",), "requests_per_second", "mean"),
        (("output_tokens_per_second",), "aggregate_output_tokens_per_second", "mean"),
        (("tokens_per_second",), "total_tokens_per_second", "mean"),
        (("time_per_output_token_ms",), "tpot_ms", "percentiles.p50"),
    ):
        value = _summary_number(metrics, path, "successful", summary_key)
        if value is not None:
            speed[field] = value
    per_request = _summary_number(metrics, ("text", "usage", "output_per_second"), "successful", "median")
    if per_request is not None:
        speed["decode_tokens_per_second_per_agent"] = per_request

    if not speed:
        return None
    return {"run": run, "speed": speed}


def _timestamp(scheduler_metrics: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    for key in ("start_time", "measure_start_time"):
        value = _number(scheduler_metrics.get(key))
        if value is not None and value > 0:
            return datetime.datetime.fromtimestamp(value, tz=datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _measured_duration(scheduler_metrics: dict[str, Any]) -> float | None:
    start = _number(scheduler_metrics.get("measure_start_time"))
    end = _number(scheduler_metrics.get("measure_end_time"))
    if start is None or end is None or end <= start:
        return None
    return end - start


def _summary_number(
    metrics: dict[str, Any], path: tuple[str, ...], group: str, summary_key: str
) -> float | None:
    """Walk metrics -> path -> status group -> summary key (e.g. ``mean`` or
    ``percentiles.p99``) and return a finite float, else None."""
    node: Any = metrics
    for key in (*path, group, *summary_key.split(".")):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return _number(node)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    return int(number)
