"""dense-layout runner JSON (harness/dense_layout/run.py) -> envelope v1 converter.

One raw file per (model, run): a ``config`` block, per-request ``records``,
per-``label`` aggregates, and an ``overall`` block. Every aggregate block
becomes several metric rows under the constant result name ``dense-layout``;
the label and the request concurrency travel as string tags so shards stay
comparable only within the same sweep dimension. The overall block publishes
under the ``all`` label.

CER and WER are lower-is-better and deliberately **not** ``ratio``-unit rows:
they are unbounded above (a response longer than its reference scores > 1),
so each row states its unit and its ``direction`` tag explicitly. ``thinking``
is always ``n/a`` — image-to-text transcription models have no thinking arm.
The overall block carries ``duration_seconds``; label blocks share the same
run window, so repeating it per label would overstate their individual cost.
"""

from __future__ import annotations

import datetime
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields
from mlx_benchmarks.envelope import Envelope, Result, System

# Aggregate-block key -> (metric, unit, direction). Units say what was measured;
# direction says how to read it. cer/wer are not ratios in [0, 1] and must not
# wear the ratio unit.
_METRIC_SPECS: list[tuple[str, str, str, str]] = [
    ("cer", "cer", "edit_distance_per_reference_char", "lower_is_better"),
    ("wer", "wer", "edit_distance_per_reference_word", "lower_is_better"),
    ("field_recall", "field_recall", "ratio", "higher_is_better"),
    ("seconds_per_image", "seconds_per_image", "seconds", "lower_is_better"),
    ("error_rate", "error_rate", "ratio", "lower_is_better"),
]

OVERALL_LABEL = "all"


class DenseLayoutConverter:
    kind = "dense-layout"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        overall = raw.get("overall")
        if not isinstance(overall, dict) or not overall:
            raise ValueError("dense-layout raw results contain no overall block — nothing to publish")
        labels = raw.get("labels") or {}
        if not isinstance(labels, dict):
            raise ValueError("dense-layout raw results 'labels' must be an object keyed by label")

        concurrency = str((raw.get("config") or {}).get("concurrency", ""))
        system: System = ctx.system or {}  # type: ignore[assignment]

        results: list[Result] = []
        for label, block in sorted(labels.items()):
            if isinstance(block, dict):
                results.extend(_block_results(block, str(label), concurrency, ctx))
        results.extend(_block_results(overall, OVERALL_LABEL, concurrency, ctx))

        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": ctx.timestamp_override or _extract_timestamp(raw),
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": results,
            "errors": [],
        }
        return apply_optional_fields(envelope, ctx)


def _block_results(
    block: dict[str, Any], label: str, concurrency: str, ctx: ConverterContext
) -> list[Result]:
    tags: dict[str, str] = {
        "label": label,
        "thinking": "n/a",
        "concurrency": concurrency,
        "n_records": str(block.get("n_records", "")),
        **{k: str(v) for k, v in ctx.extra_tags.items()},
    }

    results: list[Result] = []
    for raw_key, metric, unit, direction in _METRIC_SPECS:
        value = block.get(raw_key)
        if not isinstance(value, int | float):
            continue
        result: Result = {
            "name": "dense-layout",
            "metric": metric,
            "value": float(value),
            "unit": unit,
            "tags": {**tags, "direction": direction},
        }
        if label == OVERALL_LABEL and isinstance(block.get("wall_seconds"), int | float):
            result["duration_seconds"] = float(block["wall_seconds"])
        results.append(result)
    return results


def _extract_timestamp(raw: dict[str, Any]) -> str:
    ts = raw.get("timestamp")
    if isinstance(ts, str) and ts:
        return ts
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
