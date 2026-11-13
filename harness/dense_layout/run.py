#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28.1", "jiwer>=3.0"]
# ///
"""dense-layout — image-to-text extraction accuracy benchmark.

Sends each dense-layout image to an OpenAI-compatible ``/v1/chat/completions``
endpoint as a base64 ``image_url`` content part with a fixed extraction
prompt, and scores the extracted text against the dataset's reference
answer. Dense layouts — many panes, tables, tab strips, form fields — are where
small image-to-text models degrade first, so every score is deterministic and
none is model-graded:

``cer`` / ``wer``
    Character and word error rate (``jiwer``) after whitespace normalisation.
    Lower is better. Neither is bounded by 1: a response longer than the
    reference scores above 1, which is exactly the failure mode worth seeing.
``field_recall``
    Share of the answer's reference field values found verbatim in the output
    (case-insensitive, whitespace-normalised).
``seconds_per_image`` / ``error_rate``
    Mean successful-request latency and the share of requests that errored.

An answer file may set ``"text_complete": false`` when the reference text is
not the model's job to reproduce in full: that image then drops out of the
CER/WER means (a missing key means the extracted text is complete) and only
``field_recall``, latency, and ``error_rate`` score it.

An empty completion or ``finish_reason == "length"`` is a counted error and a
non-zero exit — a truncated or absent extraction must never score as a
(possibly perfect) empty answer.

Dataset layout (directory named by ``DENSE_LAYOUT_DATASET_DIR``; the dataset
never lives inside the repo)::

    manifest.json          # list of {"file": ..., "label": ...}
    images/<file>.jpg      # the dense-layout images
    answers/<file>.json    # {"text": full reference answer,
                           #  "text_complete": false,  (optional; see above)
                           #  "fields": {"key": "value", ...}}  (fields optional)

Run::

    DENSE_LAYOUT_DATASET_DIR=/path/to/dataset \\
    uv run harness/dense_layout/run.py --base-url http://127.0.0.1:11434/v1 \\
        --model <model-id> --output run-output/dense_layout.json

Publish with ``mlx-bench-publish <out.json> --kind dense-layout
--suite dense-layout-image-text``.

Scoring lives in importable pure functions (``normalize_whitespace``,
``score_image``, ``aggregate_records``, ...) so ``tests/test_dense_layout_runner.py``
exercises them without a live endpoint; httpx and jiwer are imported lazily so
the module itself stays importable in a bare test environment.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DATASET_DIR_ENV = "DENSE_LAYOUT_DATASET_DIR"

EXTRACTION_PROMPT = (
    "Extract all text visible in this image exactly as it appears, in natural "
    "reading order. Output only the extracted text, with no commentary."
)


# ---------------------------------------------------------------------------
# Pure functions (unit-tested via tests/test_dense_layout_runner.py)
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space and trim the ends."""
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate after whitespace normalisation. Lower is better.

    Not bounded by 1: an over-generating response scores above 1.
    """
    import jiwer

    return float(jiwer.cer(normalize_whitespace(reference), normalize_whitespace(hypothesis)))


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate after whitespace normalisation. Lower is better."""
    import jiwer

    return float(jiwer.wer(normalize_whitespace(reference), normalize_whitespace(hypothesis)))


def field_matches(output_text: str, field_value: str) -> bool:
    """True when the reference field value appears verbatim in the output.

    Case-insensitive and whitespace-normalised on both sides, so a value
    wrapped across a line break in the extracted text still matches.
    """
    return normalize_whitespace(field_value).casefold() in normalize_whitespace(output_text).casefold()


def field_recall(output_text: str, fields: Mapping[str, str]) -> tuple[float, int, int]:
    """Share of reference field values found verbatim in the output.

    Returns ``(recall, n_found, n_fields)``. Only values are checked — a field
    is recalled when its value string is present, whatever the label around it.
    """
    n_fields = len(fields)
    if not n_fields:
        return 0.0, 0, 0
    n_found = sum(1 for value in fields.values() if field_matches(output_text, value))
    return n_found / n_fields, n_found, n_fields


def score_image(reference_text: str, fields: Mapping[str, str], response_text: str) -> dict[str, Any]:
    """Score one response against one answer. Every check is deterministic."""
    recall, n_found, n_fields = field_recall(response_text, fields)
    return {
        "cer": cer(reference_text, response_text),
        "wer": wer(reference_text, response_text),
        "field_recall": recall if n_fields else None,
        "n_fields": n_fields,
        "n_fields_found": n_found,
    }


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def aggregate_records(records: Sequence[Mapping[str, Any]], wall_seconds: float) -> dict[str, Any]:
    """Aggregate per-request records into one metric block.

    Error records are excluded from every mean (they carry no response to
    score) and counted in ``error_rate``. Records whose answer set
    ``text_complete: false`` carry ``cer``/``wer`` of ``None``: they drop out
    of those means without scoring as 0, while still counting toward
    ``field_recall``, latency, and ``error_rate``. ``field_recall``
    is weighted by field count so an image with many fields moves the number
    more than one with a single field.
    """
    scored = [r for r in records if not r.get("error")]
    n_fields = sum(int(r.get("n_fields") or 0) for r in scored)
    n_found = sum(int(r.get("n_fields_found") or 0) for r in scored)
    seconds = [r["latency_ms"] / 1000 for r in scored if r.get("latency_ms") is not None]
    return {
        "n_records": len(records),
        "n_errors": len(records) - len(scored),
        "cer": mean([r["cer"] for r in scored if r.get("cer") is not None]),
        "wer": mean([r["wer"] for r in scored if r.get("wer") is not None]),
        "field_recall": round(n_found / n_fields, 6) if n_fields else None,
        "n_fields": n_fields,
        "n_fields_found": n_found,
        "seconds_per_image": mean(seconds),
        "error_rate": round((len(records) - len(scored)) / len(records), 6) if records else 0.0,
        "wall_seconds": round(wall_seconds, 3),
    }


def records_with_errors(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """One summary line per failed request; drives the non-zero exit."""
    return [
        f"{r.get('file', '?')} repeat {r.get('repeat', '?')}: {r['error']}" for r in records if r.get("error")
    ]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    """Read ``manifest.json`` plus every image and answer it names.

    Malformed datasets fail loudly here rather than mid-run: a manifest entry
    without its image or answer, or an answer without a non-empty ``text``,
    raises before any request is sent.
    """
    manifest_path = dataset_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read dataset manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"{manifest_path} must be a non-empty JSON list of {{file, label}} objects")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, item in enumerate(manifest):
        if not isinstance(item, dict) or not isinstance(item.get("file"), str) or not item["file"]:
            raise ValueError(f"{manifest_path} entry {position} must be an object with a non-empty 'file'")
        file = item["file"]
        if file in seen:
            raise ValueError(f"{manifest_path} lists {file!r} twice")
        seen.add(file)
        image_path = dataset_dir / "images" / file
        answer_path = dataset_dir / "answers" / f"{Path(file).stem}.json"
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"manifest entry {file!r}: cannot read image {image_path}: {exc}") from exc
        try:
            answer = json.loads(answer_path.read_text())
        except OSError as exc:
            raise ValueError(f"manifest entry {file!r}: cannot read answer {answer_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest entry {file!r}: {answer_path} is not valid JSON: {exc}") from exc
        if not isinstance(answer, dict) or not isinstance(answer.get("text"), str):
            raise ValueError(f"manifest entry {file!r}: {answer_path} must be an object with a 'text' string")
        if not normalize_whitespace(answer["text"]):
            raise ValueError(f"manifest entry {file!r}: {answer_path} has an empty reference 'text'")
        fields = answer.get("fields") or {}
        if not isinstance(fields, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in fields.items()
        ):
            raise ValueError(f"manifest entry {file!r}: 'fields' must be an object of string to string")
        text_complete = answer.get("text_complete", True)
        if not isinstance(text_complete, bool):
            raise ValueError(f"manifest entry {file!r}: 'text_complete' must be a boolean")
        entries.append(
            {
                "file": file,
                "label": str(item.get("label", "unlabeled")),
                "image_bytes": image_bytes,
                "reference_text": answer["text"],
                "text_complete": text_complete,
                "fields": fields,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Endpoint I/O (httpx imported lazily — the module stays importable in the
# test env, where only the pure functions above are exercised)
# ---------------------------------------------------------------------------


def extraction_body(args: argparse.Namespace, image_bytes: bytes) -> dict[str, Any]:
    """One chat completion: fixed extraction prompt + base64 ``image_url``."""
    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }


async def one_request(client: Any, body: dict[str, Any]) -> dict[str, Any]:
    """Send one request and return the raw outcome; network errors become records."""
    record: dict[str, Any] = {
        "content": None,
        "finish_reason": None,
        "latency_ms": None,
        "completion_tokens": None,
        "http_status": None,
        "error": None,
    }
    start = time.monotonic()
    try:
        response = await client.post("chat/completions", json=body)
        record["http_status"] = response.status_code
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        record["content"] = message.get("content")
        record["finish_reason"] = choice.get("finish_reason")
        record["completion_tokens"] = (data.get("usage") or {}).get("completion_tokens")
    except Exception as exc:  # broad on purpose: any network/timeout error is a counted failure
        record["error"] = type(exc).__name__
    record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    return record


def score_request(
    entry: Mapping[str, Any],
    repeat: int,
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn one raw response into one scored record, or one counted error.

    An empty completion and ``finish_reason == "length"`` are errors — never a
    valid (and deceptively cheap) empty extraction. When the answer sets
    ``text_complete: false`` the record skips CER/WER (they stay ``None`` so
    the aggregator cannot average them) and keeps ``field_recall``.
    """
    record: dict[str, Any] = {
        "file": entry["file"],
        "label": entry["label"],
        "repeat": repeat,
        "extracted_text": None,
        "text_complete": bool(entry.get("text_complete", True)),
        "cer": None,
        "wer": None,
        "field_recall": None,
        "n_fields": len(entry.get("fields") or {}),
        "n_fields_found": 0,
        "latency_ms": response.get("latency_ms"),
        "completion_tokens": response.get("completion_tokens"),
        "finish_reason": response.get("finish_reason"),
        "http_status": response.get("http_status"),
        "error": response.get("error"),
    }
    if record["error"]:
        return record
    content = response.get("content")
    text = content if isinstance(content, str) else ""
    if not normalize_whitespace(text):
        record["error"] = "empty_completion"
        return record
    if record["finish_reason"] == "length":
        record["error"] = "finish_reason=length"
        record["extracted_text"] = text
        return record
    record["extracted_text"] = text
    if record["text_complete"]:
        record.update(score_image(entry["reference_text"], entry.get("fields") or {}, text))
    else:
        recall, n_found, n_fields = field_recall(text, entry.get("fields") or {})
        record["field_recall"] = recall if n_fields else None
        record["n_fields"] = n_fields
        record["n_fields_found"] = n_found
    return record


async def run_entry(
    client: Any,
    args: argparse.Namespace,
    entry: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Score one manifest entry ``--repeats`` times."""
    body = extraction_body(args, entry["image_bytes"])
    records: list[dict[str, Any]] = []

    async def once(repeat: int) -> None:
        async with semaphore:
            response = await one_request(client, body)
        records.append(score_request(entry, repeat, response))

    await asyncio.gather(*(once(repeat) for repeat in range(args.repeats)))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dense-layout image-to-text extraction accuracy benchmark"
    )
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="NAME of the env var holding the API key (never pass a literal key)",
    )
    parser.add_argument("--model", required=True, help="Model id as served by the endpoint")
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Per-request output cap; hitting it is an error"
    )
    parser.add_argument("--concurrency", type=int, default=1, help="In-flight request count")
    parser.add_argument("--repeats", type=int, default=2, help="Requests per image (replicated pair)")
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout, seconds")
    parser.add_argument(
        "--output", type=Path, help="Results JSON path (default: dense_layout_results_<ts>.json)"
    )
    return parser


async def _run(args: argparse.Namespace, timestamp: str) -> dict[str, Any]:
    dataset_dir = os.environ.get(DATASET_DIR_ENV, "").strip()
    if not dataset_dir:
        raise ValueError(f"set {DATASET_DIR_ENV} to the dataset directory (never a path inside the repo)")
    entries = load_dataset(Path(dataset_dir))
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    # Imported after dataset validation so a malformed dataset fails loudly even
    # where httpx is not installed.
    import httpx

    headers = {}
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # httpx resolves a relative request path against base_url only when base_url
    # ends with "/" — otherwise it drops base_url's last path segment (the "/v1"
    # prefix). Keep the trailing slash and post a relative path below.
    base_url = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    semaphore = asyncio.Semaphore(args.concurrency)
    records: list[dict[str, Any]] = []
    start = time.monotonic()
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=args.timeout) as client:
        for entry in entries:
            print(f"image {entry['file']} (label {entry['label']}) ...", file=sys.stderr)
            records.extend(await run_entry(client, args, entry, semaphore))
    wall = time.monotonic() - start

    errors = records_with_errors(records)
    labels: dict[str, dict[str, Any]] = {}
    for label in sorted({r["label"] for r in records}):
        label_records = [r for r in records if r["label"] == label]
        labels[label] = aggregate_records(label_records, wall)

    return {
        "benchmark": "dense-layout",
        "model": args.model,
        "timestamp": timestamp,
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "dataset_dir": dataset_dir,
            "n_images": len(entries),
            "repeats": args.repeats,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "timeout": args.timeout,
        },
        "records": records,
        "labels": labels,
        "overall": aggregate_records(records, wall),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ts = timestamp.replace(":", "").replace("-", "")
    out: Path = args.output or Path(f"dense_layout_results_{ts}.json")
    try:
        results = asyncio.run(_run(args, timestamp))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    for error in results["errors"]:
        print(f"error: {error}", file=sys.stderr)
    if results["errors"]:
        print(
            f"{len(results['errors'])} of {len(results['records'])} requests failed; "
            "results are written but the run did not succeed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
