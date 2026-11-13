"""Unit tests for the dense-layout runner's pure functions and dataset loader.

``harness/dense_layout/run.py`` is a standalone PEP 723 script (not part of the
package), so it is loaded here via importlib — the same pattern
``test_factual_runner.py`` uses. Its network (httpx) and scoring (jiwer)
dependencies are imported lazily inside the functions, keeping this module
importable in a bare test environment; the tests that need those dependencies
skip when they are absent and run in CI where they are installed.

The synthetic dataset is generated inside ``tmp_path``: manifest, answers, and
stub JPEG images (the harness base64-encodes image bytes and never decodes
them, so a minimal JPEG-shaped byte string exercises file handling without an
imaging dependency).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    path = REPO_ROOT / "harness" / "dense_layout" / "run.py"
    spec = importlib.util.spec_from_file_location("dense_layout_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

# SOI + JFIF APP0 + EOI: structurally a JPEG, decodable by nothing — which is
# fine, because the harness only reads bytes to base64-encode them.
JPEG_STUB = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def make_dataset(
    tmp_path: Path,
    entries: list[tuple[str, str, str, dict[str, str]]],
) -> Path:
    """entries: (file, label, reference text, fields)."""
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "answers").mkdir(parents=True)
    (dataset / "manifest.json").write_text(
        json.dumps([{"file": file, "label": label} for file, label, _, _ in entries])
    )
    for file, _, text, fields in entries:
        (dataset / "images" / file).write_bytes(JPEG_STUB)
        (dataset / "answers" / f"{Path(file).stem}.json").write_text(
            json.dumps({"text": text, "fields": fields})
        )
    return dataset


# --- whitespace normalisation -------------------------------------------------


def test_normalize_whitespace_collapses_runs_and_trims() -> None:
    assert runner.normalize_whitespace("  alpha\t\tbeta\ngamma  ") == "alpha beta gamma"


# --- dataset loading ------------------------------------------------------------


def test_load_dataset_reads_manifest_images_and_answers(tmp_path: Path) -> None:
    dataset = make_dataset(
        tmp_path,
        [
            ("pane-01.jpg", "tables", "Region north 12", {"total": "1284"}),
            ("form-01.jpg", "forms", "Name North Harbor", {}),
        ],
    )
    entries = runner.load_dataset(dataset)
    assert [e["file"] for e in entries] == ["pane-01.jpg", "form-01.jpg"]
    assert [e["label"] for e in entries] == ["tables", "forms"]
    assert entries[0]["fields"] == {"total": "1284"}
    assert entries[0]["image_bytes"] == JPEG_STUB


def test_load_dataset_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read dataset manifest"):
        runner.load_dataset(tmp_path / "nowhere")


def test_load_dataset_rejects_missing_image(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "answers").mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps([{"file": "gone.jpg", "label": "x"}]))
    (dataset / "answers" / "gone.json").write_text(json.dumps({"text": "answer"}))
    with pytest.raises(ValueError, match="cannot read image"):
        runner.load_dataset(dataset)


def test_load_dataset_rejects_missing_answer(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps([{"file": "pane.jpg", "label": "x"}]))
    (dataset / "images" / "pane.jpg").write_bytes(JPEG_STUB)
    with pytest.raises(ValueError, match="cannot read answer"):
        runner.load_dataset(dataset)


def test_load_dataset_rejects_empty_reference_text(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, [("pane.jpg", "x", "   \n\t  ", {})])
    with pytest.raises(ValueError, match="empty reference"):
        runner.load_dataset(dataset)


def test_load_dataset_rejects_duplicate_manifest_entries(tmp_path: Path) -> None:
    dataset = make_dataset(
        tmp_path,
        [("pane.jpg", "x", "answer", {}), ("pane.jpg", "y", "answer", {})],
    )
    with pytest.raises(ValueError, match="twice"):
        runner.load_dataset(dataset)


# --- scoring (needs jiwer) -------------------------------------------------------


def test_score_image_perfect_extraction(tmp_path: Path) -> None:
    pytest.importorskip("jiwer")
    reference = "Region north 12 Region east 8 TOTAL 1284"
    scores = runner.score_image(reference, {"total": "1284"}, reference)
    assert scores["cer"] == 0.0
    assert scores["wer"] == 0.0
    assert scores["field_recall"] == 1.0


def test_score_image_ignores_whitespace_differences() -> None:
    pytest.importorskip("jiwer")
    reference = "Alpha Beta Gamma"
    response = "Alpha\nBeta\t   Gamma"
    scores = runner.score_image(reference, {}, response)
    assert scores["cer"] == 0.0
    assert scores["wer"] == 0.0


def test_score_image_flags_wrong_extraction() -> None:
    pytest.importorskip("jiwer")
    scores = runner.score_image("Region north 12", {}, "Region south 13")
    assert scores["cer"] > 0
    assert scores["wer"] > 0


def test_cer_is_not_bounded_by_one() -> None:
    pytest.importorskip("jiwer")
    # Over-generation is a real failure mode: the ratio must be allowed to say so.
    assert runner.cer("cat", "cat cat cat cat cat") > 1.0


def test_field_recall_matches_case_and_whitespace_insensitively() -> None:
    fields = {"name": "North Harbor", "zip": "94107"}
    recall, n_found, n_fields = runner.field_recall("north\n  harbor ZIP 94107", fields)
    assert (recall, n_found, n_fields) == (1.0, 2, 2)


def test_field_recall_counts_only_present_values() -> None:
    fields = {"name": "North Harbor", "zip": "94107"}
    recall, n_found, n_fields = runner.field_recall("North Harbor", fields)
    assert (recall, n_found, n_fields) == (0.5, 1, 2)


def test_field_recall_without_fields_is_zero() -> None:
    assert runner.field_recall("anything", {}) == (0.0, 0, 0)


def test_score_image_without_fields_leaves_recall_null() -> None:
    pytest.importorskip("jiwer")
    scores = runner.score_image("some text", {}, "some text")
    assert scores["field_recall"] is None
    assert scores["n_fields"] == 0


# --- request classification (empty completion / truncation are errors) ----------


def test_score_request_scores_a_valid_extraction() -> None:
    pytest.importorskip("jiwer")
    entry = {"file": "pane.jpg", "label": "tables", "reference_text": "Alpha Beta", "fields": {}}
    record = runner.score_request(entry, 0, {"content": "Alpha Beta", "finish_reason": "stop"})
    assert record["error"] is None
    assert record["extracted_text"] == "Alpha Beta"
    assert record["cer"] == 0.0


def test_score_request_empty_completion_is_a_counted_error() -> None:
    entry = {"file": "pane.jpg", "label": "tables", "reference_text": "Alpha Beta", "fields": {}}
    record = runner.score_request(entry, 0, {"content": "   ", "finish_reason": "stop"})
    assert record["error"] == "empty_completion"
    # The unscored record must not carry accuracy numbers an aggregator could average.
    assert record["cer"] is None and record["wer"] is None


def test_score_request_null_content_is_a_counted_error() -> None:
    entry = {"file": "pane.jpg", "label": "tables", "reference_text": "Alpha Beta", "fields": {}}
    record = runner.score_request(entry, 0, {"content": None, "finish_reason": "stop"})
    assert record["error"] == "empty_completion"


def test_score_request_truncation_is_a_counted_error() -> None:
    entry = {"file": "pane.jpg", "label": "tables", "reference_text": "Alpha Beta", "fields": {}}
    record = runner.score_request(entry, 0, {"content": "Alpha Beta Gamma Delta", "finish_reason": "length"})
    assert record["error"] == "finish_reason=length"
    assert record["cer"] is None and record["wer"] is None


def test_score_request_propagates_transport_errors() -> None:
    entry = {"file": "pane.jpg", "label": "tables", "reference_text": "Alpha Beta", "fields": {}}
    record = runner.score_request(entry, 0, {"error": "ConnectError", "content": None})
    assert record["error"] == "ConnectError"


def test_score_request_text_complete_false_skips_cer_and_wer() -> None:
    # "text_complete": false in the answer exempts the image from CER/WER: the
    # record stays a success with null cer/wer (never scored as 0), while
    # field_recall and latency still score and the row counts in error_rate.
    entry = {
        "file": "pane.jpg",
        "label": "tables",
        "reference_text": "Alpha Beta",
        "text_complete": False,
        "fields": {"zip": "94107"},
    }
    response = {"content": "Alpha Beta ZIP 94107", "finish_reason": "stop", "latency_ms": 1200.0}
    record = runner.score_request(entry, 0, response)
    assert record["error"] is None
    assert record["extracted_text"] == "Alpha Beta ZIP 94107"
    assert record["cer"] is None and record["wer"] is None
    assert record["field_recall"] == 1.0
    block = runner.aggregate_records([record], wall_seconds=1.0)
    assert block["cer"] is None and block["wer"] is None
    assert block["field_recall"] == 1.0
    assert block["seconds_per_image"] == 1.2
    assert block["error_rate"] == 0.0


# --- endpoint I/O (needs httpx) --------------------------------------------------


def test_one_request_parses_completion_and_usage() -> None:
    httpx = pytest.importorskip("httpx")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content.decode())
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Alpha Beta"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 7},
            },
        )

    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "prompt"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
                ],
            }
        ],
    }
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1/v1/")
    record = _run_sync(runner.one_request(client, body))
    assert record["content"] == "Alpha Beta"
    assert record["finish_reason"] == "stop"
    assert record["completion_tokens"] == 7
    assert record["error"] is None
    assert record["http_status"] == 200


def test_one_request_records_transport_error_without_raising() -> None:
    httpx = pytest.importorskip("httpx")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1/v1/")
    record = _run_sync(runner.one_request(client, {"messages": []}))
    assert record["error"] == "ConnectError"
    assert record["http_status"] is None
    assert record["latency_ms"] is not None


def _run_sync(coroutine: object) -> object:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


# --- end-to-end pipeline against a stub endpoint ---------------------------------


def test_run_pipeline_against_a_stub_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercises dataset load -> requests -> scoring -> aggregation -> labels as
    # one pipeline, with the HTTP layer stubbed so no endpoint is needed. Scoring
    # calls jiwer, so this runs where the full dev dependency set is installed.
    pytest.importorskip("jiwer")

    dataset = make_dataset(
        tmp_path,
        [
            ("pane-01.jpg", "tables", "Alpha Beta", {"total": "1284"}),
            ("form-01.jpg", "forms", "Name North Harbor", {}),
        ],
    )
    monkeypatch.setenv(runner.DATASET_DIR_ENV, str(dataset))

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "Alpha Beta TOTAL 1284"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 9},
            }

    calls: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def post(self, path: str, json: dict | None = None) -> FakeResponse:
            calls.append(json or {})
            return FakeResponse()

    class FakeHttpx(ModuleType):
        AsyncClient = FakeClient

    monkeypatch.setitem(sys.modules, "httpx", FakeHttpx("httpx"))

    # Args must come from the parser: _run reads every CLI default (api_key_env
    # among them), and a hand-rolled Namespace would silently miss one.
    args = runner.build_parser().parse_args(
        [
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--model",
            "m",
            "--max-tokens",
            "128",
            "--concurrency",
            "2",
            "--repeats",
            "2",
            "--timeout",
            "5.0",
        ]
    )
    results = asyncio.run(runner._run(args, "2026-01-01T00:00:00Z"))

    assert len(calls) == 4  # 2 images x 2 repeats
    body = calls[0]
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert results["config"]["n_images"] == 2
    assert results["overall"]["n_records"] == 4
    assert results["overall"]["n_errors"] == 0
    assert set(results["labels"]) == {"tables", "forms"}
    for record in results["records"]:
        assert record["error"] is None
        assert isinstance(record["cer"], float)  # scored, never left null on success
    assert results["errors"] == []


# --- aggregation ------------------------------------------------------------------


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "file": "pane.jpg",
        "label": "tables",
        "repeat": 0,
        "cer": None,
        "wer": None,
        "field_recall": None,
        "n_fields": 0,
        "n_fields_found": 0,
        "latency_ms": 1000.0,
        "error": None,
    }
    record.update(overrides)
    return record


def test_aggregate_records_excludes_errors_from_means() -> None:
    records = [
        _record(cer=0.02, wer=0.05, latency_ms=1000.0),
        _record(cer=0.06, wer=0.10, latency_ms=1400.0, repeat=1),
        _record(error="empty_completion", latency_ms=900.0, repeat=1),
    ]
    block = runner.aggregate_records(records, wall_seconds=4.0)
    assert block["n_records"] == 3
    assert block["n_errors"] == 1
    assert block["cer"] == 0.04
    assert block["wer"] == 0.075
    assert block["seconds_per_image"] == 1.2
    assert block["error_rate"] == pytest.approx(1 / 3)
    assert block["field_recall"] is None
    assert block["wall_seconds"] == 4.0


def test_aggregate_records_weights_field_recall_by_field_count() -> None:
    records = [
        _record(n_fields=1, n_fields_found=1),
        _record(n_fields=3, n_fields_found=1, repeat=1),
    ]
    block = runner.aggregate_records(records, wall_seconds=1.0)
    # 2 of 4 fields found: the three-field image counts three times, not once.
    assert block["field_recall"] == 0.5
    assert block["n_fields"] == 4
    assert block["n_fields_found"] == 2


def test_aggregate_records_empty_group_has_zero_error_rate() -> None:
    block = runner.aggregate_records([], wall_seconds=0.0)
    assert block["n_records"] == 0
    assert block["error_rate"] == 0.0
    assert block["cer"] is None


def test_records_with_errors_summarizes_failures() -> None:
    records = [
        _record(error="empty_completion", repeat=1),
        _record(repeat=0),
    ]
    assert runner.records_with_errors(records) == ["pane.jpg repeat 1: empty_completion"]


# --- exit code ---------------------------------------------------------------------


def _stub_asyncio_run(monkeypatch: pytest.MonkeyPatch, results: dict) -> None:
    """Replace ``asyncio.run`` with one that returns ``results``.

    The real coroutine object is closed rather than awaited — the stub must not
    reach the network, and an abandoned coroutine would surface as an unraisable
    warning (an error under this repo's ``filterwarnings``).
    """

    def fake_run(coro: object) -> dict:
        coro.close()  # type: ignore[attr-defined]
        return results

    monkeypatch.setattr(runner.asyncio, "run", fake_run)


def test_main_exits_nonzero_when_any_request_erred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the async run so the exit-code rule is checked without an endpoint:
    # a run with a counted error must exit 1 even though results were written.
    results = {
        "records": [_record(error="empty_completion"), _record(repeat=1)],
        "errors": ["pane.jpg repeat 0: empty_completion"],
    }
    _stub_asyncio_run(monkeypatch, results)
    out = tmp_path / "out.json"
    argv = ["--base-url", "http://127.0.0.1:1/v1", "--model", "m", "--output", str(out)]
    assert runner.main(argv) == 1
    assert json.loads(out.read_text()) == results


def test_main_exits_zero_when_every_request_succeeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = {
        "records": [_record()],
        "errors": [],
    }
    _stub_asyncio_run(monkeypatch, results)
    out = tmp_path / "out.json"
    argv = ["--base-url", "http://127.0.0.1:1/v1", "--model", "m", "--output", str(out)]
    assert runner.main(argv) == 0


def test_main_fails_loudly_without_the_dataset_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runner.DATASET_DIR_ENV, raising=False)
    out = tmp_path / "out.json"
    argv = ["--base-url", "http://127.0.0.1:1/v1", "--model", "m", "--output", str(out)]
    assert runner.main(argv) == 2
    assert not out.exists()
