"""Unit tests for the AA-LCR runner: prompt assembly, judge parsing, infra accounting.

``harness/lcr/run.py`` is a standalone PEP 723 script (not part of the package),
so it is loaded here via importlib, as ``test_factual_runner.py`` does. Nothing in
this module touches the network: the dataset comes from ``tests/fixtures/lcr_sample/``
and the endpoints are in-memory fakes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import zipfile
from argparse import Namespace
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "lcr_sample"
DOCUMENTS_DIR = FIXTURE_DIR / "documents"


def _load_runner() -> ModuleType:
    path = REPO_ROOT / "harness" / "lcr" / "run.py"
    spec = importlib.util.spec_from_file_location("lcr_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


@pytest.fixture
def archive(tmp_path: Path) -> Iterator[zipfile.ZipFile]:
    """The fixture documents packed the way the dataset archive stores them."""
    path = tmp_path / "extracted_text.zip"
    with zipfile.ZipFile(path, "w") as writer:
        for doc in sorted(DOCUMENTS_DIR.rglob("*.txt")):
            writer.write(doc, "lcr/" + doc.relative_to(DOCUMENTS_DIR).as_posix())
    with zipfile.ZipFile(path) as reader:
        yield reader


# --- prompt assembly ----------------------------------------------------------


def test_build_prompt_follows_the_card_template() -> None:
    prompt = runner.build_prompt(["DOC ONE", "DOC TWO"], "What is X?")
    assert prompt == (
        "BEGIN INPUT DOCUMENTS\n\n"
        "BEGIN DOCUMENT 1:\nDOC ONE\nEND DOCUMENT 1\n\n"
        "BEGIN DOCUMENT 2:\nDOC TWO\nEND DOCUMENT 2\n\n"
        "END INPUT DOCUMENTS\n\n"
        "Answer the following question using the input documents provided above.\n\n"
        "START QUESTION\n\nWhat is X?\n\nEND QUESTION\n"
    )


def test_fixture_rows_reference_existing_documents() -> None:
    rows = runner.read_questions(FIXTURE_DIR / "AA-LCR_Dataset.csv")
    assert [row["question_id"] for row in rows] == ["1", "2"]
    for row in rows:
        for name in row["data_source_filenames"].split(";"):
            path = DOCUMENTS_DIR / row["document_category"] / row["document_set_id"] / name
            assert path.is_file(), path


def test_document_set_follows_the_question_order(archive: zipfile.ZipFile) -> None:
    index = runner.index_archive(archive)
    docs = runner.load_document_set(
        archive, index, "Company_Documents", "fake_set_a", ["fund_beta_2024.txt", "fund_alpha_2024.txt"]
    )
    assert [doc.splitlines()[0] for doc in docs] == [
        "Fund Beta annual letter for 2024.",
        "Fund Alpha annual letter for 2024.",
    ]


def test_question_prompt_places_documents_before_the_question(archive: zipfile.ZipFile) -> None:
    row = runner.read_questions(FIXTURE_DIR / "AA-LCR_Dataset.csv")[0]
    index = runner.index_archive(archive)
    docs = runner.load_document_set(
        archive,
        index,
        row["document_category"],
        row["document_set_id"],
        row["data_source_filenames"].split(";"),
    )
    prompt = runner.build_prompt(docs, row["question"])
    assert prompt.index("BEGIN DOCUMENT 1:\nFund Alpha") < prompt.index("BEGIN DOCUMENT 2:\nFund Beta")
    assert prompt.index("END DOCUMENT 2") < prompt.index("START QUESTION")


def test_missing_document_is_an_error_not_a_skip(archive: zipfile.ZipFile) -> None:
    index = runner.index_archive(archive)
    with pytest.raises(KeyError, match=r"nope\.txt"):
        runner.load_document_set(archive, index, "Marketing", "fake_set_b", ["nope.txt"])


def test_archive_name_recovers_cp437_mojibake() -> None:
    # zipfile reads a name without the UTF-8 flag as CP437, which mangles UTF-8 bytes.
    original = "What the EU AI Act means for you\u2014and how to prepare.txt"
    mojibake = original.encode("utf-8").decode("cp437")
    info = zipfile.ZipInfo(f"lcr/Legal/legal_eu_ai/{mojibake}")
    assert runner.archive_member_key(info) == f"lcr/Legal/legal_eu_ai/{original}"


def test_archive_name_is_normalized_to_nfc() -> None:
    decomposed = "Ba" + "s\u0327" + "ev.txt"  # s followed by a combining cedilla
    expected = "lcr/Marketing/x/Ba\u015fev.txt"
    # With the UTF-8 flag set the name is taken as stored, so only normalization applies.
    flagged = zipfile.ZipInfo(f"lcr/Marketing/x/{decomposed}")
    flagged.flag_bits |= 0x800
    assert runner.archive_member_key(flagged) == expected
    # Without the flag, the stored bytes are recovered from CP437 first, then normalized.
    unflagged = zipfile.ZipInfo(f"lcr/Marketing/x/{decomposed.encode('utf-8').decode('cp437')}")
    assert runner.archive_member_key(unflagged) == expected


# --- judge request and verdict parsing ----------------------------------------


def test_judge_request_carries_the_card_prompts() -> None:
    system, user = runner.judge_messages("Which?", "OFFICIAL", "CANDIDATE")
    assert system == {"role": "system", "content": runner.JUDGE_SYSTEM_PROMPT}
    assert "START QUESTION Which?" in user["content"]
    assert "The OFFICIAL ANSWER: OFFICIAL" in user["content"]
    assert (
        "BEGIN CANDIDATE ANSWER TO ASSESS\n\nCANDIDATE\n\nEND CANDIDATE ANSWER TO ASSESS" in user["content"]
    )
    assert user["content"].endswith("Reply as JSON, with a verdict of CORRECT or INCORRECT.")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"verdict": "CORRECT"}', True),
        ('```json\n{"verdict": "INCORRECT"}\n```', False),
        ('{"verdict": "correct"}', True),
        ("CORRECT", True),
        ("incorrect.", False),
        ('{"verdict": "maybe"}', None),
        ('{"correct": true}', None),
        ("I think it is correct", None),
        ("", None),
    ],
)
def test_parse_verdict(reply: str, expected: bool | None) -> None:
    assert runner.parse_verdict(reply) is expected


# --- error classification and served context ----------------------------------


def test_context_overflow_is_classified_apart_from_other_failures() -> None:
    overflow = "This model's maximum context length is 8192 tokens."
    assert runner.classify_http_error(400, overflow) == "context_window_exceeded"
    assert runner.classify_http_error(400, '{"error": "bad json"}') == "http_400"
    assert runner.classify_http_error(500, "context window exceeded") == "http_500"


def test_served_context_is_null_unless_the_endpoint_reports_it() -> None:
    assert runner.served_context({"usage": {"prompt_tokens": 10}}) is None
    assert runner.served_context({"n_ctx": 32768}) == 32768
    assert runner.served_context({"n_ctx": True}) is None


# --- infra-error accounting ---------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class FakeClient:
    """Stands in for httpx.AsyncClient. Each request goes to one handler."""

    def __init__(self, handler: Callable[[dict[str, Any]], FakeResponse]) -> None:
        self._handler = handler

    async def post(self, path: str, **kwargs: Any) -> FakeResponse:
        assert path == "chat/completions"
        return self._handler(kwargs["json"])


def _chat_reply(content: str) -> FakeResponse:
    return FakeResponse(
        payload={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 1234}}
    )


def _route(body: dict[str, Any]) -> FakeResponse:
    """Candidate prompts carry a [case: ...] tag; the candidate's answer carries the verdict case."""
    if body["model"] == "judge":
        candidate = body["messages"][1]["content"]
        if "ANSWER-TIMEOUT" in candidate:
            raise TimeoutError("judge timed out")
        if "ANSWER-GARBLE" in candidate:
            return _chat_reply("maybe")
        verdict = "CORRECT" if "ANSWER-RIGHT" in candidate else "INCORRECT"
        return _chat_reply(f'{{"verdict": "{verdict}"}}')

    prompt = body["messages"][0]["content"]
    if "[case: overflow]" in prompt:
        return FakeResponse(400, text="This model's maximum context length is 8192 tokens.")
    answers = {
        "right": "ANSWER-RIGHT",
        "wrong": "ANSWER-WRONG",
        "timeout": "ANSWER-TIMEOUT",
        "garble": "ANSWER-GARBLE",
    }
    for tag, answer in answers.items():
        if f"[case: {tag}]" in prompt:
            return _chat_reply(answer)
    raise AssertionError(f"no route for prompt: {prompt[:60]!r}")


def _args() -> Namespace:
    return Namespace(model="candidate", judge_model="judge", max_tokens=16, temperature=0.0)


def _score_all(cases: list[str]) -> list[dict[str, Any]]:
    client = FakeClient(_route)

    async def run() -> list[dict[str, Any]]:
        gate = asyncio.Semaphore(2)
        items = [
            runner.score_question(
                client,
                client,
                _args(),
                {"question_id": str(number), "question": "q", "answer": "a"},
                f"[case: {tag}] prompt",
                gate,
            )
            for number, tag in enumerate(cases, start=1)
        ]
        return list(await asyncio.gather(*items))

    return asyncio.run(run())


def test_infra_errors_are_counted_apart_from_wrong_answers() -> None:
    records = _score_all(["right", "wrong", "overflow", "timeout", "garble"])

    assert records[0] == {
        "id": "1",
        "input_tokens": 1234,
        "served_context": None,
        "correct": True,
        "error": None,
    }
    assert records[1]["correct"] is False and records[1]["error"] is None
    assert records[2]["correct"] is None and records[2]["error"] == "answer: context_window_exceeded"
    assert records[3]["correct"] is None and records[3]["error"] == "judge: TimeoutError"
    assert records[4]["correct"] is None and records[4]["error"] == "judge: unparseable_verdict"

    assert runner.summarize(records) == {"score": 0.5, "n_scored": 2, "n_infra_error": 3}


def test_summarize_with_nothing_scored_has_no_score() -> None:
    assert runner.summarize([]) == {"score": None, "n_scored": 0, "n_infra_error": 0}
    only_infra = [{"correct": None, "error": "answer: http_500"}]
    assert runner.summarize(only_infra) == {"score": None, "n_scored": 0, "n_infra_error": 1}


def test_judge_receives_the_card_system_prompt_on_every_request() -> None:
    seen: list[dict[str, Any]] = []

    def recording(body: dict[str, Any]) -> FakeResponse:
        seen.append(body)
        return _route(body)

    client = FakeClient(recording)

    async def run() -> None:
        await runner.score_question(
            client,
            client,
            _args(),
            {"question_id": "1", "question": "q", "answer": "a"},
            "[case: right] prompt",
            asyncio.Semaphore(1),
        )

    asyncio.run(run())
    judge_requests = [body for body in seen if body["model"] == "judge"]
    assert len(judge_requests) == 1
    assert judge_requests[0]["messages"][0] == {"role": "system", "content": runner.JUDGE_SYSTEM_PROMPT}
    assert judge_requests[0]["temperature"] == 0.0


# --- --require-key ------------------------------------------------------------


def _fail_if_fetched(revision: str) -> tuple[Path, Path]:
    raise AssertionError(f"dataset fetched before the key check: {revision}")


def _base_argv(*extra: str) -> list[str]:
    return [
        "--base-url",
        "http://unused.invalid/v1",
        "--model",
        "candidate",
        "--judge-model",
        "judge",
        "--api-key-env",
        "LCR_TEST_KEY",
        "--judge-api-key-env",
        "LCR_TEST_JUDGE_KEY",
        "--require-key",
        *extra,
    ]


def test_require_key_exits_2_when_the_api_key_is_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LCR_TEST_KEY", raising=False)
    monkeypatch.setenv("LCR_TEST_JUDGE_KEY", "judge-secret-value")
    monkeypatch.setattr(runner, "fetch_dataset", _fail_if_fetched)
    assert runner.main(_base_argv()) == 2
    assert "LCR_TEST_KEY" in capsys.readouterr().err


def test_require_key_treats_an_empty_judge_key_as_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LCR_TEST_KEY", "candidate-secret-value")
    monkeypatch.setenv("LCR_TEST_JUDGE_KEY", "")
    monkeypatch.setattr(runner, "fetch_dataset", _fail_if_fetched)
    assert runner.main(_base_argv()) == 2
    err = capsys.readouterr().err
    assert "LCR_TEST_JUDGE_KEY" in err and "LCR_TEST_KEY" not in err


def test_require_key_passes_once_both_keys_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LCR_TEST_KEY", "candidate-secret-value")
    monkeypatch.setenv("LCR_TEST_JUDGE_KEY", "judge-secret-value")

    def stop_after_gate(revision: str) -> tuple[Path, Path]:
        raise RuntimeError("reached dataset fetch")

    monkeypatch.setattr(runner, "fetch_dataset", stop_after_gate)
    with pytest.raises(RuntimeError, match="reached dataset fetch"):
        runner.main(_base_argv())
