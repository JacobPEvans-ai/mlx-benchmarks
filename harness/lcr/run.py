#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28.1", "huggingface-hub>=1.28.0"]
# ///
"""aa-lcr — Artificial Analysis Long Context Reasoning (AA-LCR) runner.

Each of the 100 questions comes with a document set of roughly 100k tokens and
needs reasoning across several of its documents. The runner sends each question,
with all of its documents, to an OpenAI-compatible chat endpoint. A judge model
then decides whether the candidate answer matches the official answer, using the
equality checker that the dataset card publishes.

Run (the dataset is read at DATASET_REVISION unless --dataset-revision is given)::

    uv run harness/lcr/run.py --base-url "$BASE_URL" --model "$MODEL" \\
        --judge-base-url "$JUDGE_BASE_URL" --judge-model "$JUDGE_MODEL" \\
        --judge-api-key-env JUDGE_API_KEY --limit 5

Output is one raw-results JSON: a record per question plus totals. A request that
fails, overflows the endpoint's context window, or gets a judge reply with no
verdict is an infra error. It is recorded with its error, left out of the score,
and counted in ``n_infra_error``. It is never counted as a wrong answer.

Scoring lives in importable pure functions (``build_prompt``, ``parse_verdict``,
``summarize``, ...) so ``tests/test_lcr_runner.py`` exercises them without a live
endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import datetime
import io
import json
import os
import re
import sys
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DATASET_REPO = "ArtificialAnalysis/AA-LCR"
# Dataset revision read on the day this runner was written: the card's version 1.1
# (September 2026 update). Scores from version 1.0.0 are not comparable with it.
DATASET_REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"
QUESTIONS_FILE = "AA-LCR_Dataset.csv"
DOCUMENTS_FILE = "extracted_text/AA-LCR_extracted-text.zip"
# Archive members are stored as lcr/<document_category>/<document_set_id>/<filename>.
ARCHIVE_ROOT = "lcr"
_UTF8_FLAG = 0x800

# Candidate prompt, quoted verbatim from the "Prompt Template" section of the card:
# https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/blob/9a77ef56b717057ade24ceab4d273712a0b4f19e/README.md
PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

# Judge prompts, quoted verbatim from the "Scoring Approach" section of the card
# (the version 1.1 prompts, which add the system prompt):
# https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/blob/9a77ef56b717057ade24ceab4d273712a0b4f19e/README.md
JUDGE_SYSTEM_PROMPT = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""

JUDGE_USER_PROMPT = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""

_CONTEXT_ERROR_RE = re.compile(r"context|max_model_len|too long", re.IGNORECASE)
# Response fields a server may use to report the context window it served with.
# Names differ by server; a response that carries none of them reports null.
_SERVED_CONTEXT_KEYS = ("n_ctx", "max_model_len", "context_length")


# ---------------------------------------------------------------------------
# Pure functions (unit-tested via tests/test_lcr_runner.py)
# ---------------------------------------------------------------------------


def build_prompt(documents: Sequence[str], question: str) -> str:
    """The card's candidate prompt: every document of the set in order, then the question."""
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(documents)
    )
    return PROMPT_TEMPLATE.format(documents_text=documents_text, question=question)


def judge_messages(question: str, official_answer: str, candidate_answer: str) -> list[dict[str, str]]:
    """The card's equality-checker request: the system prompt, then the user prompt."""
    user = JUDGE_USER_PROMPT.format(
        question=question, official_answer=official_answer, candidate_answer=candidate_answer
    )
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_verdict(reply: str) -> bool | None:
    """True for CORRECT, False for INCORRECT, None when the judge gave no verdict.

    The card asks for JSON with a verdict. A bare CORRECT or INCORRECT reply, as the
    card's version 1.0.0 prompt asked for, is accepted as well.
    """
    text = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        payload = text
    if isinstance(payload, dict):
        payload = payload.get("verdict")
    if not isinstance(payload, str):
        return None
    return {"CORRECT": True, "INCORRECT": False}.get(payload.strip(" \t\n.\"'").upper())


def classify_http_error(status: int, body: str) -> str:
    """Name a failed request. A 4xx that says the prompt is too long is a context-window overflow."""
    if 400 <= status < 500 and _CONTEXT_ERROR_RE.search(body):
        return "context_window_exceeded"
    return f"http_{status}"


def served_context(payload: Mapping[str, Any]) -> int | None:
    """Context window the endpoint reports for this response, or None when it reports none."""
    for key in _SERVED_CONTEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def message_text(payload: Mapping[str, Any]) -> str:
    """Assistant content of the first choice, or "" when the response carries none."""
    choice = (payload.get("choices") or [{}])[0] or {}
    return (choice.get("message") or {}).get("content") or ""


def archive_member_key(info: zipfile.ZipInfo) -> str:
    """The member's path spelled the way the question CSV spells it.

    zipfile decodes a name without the UTF-8 flag as CP437, which turns UTF-8 bytes
    into mojibake. Reversing that decode recovers the original bytes. Names are then
    NFC-normalized, because some members are stored in decomposed form.
    """
    name = info.filename
    if (info.flag_bits & _UTF8_FLAG) == 0:
        with contextlib.suppress(UnicodeError):
            name = name.encode("cp437").decode("utf-8")
    return unicodedata.normalize("NFC", name)


def index_archive(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Every file member of the archive, keyed by archive_member_key."""
    return {archive_member_key(info): info for info in archive.infolist() if not info.is_dir()}


def load_document_set(
    archive: zipfile.ZipFile,
    index: Mapping[str, zipfile.ZipInfo],
    category: str,
    set_id: str,
    filenames: Sequence[str],
) -> list[str]:
    """Texts of one question's document set, in the order the question lists its files."""
    texts: list[str] = []
    for filename in filenames:
        key = unicodedata.normalize("NFC", f"{ARCHIVE_ROOT}/{category}/{set_id}/{filename}")
        if key not in index:
            raise KeyError(f"document missing from archive: {key}")
        # Text mode with universal newlines, as the card's loader reads the files.
        with io.TextIOWrapper(archive.open(index[key]), encoding="utf-8") as handle:
            texts.append(handle.read())
    return texts


def read_questions(path: Path) -> list[dict[str, str]]:
    """Question rows in dataset order. Answers are kept as written."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals over scored questions. Infra errors are counted, never scored as wrong."""
    verdicts = [r["correct"] for r in records if r["correct"] is not None]
    return {
        "score": sum(verdicts) / len(verdicts) if verdicts else None,
        "n_scored": len(verdicts),
        "n_infra_error": sum(1 for r in records if r["error"] is not None),
    }


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


def fetch_dataset(revision: str) -> tuple[Path, Path]:
    """Download the question CSV and the extracted-text archive at one pinned revision."""
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(
        repo_id=DATASET_REPO, filename=QUESTIONS_FILE, repo_type="dataset", revision=revision
    )
    zip_path = hf_hub_download(
        repo_id=DATASET_REPO, filename=DOCUMENTS_FILE, repo_type="dataset", revision=revision
    )
    return Path(csv_path), Path(zip_path)


# ---------------------------------------------------------------------------
# Endpoint I/O (httpx imported lazily, as in harness/factual/run.py)
# ---------------------------------------------------------------------------


async def chat(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """One chat completion. Returns (payload, None) on success, (None, error) on failure."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        response = await client.post("chat/completions", json=body)
    except Exception as exc:  # broad on purpose: any transport error is an infra error
        return None, type(exc).__name__
    if response.status_code != 200:
        return None, classify_http_error(response.status_code, response.text)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        return None, "invalid_json"
    return payload, None


async def score_question(
    answer_client: Any,
    judge_client: Any,
    args: argparse.Namespace,
    question: Mapping[str, str],
    prompt: str,
    gate: asyncio.Semaphore,
) -> dict[str, Any]:
    """Ask the candidate, then have the judge grade the answer. Failures stay per-question."""
    record: dict[str, Any] = {
        "id": question["question_id"],
        "input_tokens": None,
        "served_context": None,
        "correct": None,
        "error": None,
    }
    async with gate:
        answer, error = await chat(
            answer_client,
            args.model,
            [{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if answer is None:
            record["error"] = f"answer: {error}"
            return record
        usage = answer.get("usage")
        record["input_tokens"] = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        record["served_context"] = served_context(answer)

        # The judge runs deterministically: the card sets no sampling for it.
        judged, error = await chat(
            judge_client,
            args.judge_model,
            judge_messages(question["question"], question["answer"], message_text(answer)),
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        if judged is None:
            record["error"] = f"judge: {error}"
            return record
        verdict = parse_verdict(message_text(judged))
        if verdict is None:
            record["error"] = "judge: unparseable_verdict"
        else:
            record["correct"] = verdict
    return record


def _with_trailing_slash(url: str) -> str:
    # httpx drops the last path segment of base_url unless it ends with "/".
    return url if url.endswith("/") else url + "/"


async def run_all(
    args: argparse.Namespace, items: Sequence[tuple[Mapping[str, str], str]]
) -> list[dict[str, Any]]:
    import httpx

    api_key = os.environ.get(args.api_key_env)
    judge_key = os.environ.get(args.judge_api_key_env)
    answer_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    judge_headers = {"Authorization": f"Bearer {judge_key}"} if judge_key else {}
    judge_base = args.judge_base_url or args.base_url
    gate = asyncio.Semaphore(args.concurrency)
    async with (
        httpx.AsyncClient(
            base_url=_with_trailing_slash(args.base_url),
            headers=answer_headers,
            timeout=args.request_timeout_s,
        ) as answer_client,
        httpx.AsyncClient(
            base_url=_with_trailing_slash(judge_base),
            headers=judge_headers,
            timeout=args.request_timeout_s,
        ) as judge_client,
    ):
        return list(
            await asyncio.gather(
                *(
                    score_question(answer_client, judge_client, args, question, prompt, gate)
                    for question, prompt in items
                )
            )
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AA-LCR long-context reasoning benchmark")
    parser.add_argument(
        "--base-url", required=True, help="OpenAI-compatible /v1 base URL of the model under test"
    )
    parser.add_argument("--model", required=True, help="Model id as served by the endpoint")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="NAME of the env var holding the API key (never pass a literal key)",
    )
    parser.add_argument(
        "--require-key",
        action="store_true",
        help="Exit 2 before the first request if the API key or judge key variable is unset or empty",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Questions in flight at once")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0, help="Per-request timeout, seconds")
    parser.add_argument(
        "--limit", type=int, default=None, help="Number of questions, in dataset order (default: all)"
    )
    parser.add_argument(
        "--dataset-revision",
        default=DATASET_REVISION,
        help="Hugging Face dataset revision to read",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="OpenAI-compatible /v1 base URL of the judge (default: --base-url)",
    )
    parser.add_argument("--judge-model", required=True, help="Judge model id as served by the judge endpoint")
    parser.add_argument(
        "--judge-api-key-env",
        default="OPENAI_API_KEY",
        help="NAME of the env var holding the judge API key",
    )
    parser.add_argument("--output", type=Path, help="Results JSON path (default: aa_lcr_results_<ts>.json)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.require_key:
        missing = [name for name in (args.api_key_env, args.judge_api_key_env) if not os.environ.get(name)]
        if missing:
            print(f"--require-key: unset or empty: {', '.join(missing)}", file=sys.stderr)
            return 2

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ts = timestamp.replace(":", "").replace("-", "")
    out: Path = args.output or Path(f"aa_lcr_results_{ts}.json")

    csv_path, zip_path = fetch_dataset(args.dataset_revision)
    questions = read_questions(csv_path)[: args.limit]
    # Build every prompt before the first request, so a missing document fails the run early.
    with zipfile.ZipFile(zip_path) as archive:
        index = index_archive(archive)
        items = [
            (
                question,
                build_prompt(
                    load_document_set(
                        archive,
                        index,
                        question["document_category"],
                        question["document_set_id"],
                        question["data_source_filenames"].split(";"),
                    ),
                    question["question"],
                ),
            )
            for question in questions
        ]

    records = asyncio.run(run_all(args, items))
    results = {
        "benchmark": "aa-lcr",
        "model": args.model,
        "judge_model": args.judge_model,
        "dataset_revision": args.dataset_revision,
        "timestamp": timestamp,
        "totals": summarize(records),
        "questions": records,
    }
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
