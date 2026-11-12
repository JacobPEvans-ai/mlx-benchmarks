#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28.1"]
# ///
"""agentic — many-tool tool-call reliability benchmark.

Fires realistic chat/completions requests carrying a 22-tool registry at any
OpenAI-compatible endpoint and measures whether the model produces *valid*
structured tool calls under load: a full matrix of thinking levels,
concurrency 1/4, small/large context, and streaming/non-streaming — plus a
multi-turn degradation track (mlx-lm #1011: stock 4-bit quants fall back to
plain-text ``[Tool call: ...]`` around round 5).

Run (never against a busy Studio without asking)::

    uv run harness/agentic/run.py --base-url http://localhost:11434/v1 \\
        --model mlx-community/Qwen3.6-35B-A3B-4bit --api-key-env OPENAI_API_KEY

Output is one raw-results JSON; publish it with
``mlx-bench-publish <out.json> --kind agentic --suite tool-calling``.

Response parsing and validity classification live in importable pure
functions (``classify_message``, ``assemble_stream``, ...) so
``tests/test_agentic_runner.py`` can exercise them without a live endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import statistics
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tool registry — 22 realistic tools. The registry SIZE is the load under
# test: production failures only reproduce with real-sized schemas, and the
# near-duplicate distractors force the model to pick among similar names.
# ---------------------------------------------------------------------------


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "run_splunk_query",
        "Execute an SPL search against the Splunk REST API and return the result rows. "
        "Use for any log investigation, aggregation, or timechart over indexed events.",
        {
            "query": {
                "type": "string",
                "description": "Full SPL query, e.g. 'search index=main error | stats count by host'",
            },
            "earliest_time": {
                "type": "string",
                "description": "Splunk relative or ISO time, e.g. '-24h' or '2026-07-01T00:00:00'",
            },
            "latest_time": {"type": "string", "description": "Splunk relative or ISO time, default 'now'"},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50000,
                "description": "Row cap, default 100",
            },
        },
        ["query"],
    ),
    _tool(
        "get_splunk_indexes",
        "List all Splunk indexes visible to the current role, with event counts and retention.",
        {
            "include_internal": {"type": "boolean", "description": "Include _internal/_audit indexes"},
        },
        [],
    ),
    _tool(
        "get_splunk_sourcetypes",
        "List sourcetypes present in a Splunk index over the recent window.",
        {
            "index": {"type": "string", "description": "Index name, e.g. 'linux_secure'"},
            "window": {"type": "string", "description": "Lookback window, e.g. '7d'", "default": "24h"},
        },
        ["index"],
    ),
    _tool(
        "read_file",
        "Read a UTF-8 text file from the workspace filesystem and return its full contents.",
        {
            "path": {"type": "string", "description": "Absolute file path"},
            "encoding": {"type": "string", "enum": ["utf-8", "latin-1"], "default": "utf-8"},
        },
        ["path"],
    ),
    _tool(
        "write_file",
        "Write text content to a file on the workspace filesystem, creating parent directories.",
        {
            "path": {"type": "string", "description": "Absolute file path"},
            "content": {"type": "string", "description": "Full new file contents"},
            "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
        },
        ["path", "content"],
    ),
    _tool(
        "list_directory",
        "List entries in a directory with type and size metadata.",
        {
            "path": {"type": "string", "description": "Absolute directory path"},
            "recursive": {"type": "boolean", "default": False},
            "include_hidden": {"type": "boolean", "default": False},
        },
        ["path"],
    ),
    _tool(
        "execute_shell_command",
        "Run a shell command on the host and return stdout, stderr, and exit code. "
        "Prefer dedicated tools (read_file, run_splunk_query) when one exists.",
        {
            "command": {"type": "string", "description": "POSIX shell command line"},
            "working_directory": {"type": "string", "description": "cwd for the command"},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600, "default": 60},
        },
        ["command"],
    ),
    _tool(
        "search_memory",
        "Semantic search over the agent's long-term memory store; returns the top matching notes.",
        {
            "query": {"type": "string", "description": "Natural-language search query"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
            "namespace": {"type": "string", "description": "Memory namespace, e.g. 'incidents'"},
        },
        ["query"],
    ),
    _tool(
        "store_memory",
        "Persist a note to the agent's long-term memory store under a stable key.",
        {
            "key": {"type": "string", "description": "Stable identifier for later retrieval"},
            "content": {"type": "string", "description": "Note body (markdown allowed)"},
            "namespace": {"type": "string", "description": "Memory namespace, e.g. 'incidents'"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Free-form labels"},
        },
        ["key", "content"],
    ),
    _tool(
        "read_wiki_page",
        "Fetch a wiki page by title and return its markdown source.",
        {
            "title": {"type": "string", "description": "Exact page title"},
            "revision": {"type": "integer", "description": "Specific revision id; latest when omitted"},
        },
        ["title"],
    ),
    _tool(
        "write_wiki_page",
        "Create or replace a wiki page with new markdown content.",
        {
            "title": {"type": "string", "description": "Exact page title"},
            "content": {"type": "string", "description": "Full markdown body"},
            "summary": {"type": "string", "description": "Edit summary for page history"},
        },
        ["title", "content"],
    ),
    _tool(
        "post_slack_message",
        "Post a message to a Slack channel (public or private) as the agent bot user.",
        {
            "channel": {"type": "string", "description": "Channel name with #, e.g. '#incident-bridge'"},
            "text": {"type": "string", "description": "Message body (Slack mrkdwn)"},
            "thread_ts": {"type": "string", "description": "Parent message ts to reply in-thread"},
        },
        ["channel", "text"],
    ),
    _tool(
        "create_cron_job",
        "Register a named scheduled job that runs a command on a cron schedule.",
        {
            "name": {"type": "string", "description": "Unique job name, kebab-case"},
            "schedule": {"type": "string", "description": "5-field cron expression, e.g. '30 2 * * *'"},
            "command": {"type": "string", "description": "Command line to execute"},
            "enabled": {"type": "boolean", "default": True},
        },
        ["name", "schedule", "command"],
    ),
    _tool(
        "list_cron_jobs",
        "List registered scheduled jobs with schedule, last run, and next run.",
        {
            "include_disabled": {"type": "boolean", "default": False},
        },
        [],
    ),
    _tool(
        "remove_cron_job",
        "Delete a scheduled job by name. Irreversible.",
        {
            "name": {"type": "string", "description": "Exact job name"},
        },
        ["name"],
    ),
    _tool(
        "fetch_web_page",
        "Fetch a URL over HTTPS and return the page text (markdown-converted).",
        {
            "url": {"type": "string", "description": "Absolute https:// URL"},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
            "render_javascript": {"type": "boolean", "default": False},
        },
        ["url"],
    ),
    _tool(
        "get_system_status",
        "Return health/status for a named platform component (queue depth, error rate, uptime).",
        {
            "component": {"type": "string", "description": "Component name, e.g. 'ingest-gateway'"},
        },
        [],
    ),
    # -- near-duplicate distractors ------------------------------------------------
    _tool(
        "search_splunk_events",
        "Simple keyword event search in Splunk (no SPL). For full SPL use run_splunk_query.",
        {
            "search_string": {"type": "string", "description": "Plain keyword string, no SPL syntax"},
            "index": {"type": "string", "description": "Restrict to one index"},
            "time_range": {"type": "string", "enum": ["1h", "24h", "7d", "30d"], "default": "24h"},
        },
        ["search_string"],
    ),
    _tool(
        "read_file_chunk",
        "Read a byte range of a large file. For whole small files use read_file.",
        {
            "path": {"type": "string", "description": "Absolute file path"},
            "offset": {"type": "integer", "minimum": 0, "description": "Start byte offset"},
            "length": {"type": "integer", "minimum": 1, "description": "Bytes to read"},
        },
        ["path", "offset", "length"],
    ),
    _tool(
        "grep_files",
        "Regex search across files under a directory; returns matching lines with paths.",
        {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "Directory to search"},
            "case_sensitive": {"type": "boolean", "default": True},
        },
        ["pattern", "path"],
    ),
    _tool(
        "send_slack_dm",
        "Send a direct message to one Slack user. For channels use post_slack_message.",
        {
            "user_id": {"type": "string", "description": "Slack member id, e.g. 'U0123ABCD'"},
            "text": {"type": "string", "description": "Message body"},
        },
        ["user_id", "text"],
    ),
    _tool(
        "query_memory_graph",
        "Run a graph query against the memory knowledge graph. For text search use search_memory.",
        {
            "cypher": {"type": "string", "description": "Cypher query string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 25},
        },
        ["cypher"],
    ),
]

# Single-shot scenarios: every prompt SHOULD produce a tool call, and several
# force a choice among the near-duplicates above.
SCENARIOS: list[str] = [
    "Investigate the spike in failed SSH logins over the last 24 hours: run a Splunk search over "
    "index=linux_secure that counts failures by source host, worst offenders first.",
    "Read the deployment config at /etc/app/deploy.yaml so we can check the rollout settings.",
    "Post a status update to the #incident-bridge channel saying the ingest backlog has cleared "
    "and dashboards are green again.",
    "Search long-term memory for previous incidents involving certificate expiry on the ingest tier.",
    "Schedule a job named nightly-backup that runs /usr/local/bin/backup.sh every day at 02:30.",
]

# Multi-turn follow-up tasks — cycled so each round has a fresh tool-shaped ask.
MULTITURN_TASKS: list[str] = [
    *SCENARIOS,
    "Now list which Splunk indexes we actually have, so we know where else to look.",
    "Check what sourcetypes exist in the linux_secure index over the last 7 days.",
    "List the files under /var/log/app so we can see which logs rotated recently.",
    "Fetch https://status.example.com/api/incidents and summarize any open incidents.",
    "Store a memory note under key ssh-bruteforce-2026-07 describing what we found so far.",
    "Read the wiki page titled 'Runbook: SSH bruteforce response'.",
    "Search the /etc/fail2ban directory for any file mentioning 'maxretry' using a regex search.",
    "Check the current health of the ingest-gateway component.",
    "List all scheduled jobs including disabled ones.",
    "Update the wiki page 'Runbook: SSH bruteforce response' appending today's findings section.",
]

VALID = "valid"
FAILURE_KINDS = (
    "no_tool_call",
    "empty_function_name",
    "bad_json_args",
    "unknown_tool",
    "http_error",
    "bad_response_body",
    "timeout",
    "stream_truncated",
)


# ---------------------------------------------------------------------------
# Pure functions (unit-tested via tests/test_agentic_runner.py)
# ---------------------------------------------------------------------------


def required_params(tools: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Map tool name -> list of required argument keys."""
    out: dict[str, list[str]] = {}
    for tool in tools:
        fn = tool.get("function") or {}
        params = fn.get("parameters") or {}
        out[fn["name"]] = list(params.get("required") or [])
    return out


def classify_message(
    message: Mapping[str, Any] | None,
    finish_reason: str | None,
    required: Mapping[str, Sequence[str]],
) -> str:
    """Classify one assistant response into ``valid`` or a failure kind.

    A response is valid iff it carries at least one tool call whose function
    name is non-empty AND in the registry, whose arguments parse as a JSON
    object containing every required key — and the request finished with
    ``finish_reason == "tool_calls"`` (anything else means the call was cut
    off or the server mislabeled it; production truncation looked exactly
    like this).
    """
    tool_calls = (message or {}).get("tool_calls") or []
    if not tool_calls:
        return "no_tool_call"
    for call in tool_calls:
        fn = call.get("function") or {}
        name = (fn.get("name") or "").strip()
        if not name:
            return "empty_function_name"
        if name not in required:
            return "unknown_tool"
        try:
            args = json.loads(fn.get("arguments") or "")
        except (json.JSONDecodeError, TypeError):
            return "bad_json_args"
        if not isinstance(args, dict) or any(key not in args for key in required[name]):
            return "bad_json_args"
    if finish_reason != "tool_calls":
        return "stream_truncated"
    return VALID


def has_output_delta(delta: Mapping[str, Any]) -> bool:
    """Whether an SSE delta carries generated text or a tool-call fragment."""
    if any(delta.get(field) for field in ("content", "reasoning_content", "refusal")):
        return True
    for call in delta.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") or function.get("arguments"):
            return True
    return False


def assemble_stream(
    events: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    """Assemble streamed chat-completion chunks into (message, finish_reason, usage).

    Mirrors what a real client must do: concatenate ``delta.content`` /
    ``delta.reasoning_content`` and merge ``delta.tool_calls`` fragments by
    ``index`` (id and name arrive once, arguments arrive as string pieces).
    """
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    for event in events:
        if event.get("usage"):
            usage = dict(event["usage"])
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        for frag in delta.get("tool_calls") or []:
            idx = frag.get("index", 0)
            entry = tool_calls.setdefault(
                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if frag.get("id"):
                entry["id"] = frag["id"]
            fn = frag.get("function") or {}
            if fn.get("name"):
                entry["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                entry["function"]["arguments"] += fn["arguments"]

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return message, finish_reason, usage


def approx_tokens(text: str) -> int:
    """Cheap token estimate (chars/4) — good enough to size synthetic context."""
    return len(text) // 4


def build_history(target_tokens: int) -> list[dict[str, Any]]:
    """Synthetic prior tool-exchange history totalling ~``target_tokens`` tokens.

    Repeated realistic (user ask -> assistant tool_call -> tool result)
    triplets, with matching tool_call ids, so large-context cells exercise the
    same message shapes a long-lived agent session accumulates.
    """
    result_rows = json.dumps(
        [
            {
                "_time": f"2026-07-07T0{h}:12:{s:02d}",
                "host": f"web-{h:02d}",
                "source": "/var/log/auth.log",
                "sourcetype": "linux_secure",
                "action": "failure",
                "src_ip": f"203.0.113.{s}",
                "count": 40 + s,
            }
            for h in range(3)
            for s in range(10)
        ]
    )
    history: list[dict[str, Any]] = []
    total = 0
    i = 0
    while total < target_tokens:
        call_id = f"call_hist_{i:04d}"
        block: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Pull the auth failure summary for batch {i} and note anything unusual.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "run_splunk_query",
                            "arguments": json.dumps(
                                {
                                    "query": "search index=linux_secure action=failure "
                                    f"batch={i} | stats count by host, src_ip | sort -count",
                                    "earliest_time": "-24h",
                                    "max_results": 100,
                                }
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": result_rows},
            {
                "role": "assistant",
                "content": f"Batch {i}: failures concentrated on the web tier from a small set of "
                "source addresses; counts are elevated but consistent with the earlier batches. "
                "No new hosts appeared and no successes followed the failure bursts.",
            },
        ]
        history.extend(block)
        total += sum(approx_tokens(json.dumps(m)) for m in block)
        i += 1
    return history


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; single sorted pass, no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]


def inter_chunk_gaps_ms(received_at: Sequence[float]) -> list[float]:
    """Milliseconds between output-bearing SSE chunks."""
    return [(current - previous) * 1000 for previous, current in pairwise(received_at)]


def summarize_cell(records: Sequence[Mapping[str, Any]], wall_seconds: float) -> dict[str, Any]:
    """Aggregate per-request records into the per-cell metric block."""
    n = len(records)
    failures = {kind: sum(1 for r in records if r["outcome"] == kind) for kind in FAILURE_KINDS}
    latencies = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
    first_tokens = [r["first_token_ms"] for r in records if r.get("first_token_ms") is not None]
    per_request_tps = [
        r["completion_tokens"] / (r["latency_ms"] / 1000)
        for r in records
        if r.get("completion_tokens") and r["latency_ms"]
    ]
    total_completion = sum(r.get("completion_tokens") or 0 for r in records)
    return {
        "n_requests": n,
        "wall_seconds": round(wall_seconds, 3),
        "valid_tool_call_rate": (sum(1 for r in records if r["outcome"] == VALID) / n) if n else 0.0,
        "finish_reason_tool_calls_rate": (
            sum(1 for r in records if r.get("finish_reason") == "tool_calls") / n if n else 0.0
        ),
        "reasoning_present_rate": (sum(1 for r in records if r.get("reasoning_present")) / n) if n else 0.0,
        "failures": failures,
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "first_token_p50_ms": round(percentile(first_tokens, 50), 1) if first_tokens else None,
        "effective_tokens_per_second": round(statistics.mean(per_request_tps), 2)
        if per_request_tps
        else None,
        "aggregate_tokens_per_second": round(total_completion / wall_seconds, 2)
        if wall_seconds > 0
        else None,
        "http_429": sum(1 for r in records if r.get("http_status") == 429),
        "http_5xx": sum(1 for r in records if (r.get("http_status") or 0) >= 500),
        "requests": list(records),
    }


def cell_name(concurrency: int, thinking: str, context: str, stream: bool) -> str:
    """Cell identity, and a public join key.

    ``--cells`` filters on it, the crash-recovery JSONL is keyed by it, and it is
    published as ``tags.cell``. So ``on`` and ``off`` keep the exact names they
    have always produced — renaming them would orphan every published row — and a
    graded level simply names its own cell.
    """
    return f"conc{concurrency}_think-{thinking}_ctx-{context}_{'stream' if stream else 'nostream'}"


def synth_tool_result(call: Mapping[str, Any]) -> dict[str, Any]:
    """Synthesize a plausible tool-result message for a returned tool call."""
    fn = call.get("function") or {}
    return {
        "role": "tool",
        "tool_call_id": call.get("id") or "call_unknown",
        "content": json.dumps({"ok": True, "tool": fn.get("name"), "rows": 3, "summary": "completed"}),
    }


# ---------------------------------------------------------------------------
# Endpoint I/O (httpx imported lazily — the module stays importable in the
# test env, where only the pure functions above are exercised)
# ---------------------------------------------------------------------------


def thinking_body_kwargs(kwarg: str, level: str) -> dict[str, Any]:
    """Per-family thinking control: chat_template_kwargs bool, or reasoning_effort.

    ``on``/``off`` keep the exact bodies they have always sent, so a re-run of a
    published cell is byte-identical. Any other level is passed through verbatim —
    the point of a graded sweep is that ``high`` and ``xhigh`` reach the model as
    themselves rather than collapsing into one request.
    """
    if kwarg == "reasoning_effort":
        # ponytail: harmony models can't fully disable reasoning; low is the off-analog
        if level in ("on", "off"):
            return {"reasoning_effort": "high" if level == "on" else "low"}
        return {"reasoning_effort": level}
    if level in ("on", "off"):
        return {"chat_template_kwargs": {kwarg: level == "on"}}
    return {"chat_template_kwargs": {kwarg: level}}


async def one_request(
    client: Any,
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    thinking: str,
    stream: bool,
    required: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": stream,
        **thinking_body_kwargs(args.thinking_kwarg, thinking),
    }
    if args.repetition_penalty is not None:
        body["repetition_penalty"] = args.repetition_penalty
    if stream:
        body["stream_options"] = {"include_usage": True}

    import httpx

    record: dict[str, Any] = {
        "outcome": "http_error",
        "finish_reason": None,
        "reasoning_present": False,
        "latency_ms": None,
        "first_token_ms": None,
        "inter_chunk_gap_p50_ms": None,
        "inter_chunk_gap_p95_ms": None,
        "inter_chunk_gap_max_ms": None,
        "completion_tokens": None,
        "http_status": None,
    }
    start = time.monotonic()
    try:
        if stream:
            events: list[dict[str, Any]] = []
            output_times: list[float] = []
            async with client.stream("POST", "/chat/completions", json=body) as response:
                record["http_status"] = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line.removeprefix("data:").strip()
                    if not payload or payload == "[DONE]":
                        continue
                    event = json.loads(payload)
                    choices = event.get("choices") or []
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    if has_output_delta(delta):
                        output_times.append(time.monotonic())
                    events.append(event)
            message, finish_reason, usage = assemble_stream(events)
            if output_times:
                record["first_token_ms"] = round((output_times[0] - start) * 1000, 1)
                gaps = inter_chunk_gaps_ms(output_times)
                if gaps:
                    record["inter_chunk_gap_p50_ms"] = round(percentile(gaps, 50), 1)
                    record["inter_chunk_gap_p95_ms"] = round(percentile(gaps, 95), 1)
                    record["inter_chunk_gap_max_ms"] = round(max(gaps), 1)
        else:
            response = await client.post("/chat/completions", json=body)
            record["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage")
    except httpx.TimeoutException:
        record["outcome"] = "timeout"
        record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return record
    except json.JSONDecodeError:
        # Malformed/truncated body: a bad SSE chunk (json.loads on a stream
        # fragment) or a non-JSON 200 (response.json()). Record the sample as
        # failed instead of letting it bubble up and kill the whole matrix —
        # results are only written after every cell finishes, so an uncaught
        # decode error mid-run would lose every completed cell.
        record["outcome"] = "bad_response_body"
        record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return record
    except httpx.HTTPError:
        record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return record

    record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    record["finish_reason"] = finish_reason
    record["reasoning_present"] = bool(message.get("reasoning_content"))
    record["completion_tokens"] = (usage or {}).get("completion_tokens")
    record["outcome"] = classify_message(message, finish_reason, required)
    record["message"] = message  # callers that build multi-turn history need it
    return record


async def run_cell(
    client: Any,
    args: argparse.Namespace,
    concurrency: int,
    thinking: str,
    context: str,
    stream: bool,
    required: Mapping[str, Sequence[str]],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    prefix = history if context == "large" else []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(i: int) -> dict[str, Any]:
        messages = [*prefix, {"role": "user", "content": SCENARIOS[i % len(SCENARIOS)]}]
        async with semaphore:
            record = await one_request(client, args, messages, thinking, stream, required)
        record.pop("message", None)  # keep the results JSON compact
        return record

    # Untimed warm-up: one request excluded from every statistic. The first
    # request after a context/thinking switch pays cold-start cost (prompt-cache
    # miss, weight residency) that would otherwise skew this cell's numbers.
    warmup = [*prefix, {"role": "user", "content": SCENARIOS[0]}]
    await one_request(client, args, warmup, thinking, stream, required)

    start = time.monotonic()
    records = await asyncio.gather(*(bounded(i) for i in range(args.repeats)))
    wall = time.monotonic() - start
    return {
        "name": cell_name(concurrency, thinking, context, stream),
        "concurrency": concurrency,
        "thinking": thinking,
        "context": context,
        "stream": stream,
        **summarize_cell(records, wall),
    }


async def run_multiturn(
    client: Any,
    args: argparse.Namespace,
    thinking: str,
    required: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    first_degraded: int | None = None
    for round_no in range(1, args.multiturn_rounds + 1):
        messages.append({"role": "user", "content": MULTITURN_TASKS[(round_no - 1) % len(MULTITURN_TASKS)]})
        record = await one_request(client, args, messages, thinking, stream=False, required=required)
        message = record.pop("message", None) or {"role": "assistant", "content": ""}
        rounds.append(
            {
                "round": round_no,
                "outcome": record["outcome"],
                "finish_reason": record["finish_reason"],
                "reasoning_present": record["reasoning_present"],
            }
        )
        if record["outcome"] != VALID and first_degraded is None:
            first_degraded = round_no
        # Extend history the way a real agent loop does: the assistant message
        # as returned, plus a synthesized result for every tool call it made.
        messages.append({k: v for k, v in message.items() if k != "reasoning_content"})
        for call in message.get("tool_calls") or []:
            messages.append(synth_tool_result(call))
    return {"thinking": thinking, "rounds": rounds, "first_degraded_round": first_degraded}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Many-tool tool-call reliability benchmark")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="NAME of the env var holding the API key (never pass a literal key)",
    )
    parser.add_argument("--model", required=True, help="Model id as served by the endpoint")
    parser.add_argument(
        "--cells",
        help="Comma-separated substrings; only cells whose name matches one run "
        "(include 'multiturn' to keep the multi-turn track)",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Requests per cell")
    parser.add_argument("--concurrency", default="1,4", help="Comma list of in-flight request counts")
    parser.add_argument(
        "--thinking",
        default="on,off",
        help="Comma list of effort levels, sent verbatim. 'on'/'off' keep their "
        "per-family mapping and cell names; any other value (low/high/xhigh/max) "
        "is passed straight through and names its own cell",
    )
    parser.add_argument("--context", default="small,large", help="Comma list from {small,large}")
    parser.add_argument("--stream", default="stream,nostream", help="Comma list from {stream,nostream}")
    parser.add_argument("--multiturn-rounds", type=int, default=20)
    parser.add_argument(
        "--dedicated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="declare that this arm was the ONLY consumer of its endpoint. Cannot "
        "be detected — the harness cannot see what else is on the box — so it is "
        "declared, and defaults to false because an unstated measurement "
        "environment is an untrusted one. Runs are comparable only within one "
        "value of this flag",
    )
    parser.add_argument(
        "--thinking-kwarg",
        default="enable_thinking",
        help="chat_template_kwargs bool name, or 'reasoning_effort' for harmony models",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Send repetition_penalty in the request body (omitted if unset)",
    )
    parser.add_argument("--timeout", type=float, default=1200, help="Per-request timeout, seconds")
    parser.add_argument("--large-context-tokens", type=int, default=20000)
    parser.add_argument("--output", type=Path, help="Results JSON path (default: agentic_results_<ts>.json)")
    return parser


def _selected(name: str, cells_filter: str | None) -> bool:
    if not cells_filter:
        return True
    return any(part.strip() and part.strip() in name for part in cells_filter.split(","))


def _append_partial(path: Path, kind: str, obj: Mapping[str, Any]) -> None:
    """Append one completed unit to a crash-recovery JSONL beside the output.

    The final results JSON is only written after the whole matrix finishes, so
    without this a late failure loses every completed cell. Each line is a
    standalone JSON object tagged by ``kind`` (``cell`` or ``multiturn``); on a
    clean run the file is deleted and the canonical JSON supersedes it.
    """
    with path.open("a") as f:
        f.write(json.dumps({"kind": kind, **obj}) + "\n")


async def _run(args: argparse.Namespace, timestamp: str, partial_path: Path) -> dict[str, Any]:
    import httpx

    required = required_params(TOOLS)
    history = build_history(args.large_context_tokens)
    # Verbatim, never coerced. The `== "on"` form this replaced turned every
    # unrecognised level into False, so `--thinking high,xhigh` ran the same cell
    # twice under one name and reported no error at all.
    thinking_modes = [m.strip() for m in args.thinking.split(",") if m.strip()]
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    contexts = [c.strip() for c in args.context.split(",") if c.strip()]
    streams = [s.strip() == "stream" for s in args.stream.split(",") if s.strip()]

    headers = {}
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    cells: list[dict[str, Any]] = []
    multiturn: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), headers=headers, timeout=args.timeout
    ) as client:
        for concurrency in concurrencies:
            for thinking in thinking_modes:
                for context in contexts:
                    for stream in streams:
                        name = cell_name(concurrency, thinking, context, stream)
                        if not _selected(name, args.cells):
                            continue
                        print(f"cell {name} ...", file=sys.stderr)
                        cell = await run_cell(
                            client, args, concurrency, thinking, context, stream, required, history
                        )
                        cells.append(cell)
                        _append_partial(partial_path, "cell", cell)
        for thinking in thinking_modes:
            name = f"multiturn_think-{thinking}"
            if not _selected(name, args.cells):
                continue
            print(f"track {name} ...", file=sys.stderr)
            track = await run_multiturn(client, args, thinking, required)
            multiturn.append(track)
            _append_partial(partial_path, "multiturn", track)

    return {
        "benchmark": "agentic",
        "model": args.model,
        "timestamp": timestamp,
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "n_tools": len(TOOLS),
            "repeats": args.repeats,
            "concurrency": concurrencies,
            "thinking": list(thinking_modes),
            "context": contexts,
            "stream": [("stream" if s else "nostream") for s in streams],
            "multiturn_rounds": args.multiturn_rounds,
            "thinking_kwarg": args.thinking_kwarg,
            # Declared, never inferred: the harness cannot see what else is on
            # the box. A tool-calling rate measured while another consumer holds
            # the endpoint's only slot describes the sharing, not the model, so
            # runs compare only within one value of this.
            "dedicated": args.dedicated,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "timeout": args.timeout,
            "large_context_tokens": args.large_context_tokens,
        },
        "cells": cells,
        "multiturn": multiturn,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ts = timestamp.replace(":", "").replace("-", "")
    out: Path = args.output or Path(f"agentic_results_{ts}.json")
    partial_path = Path(f"{out}.partial.jsonl")
    results = asyncio.run(_run(args, timestamp, partial_path))
    out.write_text(json.dumps(results, indent=2) + "\n")
    partial_path.unlink(missing_ok=True)  # clean run: canonical JSON supersedes the crash log
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
