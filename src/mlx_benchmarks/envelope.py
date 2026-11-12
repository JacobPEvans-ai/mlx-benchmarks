"""Envelope v1 types and schema validation.

The envelope is the authoritative contract between benchmark runners and
downstream consumers (the HuggingFace dataset, the Gradio viewer, any future
analysis pipeline). This module provides typed views of the envelope and a
runtime validator backed by ``schema.json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Required, TypedDict, cast

import jsonschema
from jsonschema import Draft7Validator

# model_kind value for a run served by an OpenAI-compatible cloud endpoint.
# Only this kind skips the Hub model lookup; local models keep it.
CLOUD_ENDPOINT_MODEL_KIND = "cloud-endpoint"


def model_identity(envelope: Envelope | dict[str, Any]) -> str:
    """Published identity of the run's model.

    Cloud-endpoint records use the declared ``model_provider`` plus ``model``
    (``provider/model``); local models use the Hub model id verbatim.
    """
    model = envelope["model"]
    if envelope.get("model_kind") == CLOUD_ENDPOINT_MODEL_KIND:
        provider = envelope.get("model_provider")
        if provider:
            return f"{provider}/{model}"
    return model


class TopologyNode(TypedDict, total=False):
    hostname: str
    chip: str
    memory_gb: int


class Topology(TypedDict, total=False):
    world_size: int
    parallelism: str
    interconnect: str
    nodes: list[TopologyNode]


class Gpu(TypedDict, total=False):
    model: str
    vram_gb: float
    driver: str
    cuda: str


class Engine(TypedDict, total=False):
    name: str
    version: str


class System(TypedDict, total=False):
    os: str | None
    chip: str | None
    memory_gb: int | None
    hostname: str
    vllm_mlx_version: str
    runner: str
    python_version: str
    mlx_version: str
    mlx_lm_version: str
    lm_eval_version: str
    kernel: str
    gpu: Gpu
    engine: Engine
    power_limit_w: float
    container: str
    topology: Topology


class Serving(TypedDict, total=False):
    stack: str
    endpoint_port: int
    served_model: str


class Result(TypedDict, total=False):
    name: str
    metric: str
    value: float
    unit: str
    tags: dict[str, str]
    raw: Any
    duration_seconds: float
    # Aggregate prefill throughput. Supporting detail — see schema.json for
    # the full rationale on why this is not the headline number.
    prompt_tokens_per_second: float
    # Aggregate decode-only throughput. Supporting detail, not the headline
    # number: it hides prefill-engine improvements even though a faster
    # prefill is a real, felt latency win. See total_tokens_per_second.
    decode_tokens_per_second: float
    # PRIMARY / headline throughput metric: (prompt + response) tokens over
    # duration_seconds. Prefer this field when reporting or comparing models.
    total_tokens_per_second: float
    first_token_latency_ms: float
    peak_rss_mb: float
    output_tokens_mean: float | None
    output_tokens_median: float | None
    output_tokens_p90: float | None
    reasoning_tokens_mean: float | None
    tokens_per_correct: float | None
    truncation_rate: float | None
    reasoning_arm: str


class GenKwargs(TypedDict, total=False):
    max_gen_toks: int
    temperature: float
    top_p: float
    top_k: int


class Campaign(TypedDict, total=False):
    id: str
    cell_id: str
    profile: str


class CampaignDimensions(TypedDict, total=False):
    environment: dict[str, Any]
    hardware: dict[str, Any]
    software: dict[str, Any]
    model: dict[str, Any]
    run: dict[str, Any]
    speed: dict[str, Any]
    resource: dict[str, Any]
    quality: dict[str, Any]
    provenance: dict[str, Any]
    cost: dict[str, Any]


class ContextDimensions(TypedDict, total=False):
    model_max_tokens: int
    catalog_max_tokens: int
    proxy_max_tokens: int
    worker_max_tokens: int
    configured_window_tokens: int
    requested_prompt_tokens: int
    actual_prompt_tokens: int
    synthetic_repetitions: int
    output_reservation_tokens: int


class Envelope(TypedDict, total=False):
    # Keys marked Required[...] mirror schema.json's `required` list — they are
    # always present on a validated envelope (and `LmEvalConverter.build_envelope`
    # always sets them), so indexing them is safe. The rest stay optional.
    schema_version: Required[str]
    timestamp: Required[str]
    git_sha: Required[str | None]
    trigger: Required[str]
    pr_number: int | None
    env_class: str
    concurrency: int
    reasoning_effort: str
    serving: Serving
    suite: Required[str]
    model: Required[str]
    model_task: str
    model_task_source: Literal["model_card", "inferred"]
    model_revision: str
    model_kind: Literal["local", "cloud-endpoint"]
    model_provider: str
    quantization: str
    campaign: Campaign
    campaign_dimensions: CampaignDimensions
    dimension_null_reasons: dict[str, str]
    cell_status: str
    context: ContextDimensions
    skipped: bool
    seed: int
    gen_kwargs: GenKwargs
    system: Required[System]
    results: Required[list[Result]]
    memory_snapshots: list[dict[str, Any]]
    errors: list[str]


class EnvelopeValidationError(ValueError):
    """Raised when an envelope fails schema validation."""

    def __init__(self, errors: list[jsonschema.ValidationError]) -> None:
        self.errors = errors
        messages = "\n".join(
            f"  - {e.message} (at ${'.'.join(str(p) for p in e.absolute_path)})" for e in errors
        )
        super().__init__(f"Envelope failed schema validation:\n{messages}")


def _find_schema_path() -> Path:
    """Resolve schema.json whether running from source tree or installed wheel.

    Two supported layouts:

    1. **Installed wheel** (``pip install mlx-benchmarks``): the build copies
       ``schema.json`` into the package directory via
       ``[tool.hatch.build.targets.wheel.force-include]``. ``importlib.resources``
       locates it reliably inside site-packages.
    2. **Source tree** (editable install or running from the repo directly):
       the schema lives at the repo root; walk upward from this file.

    We try (1) first so installed-package behavior is the fast path; fall back
    to (2) for development. Failure is noisy — a missing schema means the
    publisher cannot validate envelopes.
    """
    try:
        pkg_file = resources.files("mlx_benchmarks") / "schema.json"
        if pkg_file.is_file():
            return Path(str(pkg_file))
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "schema.json"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate schema.json — expected alongside the installed "
        "mlx_benchmarks package (force-include) or at repository root."
    )


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Load and cache ``schema.json``."""
    with _find_schema_path().open() as f:
        loaded: dict[str, Any] = json.load(f)
        return loaded


@lru_cache(maxsize=1)
def _validator() -> Draft7Validator:
    """Cached validator with format checking enabled.

    Without ``format_checker=``, jsonschema silently accepts any string for
    ``format: date-time`` / URI / regex / etc. The envelope schema relies on
    ISO-8601 timestamps; downstream consumers (the viewer's
    ``pd.to_datetime`` call) assume that contract holds.
    """
    schema = load_schema()
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema, format_checker=Draft7Validator.FORMAT_CHECKER)


def validate_envelope(envelope: Envelope | dict[str, Any]) -> None:
    """Raise :class:`EnvelopeValidationError` if the envelope is invalid.

    Collects every validation error rather than failing on the first —
    gives publishers a complete picture to fix in one pass.
    """
    # An Envelope/dict is a JSON object at runtime; cast to the plain mapping
    # jsonschema's iter_errors expects (its param type rejects TypedDicts).
    errors: list[jsonschema.ValidationError] = sorted(
        _validator().iter_errors(cast(dict[str, Any], envelope)),
        key=lambda e: list(e.absolute_path),
    )
    if errors:
        raise EnvelopeValidationError(errors)


def iter_validation_errors(envelope: Envelope | dict[str, Any]) -> Iterable[jsonschema.ValidationError]:
    """Yield schema errors without raising — useful for best-effort downgrade to ``errors[]``."""
    yield from _validator().iter_errors(cast(dict[str, Any], envelope))
