# mlx-benchmarks

[![ci-gate][badge-ci]][workflow-ci]
[![Release Please][badge-rp]][workflow-rp]
[![Schema v1][badge-schema]][schema-file]
[![Python 3.13+][badge-py]][python-downloads]
[![License: Apache 2.0][badge-license]](LICENSE)
[![HF Dataset][badge-hfds]][hf-dataset]
[![HF Space][badge-hfsp]][hf-space]

[badge-ci]: https://github.com/JacobPEvans/mlx-benchmarks/actions/workflows/ci-gate.yml/badge.svg?branch=main
[workflow-ci]: https://github.com/JacobPEvans/mlx-benchmarks/actions/workflows/ci-gate.yml
[badge-rp]: https://github.com/JacobPEvans/mlx-benchmarks/actions/workflows/release-please.yml/badge.svg?branch=main
[workflow-rp]: https://github.com/JacobPEvans/mlx-benchmarks/actions/workflows/release-please.yml
[badge-schema]: https://img.shields.io/badge/schema-v1-4FB3A9
[schema-file]: schema.json
[badge-py]: https://img.shields.io/badge/python-3.13%2B-blue
[python-downloads]: https://www.python.org/downloads/
[badge-license]: https://img.shields.io/badge/license-Apache--2.0-green.svg
[badge-hfds]: https://img.shields.io/badge/%F0%9F%A4%97%20dataset-JacobPEvans%2Fmlx--benchmarks-yellow
[hf-dataset]: https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks
[badge-hfsp]: https://img.shields.io/badge/%F0%9F%A4%97%20space-viewer-yellow
[hf-space]: https://huggingface.co/spaces/JacobPEvans/mlx-benchmarks-viewer

A reproducible benchmark harness for **MLX-quantized** and **locally-hosted**
LLMs on Apple Silicon. One envelope schema, one HuggingFace dataset, one
interactive viewer — across every upstream evaluation tool.

**Why not just use lm-eval directly?** You can. This repo wraps lm-eval (and
other harnesses) with:

- A **single versioned result contract** (`schema.json`) so every shard is
  comparable across tools, models, and dates.
- A **publish pipeline** (`mlx-bench-publish`) that validates envelopes
  against the schema and uploads to the
  [HF dataset](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks)
  with content-addressed filenames.
- A **Gradio viewer** (in [`space/`](space/)) auto-deployed to an
  [HF Space](https://huggingface.co/spaces/JacobPEvans/mlx-benchmarks-viewer)
  on every `main` push.

Read results as a pandas DataFrame with no tooling beyond `huggingface_hub` +
`pyarrow`.

## Requirements

This repo owns the result contract and publish pipeline; it delegates model
serving and evaluation execution to external tools:

- **An OpenAI-compatible inference endpoint** (default
  `http://localhost:11434/v1`) — any server speaking the OpenAI chat/completions
  API works. This repo does not start, manage, or assume a specific inference
  server; it only sends requests to the configured base URL.
- **An evaluation driver** ([lm-eval](#upstream-tools-wired-in) for accuracy or
  vllm's `benchmark_serving` for throughput) that produces a raw results file.
  The converters in `src/mlx_benchmarks/converters/` translate each driver's
  native output into the versioned envelope.
- **A HuggingFace token** with write scope on the target dataset, for the
  publish step only.

## Architecture

The repo owns the **envelope contract** (`schema.json`), the **publisher**
(`mlx-bench-publish`), and the **converters** that fan in from each upstream
evaluation tool. Its own data flow:

```mermaid
%%{init: {
  'theme':'base',
  'look':'handDrawn',
  'themeVariables':{
    'fontFamily':'Geist',
    'fontSize':'14px',
    'primaryColor':'#102937',
    'primaryTextColor':'#F4EFE6',
    'primaryBorderColor':'#4FB3A9',
    'lineColor':'#4FB3A9',
    'secondaryColor':'#0B1D2A',
    'tertiaryColor':'#1A2A38',
    'clusterBkg':'rgba(79,179,169,0.08)',
    'clusterBorder':'#4FB3A9'
  }
}}%%
flowchart LR
  Raw(["raw results_*.json"])
  Convert([converter])
  Envelope([validated envelope])
  Publish([mlx-bench-publish])
  Dataset[("HF dataset")]
  Viewer([HF Space viewer])

  Raw -->|"convert"| Convert
  Convert -->|"build_envelope"| Envelope
  Envelope -->|"schema check"| Publish
  Publish -->|"parquet upload"| Dataset
  Dataset --> Viewer

  classDef source fill:#102937,stroke:#E06B4A,stroke-width:2.5px,color:#F4EFE6;
  classDef core   fill:#102937,stroke:#4FB3A9,stroke-width:3px,color:#F4EFE6;
  classDef sink   fill:#102937,stroke:#F4EFE6,stroke-width:2.5px,color:#F4EFE6;

  class Raw source
  class Convert,Envelope,Publish core
  class Dataset,Viewer sink

  linkStyle 0,1,2,3,4 stroke:#4FB3A9,stroke-width:2px;
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed component
breakdown, data-flow, and CI diagrams.

## Upstream tools wired in

Accuracy and throughput suites run on external tools —
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and
[vllm `benchmark_serving`](https://docs.vllm.ai/en/latest/performance/benchmarks.html)
— invoked through thin `uvx` wrappers in the serving stack (nix-ai `mlx-eval` /
`mlx-bench`), not scripts in this repo. The `tool-calling`, `promptstack`,
`grounded-summary`, and AA-LCR long-context suites are the exceptions: standalone
PEP 723 runners under `harness/`, documented in [`docs/agentic.md`](docs/agentic.md),
[`docs/promptstack.md`](docs/promptstack.md), and
[`docs/shootout.md`](docs/shootout.md).

[`configs/LAYOUT.md`](configs/LAYOUT.md) is the single source of truth for which
suites are wired to which tool.

## NVIDIA campaign

Campaign runbooks, envelope fields, baseline suites, and the relaxed verdict
gate are described in [`docs/nvidia-campaign.md`](docs/nvidia-campaign.md).

## Benchmarking playbook

To benchmark **any** model on either Apple Silicon host, follow
**[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**; traps and the serving parser map are in
[`docs/benchmark-traps.md`](docs/benchmark-traps.md), the leaderboard in
**[`RANKINGS.md`](RANKINGS.md)**. Benchmarks come in three kinds — **throughput**
(headline: cumulative tok/s, not decode-only — see
[docs/schema.md](docs/schema.md)), **accuracy** (`coding` / `math-hard` / `reasoning` via lm-eval),
and **agentic** (`tool-calling` via [`harness/agentic/run.py`](harness/agentic/run.py))
— a model is "fully benchmarked" with a published shard for each.

[Standards](docs/benchmark-standards.md).

Speed and intelligence weigh equally; see
[`power_limit_w`](docs/nvidia-campaign.md).

## Repository layout

```text
.
├── README.md · CLAUDE.md · CONTRIBUTING.md · SECURITY.md · LICENSE
├── RANKINGS.md               <- model leaderboard
├── schema.json               <- envelope v1 (authoritative)
├── examples/                 <- envelope fixtures + host walkthroughs
├── pyproject.toml            <- package + lint/type/test config
├── src/mlx_benchmarks/       <- publisher, envelope, system, CLI,
│                                shootout ranker, converters/ (per --kind)
├── tests/                    <- package tests + fixtures
├── configs/                  <- one runbook per (tool, suite); see LAYOUT.md
│                                (one per suite, plus shootout/)
├── harness/                  <- standalone PEP 723 runners, per suite
├── scripts/                  <- schema validator
├── space/                    <- Gradio viewer (deployed to HF Space)
├── docs/                     <- RUNBOOK.md, architecture.md, schema.md, faq.md, journal/
└── .github/workflows/        <- ci-gate, release-please, deploy-space
```

## Installation

Requires macOS on Apple Silicon (for inference) and Python 3.13+.

```sh
git clone https://github.com/JacobPEvans/mlx-benchmarks.git
cd mlx-benchmarks

# Plain uv (recommended)
uv sync
# ...or plain pip into a venv
python -m venv .venv && source .venv/bin/activate && pip install -e .

# The Gradio result viewer (space/) installs its own deps separately:
#   pip install -r space/requirements.txt

# Install pre-commit hooks (optional but encouraged)
.venv/bin/pre-commit install
```

For Nix users: `direnv allow` activates the included `flake.nix` dev shell.

## Usage

### Run + publish a benchmark

This repo does not run models — it publishes the output of a run. Point an
OpenAI-compatible endpoint at `http://localhost:11434/v1`, drive it with a
standard tool, then convert + publish (the full per-host procedure is
[`docs/RUNBOOK.md`](docs/RUNBOOK.md)). An accuracy run with lm-eval:

```sh
# 1. Run lm-eval against the endpoint (your own install — this repo only parses
#    its JSON output). Swap in vllm benchmark_serving for --kind vllm.
MODEL="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
lm_eval --model local-chat-completions \
  --model_args "base_url=http://localhost:11434/v1/chat/completions,model=$MODEL,num_concurrent=4,max_length=32768,timeout=3600" \
  --tasks gsm8k --apply_chat_template --log_samples --output_path ./run-output

# 2. Dry-run the conversion (validates against schema.json; no upload)
.venv/bin/mlx-bench-publish ./run-output/<model-dir>/results_*.json \
  --kind lm-eval --suite reasoning \
  --published-metadata ./run-output/<model-dir>/published-metadata.json --dry-run

# 3. Publish — the ambient HF_TOKEN is read-only, so pass the write token
HF_TOKEN="$HF_WRITE_TOKEN" \
  .venv/bin/mlx-bench-publish ./run-output/<model-dir>/results_*.json \
  --kind lm-eval --suite reasoning \
  --published-metadata ./run-output/<model-dir>/published-metadata.json
```

The published-result contract, metadata source mapping, and migration procedure
are documented in [`docs/published-results.md`](docs/published-results.md).

`detect_system()` collects machine identity locally for run handling, while the
public projection omits host and network identifiers. Filenames are content-addressed
(`data/run-canonical-run-<timestamp>-<git_sha>-<suite>-<model_slug>-<hash>.parquet`)
so historical shards are never overwritten. Older source shards remain
available at their original paths; normalized copies keep the default dataset
schema stable.

### View results

Open the live HF Space:
<https://huggingface.co/spaces/JacobPEvans/mlx-benchmarks-viewer>

Or run the viewer locally:

```sh
cd space
pip install -r requirements.txt
python app.py
```

## API

### The envelope

See [`schema.json`](schema.json) — it is the authoritative, versioned contract
backing every published shard. A minimal valid envelope:

```json
{
  "schema_version": "1",
  "timestamp": "2026-04-24T18:30:00Z",
  "git_sha": "aaa3ff3",
  "trigger": "local",
  "suite": "reasoning",
  "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
  "system": {"os": "macOS 26.4.1", "chip": "Apple M4 Max", "memory_gb": 128},
  "results": [
    {"name": "gsm8k_cot_zeroshot", "metric": "exact_match_flexible",
     "value": 0.8, "unit": "ratio"}
  ]
}
```

Envelope v1 also accepts a set of optional fields, which the CLI auto-detects
at publish time — no hand-curation required.

See the [schema](docs/schema.md) and [cache fields](docs/cache-fields.md) guides
for field definitions, types, and units; [schema migration](docs/schema-migration.md)
for version upgrades; [model notes](docs/model-notes.md) for serving details; and
the [FAQ](docs/faq.md) for operational questions.

### The publisher

```python
from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.publish import publish
from mlx_benchmarks.system import detect_system

ctx = ConverterContext(
    suite="reasoning",
    model="mlx-community/Qwen3.5-9B-MLX-4bit",
    git_sha="aaa3ff3",
    system=detect_system(),
)
envelope = get_converter("lm-eval").build_envelope(raw_results, ctx)
publish(envelope, dry_run=False)  # validates + uploads
```

### Reading the dataset

```python
from datasets import load_dataset

ds = load_dataset("JacobPEvans/mlx-benchmarks")
print(ds["train"][0])
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full developer workflow.
Keep orchestration glue thin — if integrating a new upstream tool requires
more than ~50 lines of Python, re-read the tool's docs before writing code.

## Security

HF tokens, the `--confirm_run_unsafe_code` lm-eval flag, and the disclosure
policy are covered in [`SECURITY.md`](SECURITY.md).

## License

Apache 2.0. See [`LICENSE`](LICENSE).

---

> Part of a larger homelab ecosystem — [docs.jacobpevans.com](https://docs.jacobpevans.com).
