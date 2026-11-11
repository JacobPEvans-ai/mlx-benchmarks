# Architecture

High-level data flow, from "I kick off a benchmark" to "I see a chart".

## System topology

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
  subgraph Local["Apple-Silicon box"]
    Serve([vllm-mlx + llama-swap])
    Eval([lm-eval · vllm benchmark_serving])
    CLI([mlx-bench-publish])
  end

  subgraph CI["GitHub Actions"]
    Gate([ci-gate])
    RP([release-please])
    Deploy([deploy-space])
  end

  subgraph HF["HuggingFace Hub"]
    Dataset[("dataset: mlx-benchmarks")]
    Space([Space: viewer])
  end

  Users((Viewer users))

  Serve -->|":11434/v1"| Eval
  Eval -->|"results_*.json"| CLI
  CLI -->|"validate + create_commit"| Dataset
  Deploy -->|"sync space/"| Space
  Dataset --> Space
  Space -->|"read-only"| Users
  Gate -.-> CLI
  RP -.-> Deploy

  classDef stack  fill:#102937,stroke:#4FB3A9,stroke-width:2px,color:#F4EFE6;
  classDef core   fill:#102937,stroke:#4FB3A9,stroke-width:3px,color:#F4EFE6;
  classDef gate   fill:#102937,stroke:#E06B4A,stroke-width:2.5px,color:#F4EFE6;
  classDef store  fill:#102937,stroke:#F4EFE6,stroke-width:2.5px,color:#F4EFE6;
  classDef actor  fill:#102937,stroke:#E6B35A,stroke-width:2px,color:#F4EFE6;

  class Serve,Eval,Deploy stack
  class CLI core
  class Gate,RP gate
  class Dataset,Space store
  class Users actor

  linkStyle 0,1,2,3,4,5 stroke:#4FB3A9,stroke-width:2px;
  linkStyle 6,7 stroke:#E6B35A,stroke-width:1.5px,stroke-dasharray:2 4;
```

## Components

### Inference layer

`vllm-mlx` served via `llama-swap` on port 11434. OpenAI-compatible; every
tool in this repo talks to it via `http://localhost:11434/v1/chat/completions`.
Model switching is handled by `mlx-switch` / `sync-mlx-models` outside this
repo.

### Evaluation layer

Everything under `configs/` (a TOML runbook per suite). The run commands are the
serving stack's `mlx-eval` / `mlx-bench` wrappers (lm-eval / vllm
`benchmark_serving`), not scripts in this repo. No config-to-arg translation
layer lives here; the TOML is the runbook. The tau2-bench customer-service suite
runs upstream `tau2 run` the same way; see [`tau2.md`](tau2.md).

### Publisher (`src/mlx_benchmarks/`)

Pure-Python, no Apple-Silicon dependencies. Converts raw tool output into
the envelope v1 shape, validates it against `schema.json`, serializes to
Parquet, and uploads via `HfApi.create_commit` with a deterministic
content-addressed filename pattern:

```text
data/run-<ISO-timestamp>-<git_sha>-<suite>-<model_slug>-<payload_hash>.parquet
```

This guarantees idempotent re-publishes and no overwrites. The envelope
validator is invoked inside `publish()` — you cannot accidentally ship a
non-compliant shard.

#### Data flow: results.json → published shard

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
flowchart TD
  Run([lm-eval / vllm benchmark_serving])
  Raw[("results_*.json")]
  CLI([mlx-bench-publish])
  Detect([detect_system])
  Convert([get_converter.build_envelope])
  Validate{validate_envelope}
  Target([target_path])
  Parquet[("data/run-*.parquet")]
  HF[(HF dataset)]
  Fail([raise SchemaValidationError])

  Run -->|"emit"| Raw
  Raw --> CLI
  Detect --> CLI
  CLI --> Convert
  Convert -->|"envelope v1"| Validate
  Validate -->|"OK"| Target
  Validate -. "schema fail" .-> Fail
  Target -->|"deterministic name"| Parquet
  Parquet -->|"HfApi.create_commit"| HF

  classDef source fill:#102937,stroke:#E06B4A,stroke-width:2.5px,color:#F4EFE6;
  classDef stack  fill:#102937,stroke:#4FB3A9,stroke-width:2px,color:#F4EFE6;
  classDef gate   fill:#102937,stroke:#E06B4A,stroke-width:2.5px,color:#F4EFE6;
  classDef store  fill:#102937,stroke:#F4EFE6,stroke-width:2.5px,color:#F4EFE6;
  classDef err fill:#102937,stroke:#E06B4A,stroke-dasharray:4 3,color:#F4EFE6;

  class Run source
  class CLI,Convert,Detect,Target stack
  class Validate gate
  class Raw,Parquet,HF store
  class Fail err

  linkStyle 0,1,2,3,4,5,7,8 stroke:#4FB3A9,stroke-width:2px;
  linkStyle 6 stroke:#E06B4A,stroke-width:1.5px,stroke-dasharray:2 4;
```

Every score passes envelope validation and the `published_result` definition
in `schema.json` before a target path is planned. The validator requires a
recorded model SHA, every required run variable, and an explicit reason for
each null. Evaluation rows also require a registered HF Benchmark task. There
is no validation bypass. Dry runs read pinned model and benchmark metadata from
the Hub but do not write to it.

### Viewer (`space/`)

Gradio app that loads canonical Parquet shards and the run index from the HF
dataset snapshot. It prepares both viewer modes and caches the rows for 600
seconds. After expiry, it serves the current frame while a background refresh
runs. It renders a bar chart, trend chart, and summary table. The
[`Space README`](../space/README.md#performance-and-remeasurement) records
the live load baseline, targets, and remeasurement steps. The viewer deploys
to HF Spaces through `.github/workflows/deploy-space.yml` on `main` pushes
that touch `space/`.

### CI

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `ci-gate.yml` | PR | Single merge gate (see below). |
| `deploy-space.yml` | main push to listed paths; daily | Normalize shards, refresh `data/recent/latest.parquet` and the card, sync viewer to HF Space. |
| `release-please.yml` | main push | Conventional-commit releases. |

`ci-gate.yml` detects file changes and conditionally runs:

- `python-test` (ruff + pyright + pytest matrix)
- `schema-validate` (Draft-07 + TOML)
- `dry-run-publish` (publisher round-trip on fixture)
- the central reusables `_python-security.yml` (pip-audit), `_osv-scan.yml`
  (OSV lockfile scan), `_markdown-lint.yml`, `_file-size.yml`.

The final `Merge Gate` step (`re-actors/alls-green`) is the only required
check in branch protection.

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
  PR((PR opened))
  Paths{paths-filter}
  PyTest([python-test])
  Schema([schema-validate])
  Dry([dry-run-publish])
  Audit([pip-audit])
  OSV([osv-scan])
  MD([markdown-lint])
  Gate{alls-green Merge Gate}
  Ready([Mergeable])

  PR --> Paths
  Paths --> PyTest
  Paths --> Schema
  Paths --> Dry
  Paths --> Audit
  Paths --> OSV
  Paths --> MD
  PyTest --> Gate
  Schema --> Gate
  Dry --> Gate
  Audit --> Gate
  OSV --> Gate
  MD --> Gate
  Gate --> Ready

  classDef source fill:#102937,stroke:#E06B4A,stroke-width:2.5px,color:#F4EFE6;
  classDef stack  fill:#102937,stroke:#4FB3A9,stroke-width:2px,color:#F4EFE6;
  classDef gate   fill:#102937,stroke:#E06B4A,stroke-width:2.5px,color:#F4EFE6;
  classDef sink   fill:#102937,stroke:#F4EFE6,stroke-width:2.5px,color:#F4EFE6;

  class PR source
  class Paths,PyTest,Schema,Dry,Audit,OSV,MD stack
  class Gate gate
  class Ready sink

  linkStyle 0,1,2,3,4,5,6,7,8,9,10,11,12,13 stroke:#4FB3A9,stroke-width:2px;
```

Each job is skipped when paths-filter detects no relevant file changes,
but the Merge Gate still requires every job's status to be `success` or
`skipped` — it cannot pass on `failure`. This is what lets the polish
PR ship with only `markdown-lint` actually executing.

CodeQL Python + Actions scanning is provided by GitHub's
**default CodeQL setup** (repo Security settings), not by a workflow file in
this repo. A previous attempt to add a custom `codeql.yml` workflow conflicted
with the default setup ("CodeQL analyses from advanced configurations cannot
be processed when the default setup is enabled") and was removed.

## Reproducibility contract

A published shard records the full context needed to replay:

- `git_sha` — state of this repo at run time.
- `system.*` — OS, chip, memory, plus (optional) `hostname` (distinguishes
  machines with identical chip/memory), `python_version`, `mlx_version`,
  `mlx_lm_version`, `lm_eval_version`, `kernel`.
- `gen_kwargs` — generation hyperparameters passed to the inference API.
- `seed` — when the run was seeded.
- `model_revision` / `quantization` — model-side metadata when reported.

The CLI fills these in automatically; never hand-curate unless you know why.

## Non-goals

- Running benchmarks in CI. MLX requires macOS on Apple Silicon; GitHub's
  macOS runners do not offer the hardware cheaply enough. CI tests the
  **publisher**, not the benchmarks themselves.
- A custom benchmark harness. If a measurement is possible via an existing
  upstream tool, wire the tool — do not reimplement.
- Overwriting history. Published shards are immutable; a new run becomes a
  new shard.
