# Benchmark platform roadmap

## Purpose

Keep `mlx-benchmarks` a trustworthy Apple-Silicon benchmark evidence layer,
not a second implementation of every evaluator, workflow engine, experiment
tracker, and dashboard. Upstream tools should execute evaluations; this
repository should retain only the metadata, safety checks, and publication
rules that make its results comparable and reproducible.

## Current boundary

The repository already uses upstream evaluation engines for the standard
work: `lm-eval` for quality suites and vLLM-compatible serving benchmarks for
load measurements. Its custom publisher converts their heterogeneous output
into the versioned envelope, validates it, and publishes immutable Parquet
shards to the canonical Hugging Face dataset. The Space is a read-only view of
that dataset.

That boundary is sound. The growth risk is bespoke orchestration. Of the 927
lines in `scripts/`, 685 lines (74%) are suite/campaign coordination around
existing tools. Including the model-inventory helper, 779 lines (84%) are
generic workflow-shaped code. This is a maintainability signal, not a claim
that the local constraints are unimportant: endpoint identity, Apple-Silicon
capacity, isolated versus under-load evidence, replication policy, and
immutable publication remain repository-specific responsibilities.

## Target architecture

```text
Upstream executors                 Evidence layer                     Review surface

lm-eval / vLLM benchmarks  ->  envelope + provenance  ->  Hugging Face dataset
Inspect AI / Promptfoo      ->  validation + immutable  ->  Space or a tracker
                               shard + Apple-Silicon gates
```

The repository owns the middle column. It does not grow replacements for the
left or right columns without a demonstrated gap in the adopted tools.

## Tool decisions

- **Standard quality and throughput:** retain `lm-eval` and serving-engine
  benchmarks. Keep run configuration and output converters; do not duplicate
  execution engines.
- **Tool-using and long-horizon agents:** prefer
  [Inspect AI](https://inspect.aisi.org.uk/) and its agent, checkpoint, limit,
  and evaluator primitives over a new agent runner.
- **Broad task catalog or generic quality runs:** evaluate
  [LightEval](https://huggingface.co/docs/lighteval/) before adding generic
  suite runners; retain only required converters and provenance.
- **Prompt and factual regression:** follow the adopted
  [prompt-eval framework](prompt-eval-framework.md) and use
  [Promptfoo](https://www.promptfoo.dev/) (`promptstack` runner retired; historical converter retained).
- **Self-hosted traces and experiment review:** use
  [Langfuse](https://langfuse.com/) when available, while retaining the
  canonical public benchmark dataset.
- **Metrics, artifacts, and experiment comparisons:** evaluate
  [MLflow](https://mlflow.org/) rather than adding a second custom dashboard.
- **Paid collaborative evaluation UX:** consider
  [Braintrust](https://www.braintrust.dev/) or Weave only when their hosted
  review experience justifies moving non-sensitive metadata off the
  self-hosted/public path.
- **Large general benchmark platform:** use
  [OpenCompass](https://github.com/open-compass/opencompass) for breadth or
  fleet-scale work, not to replace Mac-specific safety rules.

These tools overlap roughly 55–70% of the repository's feature surface:
standard evaluation execution, task matrices, agent/prompt evaluation, result
tracking, and comparative UI. The remaining 30–45% is integration and policy;
the irreducible part is the provenance and operating contract, not the shell
orchestration itself.

## Roadmap

### 1. Freeze the boundary

New code must first use an upstream evaluator, tracker, or workflow primitive
when one already exists. The existing contributor rule is the default: if an
upstream integration needs more than about 50 lines of Python, re-read that
tool's documentation before adding the wrapper.

### 2. Consolidate generic evaluation

Use Inspect AI for new agentic evaluation work and Promptfoo for prompt/factual
regression work. Keep only adapters that preserve required envelope fields and
the published historical records. Do not delete historical shards or remove a
suite until its successor has equivalent coverage and a validated publication
path.

### 3. Choose one experiment-review system per use case

Use Langfuse for self-hosted runtime traces and experiment history. Evaluate
MLflow only for general metric/artifact tracking, or a paid service only for a
collaborative review capability that Langfuse does not provide. Avoid operating
multiple overlapping dashboards for the same evidence.

### 4. Shrink orchestration deliberately

Replace only script responsibilities that an adopted tool can prove in a real
run: matrix execution, retries/checkpoints, result persistence, or comparison.
Retain preflight checks that protect a live Apple-Silicon serving host and the
replication/environment-class gates in [the verdict policy](verdict-policy.md).

### 5. Preserve one canonical evidence path

Every retained or adopted execution path must produce schema-valid envelopes,
immutable dataset shards, and the metadata required to distinguish model,
revision, serving configuration, host, concurrency, and environment class.
The dataset remains the source of truth; dashboards are views, not a second
results store.

## Non-goals

- Replacing `lm-eval`, vLLM, Inspect AI, Promptfoo, or a selected experiment
  tracker with locally maintained equivalents.
- Reclassifying a single run as a benchmark verdict; the replication and
  dual-environment maturity gates remain mandatory.
- Moving host topology, credentials, or operational incident detail into this
  public repository.

## Success measure

The repository gets smaller or stays flat as evaluation coverage expands:
new suites reuse an adopted executor, custom scripts remain limited to proven
local gaps, and every published comparison remains reproducible from canonical
raw output and immutable dataset evidence.
