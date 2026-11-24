# LLM prompt-eval framework — decision record

Status: adopted 2026-07-19. Session artifacts (the ready-to-run scaffold prompt
and goal doc) live in
[`journal/2026-07-19-prompt-eval-framework.md`](journal/2026-07-19-prompt-eval-framework.md).

## Problem

Comparing slightly-different system prompts (Hermes agent brain, API apps)
needs to be quantifiable and cheap, with outputs readable at a glance
(web UI, GitHub Action PR comment, or markdown). Targets, in priority order:
the homelab local LLM fabric (OpenAI-compatible endpoints), the Anthropic API,
and OpenRouter. All canonical prompts live in `dryvist/ai-llm-prompts` (OKF
catalog, pinned releases). Volume is under 1k evals/week; free tiers suffice.

## Decision

**Engine: [promptfoo](https://github.com/promptfoo/promptfoo)**, in a new
dedicated repo **`dryvist/llm-prompt-evals`**. Rationale (verified July 2026):

- Purpose-built for exactly this shape: a declarative YAML matrix of
  prompts × providers × tests, with deterministic assertions
  (contains/regex/is-json/latency/cost/…) plus model-graded ones
  (`llm-rubric`, `g-eval`, `select-best`).
- Readable outputs at every tier: `promptfoo view` side-by-side web matrix,
  JSON/CSV/HTML export, and an official GitHub Action that posts before/after
  comparisons as PR comments.
- Native support for all three target provider families: OpenAI-compatible
  `apiBaseUrl` (the fabric), `anthropic:`, `openrouter:`.
- MIT-licensed, ~22k stars; acquired by OpenAI (March 2026) with a public
  commitment to keep the OSS suite maintained — OpenAI migrates its own
  deprecated Evals users to it. Growing, popular, zero infrastructure.

### Repo layout decisions

- `llm-prompt-evals` consumes `ai-llm-prompts` as a **git submodule pinned to a
  release commit** (Renovate-bumped). Prompts are never copied; a small
  promptfoo Python prompt function strips OKF frontmatter at load time.
  Candidate wording variants under test live in `llm-prompt-evals/variants/`
  and only graduate into the catalog once adopted.
- The catalog repo stays pure (its governance keeps consumer tooling,
  secrets, and schedules out); `mlx-benchmarks` stays a model-benchmark repo.

### promptstack is superseded

The bespoke `promptstack` suite ([promptstack.md](promptstack.md)) solved this
problem before a dedicated tool was chosen. Under the framework decision it is
superseded: its four probe banks (reasoning, tool_call incl. the negative
bank, instruction, homelab_qa) migrate into `llm-prompt-evals` test datasets,
and its adoption discipline (deterministic checks first, replicated runs,
no "best/worst" verdict language — see [verdict-policy.md](verdict-policy.md))
carries over as framework rules. Published promptstack shards remain valid
history in the dataset. The harness code stays until `llm-prompt-evals` covers
all four probe classes, then gets retired in a follow-up.

### Publishing Promptfoo JSON

`mlx-bench-publish --kind promptfoo --suite framework-eval` converts
`promptfoo eval -o results.json` output into one score row per test, one
pass/score row per assertion, and an arithmetic mean score row for the suite.
Use `--tag domain=<value>` for a declared domain from the envelope schema. The
converter records indexes, scores, and aggregate token statistics while keeping
prompts, variables, configuration, assertion reasons, and model outputs out of
the published envelope. It derives output token mean, median, p90, reasoning
token mean, tokens per fully successful case, and truncation rate from
per-result usage.

### Publishing Inspect logs

Use the optional `inspect` dependency when reading Zstandard-compressed
`.eval` files:

~~~sh
uv run --extra inspect mlx-bench-publish run.eval --kind inspect-ai --suite reasoning
~~~

## CI and security model

- Cloud-provider matrix (Anthropic + OpenRouter) runs on GitHub-hosted
  runners for same-repo PRs via `promptfoo/promptfoo-action` with caching.
- Fabric matrix runs only on a homelab self-hosted runner (to be deployed via
  the existing `github_runner` role in `ansible-proxmox-apps` — a runner
  inside the homelab reaches the fabric natively, unlike the org's AWS-hosted
  runs-on runners). Fabric jobs trigger only on `push` to main,
  `workflow_dispatch`, or same-repo labeled PRs — **never on fork PRs**.
  Fabric endpoints, hostnames, IPs, and API keys are read from environment
  values and never committed.

## Evaluator lanes (complements, not the core)

- **Galileo** (Cisco intent-to-acquire, April 2026): committed lane for
  observability/evaluation of the cloud-API-facing apps on its free tier
  (5k traces/mo, unlimited custom evals, Luna-2 evaluators). Not the core
  because its SaaS console cannot reach homelab endpoints.
- **Langfuse** (MIT, ClickHouse-backed): self-hosted in the homelab for
  durable prompt-experiment history and runtime tracing of deployed agents.
  Deployed as a new `ansible-proxmox-apps` docker-compose role (web, worker,
  postgres, clickhouse, redis, minio) following the existing role patterns.

## Phased roadmap (each phase in its owning repo)

1. **P1 — `dryvist/llm-prompt-evals`**: scaffold the promptfoo framework,
   first `hermes` eval (canonical vs. candidate variants), wiki pages in
   `dryvist/docs` and `dryvist/docs-starlight`.
2. **P2 — `ansible-proxmox-apps`**: `langfuse_docker` role.
3. **P3 — `ansible-proxmox-apps`**: homelab GitHub runner for
   `llm-prompt-evals` fabric jobs (security model above).
4. **P4 — API apps**: wire Galileo free tier.
5. **Completed — this repo**: retired the custom `promptstack` runner while
   retaining `--kind promptstack` for historical shards.
