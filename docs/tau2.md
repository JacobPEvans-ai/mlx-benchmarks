# tau2-bench

[tau2-bench](https://github.com/sierra-research/tau2-bench) (v2.0.0) scores an
agent on customer-service tasks: airline, retail, telecom and banking knowledge. A
user simulator talks to the agent, and the agent must change the environment
correctly under a written policy. Upstream `tau2 run` does the simulation and the
scoring. This repo converts the `results.json` it writes into envelope v1 rows.
Nothing here re-runs or re-scores an episode.

The runbook is [`configs/tau2/tau2.toml`](../configs/tau2/tau2.toml). The
converter is [`converters/tau2.py`](../src/mlx_benchmarks/converters/tau2.py), with
`--kind tau2`.

## What gets published

One `results.json` covers one domain. A text run takes exactly one domain, so one
file becomes one envelope. Every row has `name = tau2_bench` and carries the
run's `tau2_domain`, `num_trials`, `n_tasks`, and the agent and user models. The
envelope's `model` comes from `--model`.

| Metric | Unit | Meaning |
| --- | --- | --- |
| `pass_1_sim_mean` | ratio | Mean reward over the evaluated simulations. |
| `pass_hat_k` | ratio | pass^k, tag `k`, from 1 to the largest k every task supports. `k = 1` is upstream's task-level `pass_hat_ks[1]`. |
| `evaluated_simulations` | count | Simulations scored. |
| `infrastructure_errors` | count | Simulations with `termination_reason = infrastructure_error`. Excluded from every score. |

pass^k is upstream's definition. Per task, C(successes, k) / C(trials, k). The
score is the mean over tasks, and `tau2.metrics.agent_metrics.pass_hat_k` is
the reference. A success is a reward within 1e-6 of 1.

Two things to know before reading the numbers:

- `pass_1_sim_mean` and `pass_hat_k` with `k = 1` are both published. The first is the
  simulation-level mean. The second is upstream's `pass_hat_ks[1]`, a task-level mean.
  The two are equal only when every task has the same number of evaluated trials. An
  infrastructure error that drops one trial makes them differ.
- When an infrastructure error leaves some task with fewer trials than
  `num_trials`, upstream caps k at the smallest task count. The envelope then has no
  `pass_hat_k` row above that cap, and `errors[]` says so.

Each row also records the agent and user `llm_args` with `api_key` and `api_base`
removed, and the tau2 checkout's `info.git_commit` as `tau2_git_commit`.

## Running upstream tau2 against an OpenAI-compatible endpoint

Use a checkout of tau2-bench at v2.0.0 and install it into its own environment.
Do not install it into this repo's `.venv`. This converter reads JSON only.

tau2 calls models through litellm. Name the served model with the `openai/` prefix,
and pass the endpoint root in `api_base`. Local OpenAI-compatible servers usually
accept any key, but litellm needs one set, so pass `"api_key": "none"`:

```sh
tau2 run \
  --domain airline \
  --agent-llm openai/<served-model-id> \
  --agent-llm-args '{"api_base": "<BASE_URL>", "api_key": "none", "temperature": 0.0}' \
  --user-llm openai/<user-sim-served-id> \
  --user-llm-args '{"api_base": "<BASE_URL>", "api_key": "none", "temperature": 0.0}' \
  --num-trials 4 \
  --save-to airline-<run-name>
```

`<BASE_URL>` is the endpoint root, including `/v1`. The served id is the one the
endpoint advertises. Check it against the live model list before a run.

Results land in `data/simulations/<run-name>/results.json` inside the tau2-bench
checkout. The runbook renders this same command for each domain, with
`--num-trials` set from `[matrix].num_trials`.

Banking knowledge is different. When no retrieval config is given, upstream uses
`alltools` (`DEFAULT_BANKING_RETRIEVAL_CONFIG` in `tau2/config.py`). The runbook
leaves that default in place. The results file records the choice in
`info.retrieval_config`, but this converter does not publish it.

## The user simulator is fixed

The user simulator is part of the measurement. It writes every user turn the agent
sees. If it changes between arms, the arms differ in their test, not only in the
model under test.

- Set `[user].llm` in `configs/tau2/tau2.toml` once. It is shared by every arm and
  every domain.
- Use a model that is not the model under test.
- Do not leave tau2's default user model in place. Its default is a hosted OpenAI
  model (`gpt-4.1`), not a model on your endpoint.
- Changing the user model invalidates every existing arm. Re-run all of them.

Each row records `user_llm` and `user_llm_args`, so a published table with mixed
user models is visible in the data. The rule is still to keep the user model
fixed.

Upstream can also score communicate checks with an LLM judge. Look at
`info.communicate_scoring` in the results file. If the scorer is `semantic`, the judge
model is recorded there but not in the envelope. Keep the judge fixed across arms
too.

## Publishing

Convert and dry-run first. The dry run reads the Hub for the model and needs
network access. It does not write.

```sh
.venv/bin/mlx-bench-publish <tau2-checkout>/data/simulations/<run-name>/results.json \
  --kind tau2 --suite agentic \
  --model <served-model-id> \
  --published-metadata run-output/published-metadata.json \
  --dry-run
```

Drop `--dry-run` to publish. `HF_TOKEN` must be the write token, as for every
publish in this repo. `--published-metadata` is required for every published score.
Its shape is the `published_result` definition in `schema.json`.

Suite: `agentic`. tau2 rows score whole agent episodes by reward. The `tool-calling`
suite records per-call tool validity rates, so it does not describe these rows. The
runbook and every example use `--suite agentic`.

Timestamp: tau2 writes naive local time with no zone. The converter reads it as the
publishing host's local time and stores it as UTC. Every envelope converted this way
records that assumption in `errors[]`. The envelope has no notes field, so `errors[]`
is its only free-text list. If the publishing host's timezone differs from the run
host's, pass `--timestamp` with an explicit UTC time.

## Traps

- **One domain per file.** `tau2 run` rejects several domains for a text run. Run
  each domain separately, and publish each `results.json` as its own envelope.
- **Infrastructure errors are not scores.** They are excluded and counted. If they are
  frequent, the arm is not comparable. Investigate them before publishing.
- **Unscored, non-infrastructure simulations stop the publish.** A simulation that is
  not an infrastructure error but has no reward (for example `unexpected_error`) raises
  `ValueError`. Upstream labels such runs `ERROR` or `UNSCORED`. The converter does
  not guess a score for them.
- **Missing pass^k is recorded, not hidden.** Read `errors[]` whenever `num_trials > 1`.
- **Credentials and endpoints never reach the dataset.** The `llm_args` tags keep only
  scalar sampling settings (temperature, top_p, top_k, max_tokens, max_completion_tokens,
  seed, reasoning_effort); every other key is dropped. Do not put a host name or URL into a tag,
  commit message, or doc.
