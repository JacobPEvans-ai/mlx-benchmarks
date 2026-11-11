"""tau2-bench results.json (``tau2 run --save-to``) -> envelope v1.

One results.json holds one domain: a text run takes exactly one domain, so one
file becomes one envelope. Rewards come from the upstream evaluator and are not
re-scored here. The aggregation follows ``tau2.metrics.agent_metrics``
(sierra-research/tau2-bench v2.0.0):

- ``pass_1_sim_mean``: mean reward over the evaluated simulations.
- ``pass_hat_k`` (tag ``k``) for each k from 1 up to the largest k that every
  task can supply: mean over tasks of C(successes, k) / C(trials, k). The k=1 row
  is upstream's ``pass_hat_ks[1]``, a task-level mean. It differs from
  ``pass_1_sim_mean`` whenever tasks have unequal evaluated trial counts.
- ``evaluated_simulations`` and ``infrastructure_errors`` counts. Infrastructure
  errors are excluded from every score, as upstream excludes them.

tau2 records a naive local timestamp with no zone; see ``_timestamp``. The
assumption is recorded in ``errors``.
"""

from __future__ import annotations

import datetime
import json
import math
from collections import defaultdict
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields
from mlx_benchmarks.envelope import Envelope, Result, System

RESULT_NAME = "tau2_bench"
# tau2.metrics.agent_metrics.is_successful: a reward within 1e-6 of 1.
_SUCCESS_TOLERANCE = 1e-6
_INFRASTRUCTURE_ERROR = "infrastructure_error"
# Endpoint and credential keys an operator passes in --agent-llm-args or
# --user-llm-args. They are never published.
_WITHHELD_LLM_ARG_KEYS = frozenset({"api_key", "api_base"})
_NAIVE_TIMESTAMP_NOTE = (
    "tau2 recorded a timestamp with no UTC offset; it was read as the publishing host's local "
    "time and stored as UTC. Pass --timestamp to state the run time explicitly."
)


class Tau2Converter:
    kind = "tau2"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        if not isinstance(raw, dict):
            raise ValueError("tau2 results must be a results.json object")
        simulations = raw.get("simulations")
        if not isinstance(simulations, list):
            raise ValueError("tau2 results.json has no simulations list")
        info = _mapping(raw.get("info"), "info")
        num_trials = info.get("num_trials")
        if not isinstance(num_trials, int) or isinstance(num_trials, bool) or num_trials < 1:
            raise ValueError("tau2 info.num_trials must be a positive integer")
        domain = _mapping(info.get("environment_info"), "info.environment_info").get("domain_name")
        if not isinstance(domain, str) or not domain:
            raise ValueError("tau2 info.environment_info.domain_name is missing")

        infrastructure = [sim for sim in simulations if _termination(sim) == _INFRASTRUCTURE_ERROR]
        evaluated = [sim for sim in simulations if _termination(sim) != _INFRASTRUCTURE_ERROR]
        if not evaluated:
            raise ValueError("tau2 results contain no evaluated simulations — nothing to publish")

        rewards_by_task: dict[str, list[float]] = defaultdict(list)
        for sim in evaluated:
            rewards_by_task[_task_id(sim)].append(_reward(sim))
        all_rewards = [reward for rewards in rewards_by_task.values() for reward in rewards]
        min_trials = min(len(rewards) for rewards in rewards_by_task.values())
        # Upstream caps k at the fewest evaluated trials of any task.
        max_k = min(num_trials, min_trials)
        tags = _run_tags(domain, num_trials, len(rewards_by_task), info)

        results: list[Result] = [
            {
                "name": RESULT_NAME,
                "metric": "pass_1_sim_mean",
                "value": sum(all_rewards) / len(all_rewards),
                "unit": "ratio",
                "tags": dict(tags),
            }
        ]
        for k in range(1, max_k + 1):
            results.append(
                {
                    "name": RESULT_NAME,
                    "metric": "pass_hat_k",
                    "value": _pass_hat_k(rewards_by_task, k),
                    "unit": "ratio",
                    "tags": {**tags, "k": str(k)},
                }
            )
        results.append(
            {
                "name": RESULT_NAME,
                "metric": "evaluated_simulations",
                "value": float(len(evaluated)),
                "unit": "count",
                "tags": dict(tags),
            }
        )
        results.append(
            {
                "name": RESULT_NAME,
                "metric": "infrastructure_errors",
                "value": float(len(infrastructure)),
                "unit": "count",
                "tags": dict(tags),
            }
        )

        errors: list[str] = []
        if ctx.timestamp_override:
            timestamp = ctx.timestamp_override
        else:
            timestamp, naive = _timestamp(raw)
            if naive:
                errors.append(_NAIVE_TIMESTAMP_NOTE)
        if max_k < num_trials:
            errors.append(
                f"pass_hat_k not computed for k={max_k + 1} through {num_trials}: the fewest "
                f"evaluated trials for any task is {min_trials} (infrastructure errors excluded)"
            )

        system: System = ctx.system or {}  # type: ignore[assignment]
        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": timestamp,
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": results,
            "errors": errors,
        }
        return apply_optional_fields(envelope, ctx)


def _pass_hat_k(rewards_by_task: dict[str, list[float]], k: int) -> float:
    """tau2's pass^k: per task C(successes, k) / C(trials, k), then the mean over tasks."""
    per_task = []
    for rewards in rewards_by_task.values():
        successes = sum(1 for reward in rewards if _is_success(reward))
        per_task.append(math.comb(successes, k) / math.comb(len(rewards), k))
    return sum(per_task) / len(per_task)


def _is_success(reward: float) -> bool:
    return (1 - _SUCCESS_TOLERANCE) <= reward <= (1 + _SUCCESS_TOLERANCE)


def _run_tags(domain: str, num_trials: int, n_tasks: int, info: dict[str, Any]) -> dict[str, str]:
    agent = _mapping(info.get("agent_info"), "info.agent_info")
    user = _mapping(info.get("user_info"), "info.user_info")
    tags = {
        "tau2_domain": domain,
        "num_trials": str(num_trials),
        "n_tasks": str(n_tasks),
        "agent_llm": _llm_name(agent),
        "agent_llm_args": _llm_args(agent),
        "user_llm": _llm_name(user),
        "user_llm_args": _llm_args(user),
    }
    git_commit = info.get("git_commit")
    if isinstance(git_commit, str) and git_commit:
        tags["tau2_git_commit"] = git_commit
    return tags


def _llm_name(block: dict[str, Any]) -> str:
    llm = block.get("llm")
    return llm if isinstance(llm, str) and llm else "unstated"


def _llm_args(block: dict[str, Any]) -> str:
    args = block.get("llm_args")
    if args is None:
        return "unstated"
    if not isinstance(args, dict):
        raise ValueError("tau2 llm_args must be an object")
    public = {key: value for key, value in args.items() if key not in _WITHHELD_LLM_ARG_KEYS}
    return json.dumps(public, sort_keys=True)


def _timestamp(raw: dict[str, Any]) -> tuple[str, bool]:
    """Return the run start as UTC ``Z`` time, and whether tau2 wrote it with no zone.

    tau2 writes ``datetime.now()`` with no zone. A naive value is read as the local
    clock of the host running this converter, which is the publishing host in this
    repo's runbooks. The caller records that assumption in ``errors``. Pass
    ``--timestamp`` to state the time explicitly instead.
    """
    value = raw.get("timestamp")
    if not isinstance(value, str):
        raise ValueError("tau2 results.json has no timestamp")
    parsed = datetime.datetime.fromisoformat(value)
    naive = parsed.tzinfo is None
    if naive:
        parsed = parsed.astimezone()
    return parsed.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), naive


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"tau2 {label} must be an object")
    return value


def _termination(sim: Any) -> str:
    value = sim.get("termination_reason") if isinstance(sim, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError("tau2 simulation has no termination_reason")
    return value


def _task_id(sim: dict[str, Any]) -> str:
    value = sim.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"tau2 simulation {sim.get('id')!r} has no task_id")
    return value


def _reward(sim: dict[str, Any]) -> float:
    reward_info = sim.get("reward_info")
    reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
    if isinstance(reward, bool) or not isinstance(reward, int | float):
        raise ValueError(
            f"tau2 simulation {sim.get('id')!r} has no reward and is not an infrastructure error "
            f"(termination_reason={_termination(sim)!r})"
        )
    return float(reward)
