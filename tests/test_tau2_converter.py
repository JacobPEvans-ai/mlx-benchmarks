"""tau2-bench results.json -> envelope v1, against a hand-made fixture.

The fixture has two tasks, two trials each, and one infrastructure error:

- task_a: trial 0 reward 1.0, trial 1 reward 0.0
- task_b: trial 0 reward 1.0, trial 1 infrastructure_error (reward_info null)

Evaluated simulations: 3, rewards 1, 0, 1. So pass_1_sim_mean = 2/3. Upstream's
task-level pass^1 is (1/2 + 1/1) / 2 = 3/4, reported as pass_hat_k with k=1. The
infrastructure error leaves task_b with one trial, so upstream caps k at 1: no
pass_hat_k row for k=2.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from mlx_benchmarks.cli import build_parser, main
from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.converters.tau2 import _NAIVE_TIMESTAMP_NOTE, Tau2Converter
from mlx_benchmarks.envelope import Envelope, validate_envelope


def _ctx(**overrides: object) -> ConverterContext:
    defaults: dict = {
        "suite": "agentic",
        "model": "mlx-community/example-served-model",
        "git_sha": "deadbeef",
        "system": {"os": "macOS", "chip": "Apple M4", "memory_gb": 64},
        "timestamp_override": "2026-10-20T09:00:00Z",
    }
    defaults.update(overrides)
    return ConverterContext(**defaults)


def _results(envelope: Envelope, metric: str) -> list[dict[str, Any]]:
    return [result for result in envelope["results"] if result["metric"] == metric]


def _one(envelope: Envelope, metric: str, **tags: str) -> dict[str, Any]:
    matches = [
        result
        for result in _results(envelope, metric)
        if all(result["tags"].get(key) == value for key, value in tags.items())
    ]
    assert len(matches) == 1, f"expected one {metric} row for {tags}, found {len(matches)}"
    return matches[0]


def _sim(task_id: str, trial: int, reward: float | None, termination: str = "user_stop") -> dict[str, Any]:
    reward_info = None if reward is None else {"reward": reward}
    return {
        "id": f"sim-{task_id}-{trial}",
        "task_id": task_id,
        "trial": trial,
        "termination_reason": termination,
        "reward_info": reward_info,
    }


@contextmanager
def _local_zone(name: str) -> Iterator[None]:
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def test_fixture_round_trip_validates(tau2_results_sample: dict[str, Any]) -> None:
    envelope = get_converter("tau2").build_envelope(tau2_results_sample, _ctx())
    validate_envelope(envelope)

    assert envelope["schema_version"] == "1"
    assert envelope["suite"] == "agentic"
    assert envelope["model"] == "mlx-community/example-served-model"
    assert envelope["git_sha"] == "deadbeef"
    assert envelope["timestamp"] == "2026-10-20T09:00:00Z"
    assert envelope["errors"] == [
        "pass_hat_k not computed for k=2 through 2: the fewest evaluated trials for any task is 1 "
        "(infrastructure errors excluded)"
    ]


def test_sim_mean_and_upstream_task_mean_are_reported_separately(
    tau2_results_sample: dict[str, Any],
) -> None:
    envelope = Tau2Converter().build_envelope(tau2_results_sample, _ctx())

    sim_mean = _one(envelope, "pass_1_sim_mean")
    assert sim_mean["value"] == pytest.approx(2 / 3)
    assert sim_mean["unit"] == "ratio"
    assert sim_mean["name"] == "tau2_bench"
    # Upstream pass_hat_ks[1]: task_a 1/2, task_b 1/1, mean over tasks.
    task_mean = _one(envelope, "pass_hat_k", k="1")
    assert task_mean["value"] == pytest.approx(0.75)
    assert task_mean["name"] == "tau2_bench"


def test_infrastructure_errors_are_counted_and_excluded(tau2_results_sample: dict[str, Any]) -> None:
    envelope = Tau2Converter().build_envelope(tau2_results_sample, _ctx())

    evaluated = _one(envelope, "evaluated_simulations")
    infra = _one(envelope, "infrastructure_errors")
    assert evaluated["value"] == 3.0
    assert infra["value"] == 1.0
    assert evaluated["unit"] == infra["unit"] == "count"


def test_pass_hat_k_stops_at_one_when_a_task_lacks_a_second_trial(
    tau2_results_sample: dict[str, Any],
) -> None:
    envelope = Tau2Converter().build_envelope(tau2_results_sample, _ctx())

    assert [result["tags"]["k"] for result in _results(envelope, "pass_hat_k")] == ["1"]
    assert len(envelope["errors"]) == 1
    assert "k=2 through 2" in envelope["errors"][0]


def test_pass_hat_k_for_two_complete_trials(tau2_results_sample: dict[str, Any]) -> None:
    # Give task_b its missing trial a success: no infrastructure error left.
    raw = copy.deepcopy(tau2_results_sample)
    raw["simulations"][3] = _sim("task_b", 1, 1.0)

    envelope = Tau2Converter().build_envelope(raw, _ctx())
    validate_envelope(envelope)

    # Rewards: task_a [1, 0], task_b [1, 1].
    # pass_1_sim_mean = 3/4. pass^1: task_a 1/2, task_b 2/2, mean 3/4.
    # pass^2: task_a C(1,2)/C(2,2) = 0, task_b C(2,2)/C(2,2) = 1, mean 0.5.
    assert _one(envelope, "pass_1_sim_mean")["value"] == pytest.approx(0.75)
    assert _one(envelope, "pass_hat_k", k="1")["value"] == pytest.approx(0.75)
    assert _one(envelope, "pass_hat_k", k="2")["value"] == pytest.approx(0.5)
    assert envelope["errors"] == []
    assert _one(envelope, "infrastructure_errors")["value"] == 0.0


def test_pass_hat_k_matches_upstream_formula_for_three_trials(tau2_results_sample: dict[str, Any]) -> None:
    raw = copy.deepcopy(tau2_results_sample)
    raw["info"]["num_trials"] = 3
    raw["simulations"] = [
        _sim("task_a", 0, 1.0),
        _sim("task_a", 1, 1.0),
        _sim("task_a", 2, 1.0),
        _sim("task_b", 0, 1.0),
        _sim("task_b", 1, 1.0),
        _sim("task_b", 2, 0.0),
    ]

    envelope = Tau2Converter().build_envelope(raw, _ctx())
    validate_envelope(envelope)

    # pass^1: task_a 1, task_b 2/3, mean 5/6 (equal trial counts, so it matches the sim mean).
    # pass^2: task_a C(3,2)/C(3,2) = 1, task_b C(2,2)/C(3,2) = 1/3, mean 2/3.
    # pass^3: task_a C(3,3)/C(3,3) = 1, task_b C(2,3)/C(3,3) = 0, mean 1/2.
    assert _one(envelope, "pass_hat_k", k="1")["value"] == pytest.approx(5 / 6)
    assert _one(envelope, "pass_hat_k", k="2")["value"] == pytest.approx(2 / 3)
    assert _one(envelope, "pass_hat_k", k="3")["value"] == pytest.approx(0.5)
    assert _one(envelope, "pass_1_sim_mean")["value"] == pytest.approx(5 / 6)
    assert envelope["errors"] == []


def test_single_trial_run_reports_only_k_one(tau2_results_sample: dict[str, Any]) -> None:
    raw = copy.deepcopy(tau2_results_sample)
    raw["info"]["num_trials"] = 1
    raw["simulations"] = [_sim("task_a", 0, 1.0), _sim("task_b", 0, 0.0)]

    envelope = Tau2Converter().build_envelope(raw, _ctx())

    assert [result["tags"]["k"] for result in _results(envelope, "pass_hat_k")] == ["1"]
    assert envelope["errors"] == []
    assert _one(envelope, "pass_1_sim_mean")["value"] == pytest.approx(0.5)
    assert _one(envelope, "pass_hat_k", k="1")["value"] == pytest.approx(0.5)


def test_run_tags_name_the_domain_and_models(tau2_results_sample: dict[str, Any]) -> None:
    envelope = Tau2Converter().build_envelope(tau2_results_sample, _ctx())

    for result in envelope["results"]:
        tags = result["tags"]
        assert tags["tau2_domain"] == "airline"
        assert tags["num_trials"] == "2"
        assert tags["n_tasks"] == "2"
        assert tags["agent_llm"] == "openai/example-served-model"
        assert tags["user_llm"] == "openai/example-user-sim"
        assert tags["tau2_git_commit"] == "0123456789abcdef0123456789abcdef01234567"


def test_llm_args_keep_settings_and_drop_endpoint_and_key(tau2_results_sample: dict[str, Any]) -> None:
    envelope = Tau2Converter().build_envelope(tau2_results_sample, _ctx())

    tags = _one(envelope, "pass_1_sim_mean")["tags"]
    assert json.loads(tags["agent_llm_args"]) == {"temperature": 0.0}
    assert json.loads(tags["user_llm_args"]) == {"temperature": 0.0}

    published = json.dumps(envelope)
    assert "api_key" not in published
    assert "api_base" not in published
    assert "llm.example.invalid" not in published


def test_absent_llm_args_are_marked_unstated(tau2_results_sample: dict[str, Any]) -> None:
    raw = copy.deepcopy(tau2_results_sample)
    del raw["info"]["agent_info"]["llm_args"]
    raw["info"]["user_info"]["llm"] = None
    del raw["info"]["git_commit"]

    tags = _one(Tau2Converter().build_envelope(raw, _ctx()), "pass_1_sim_mean")["tags"]

    assert tags["agent_llm_args"] == "unstated"
    assert tags["user_llm"] == "unstated"
    assert "tau2_git_commit" not in tags


def test_naive_timestamp_is_read_as_local_time_and_recorded_in_errors(
    tau2_results_sample: dict[str, Any],
) -> None:
    # tau2 writes naive local time (datetime.now()). With TZ set to UTC-4 in the
    # publishing host, 09:00 local is 13:00 UTC.
    with _local_zone("America/New_York"):
        envelope = Tau2Converter().build_envelope(tau2_results_sample, _ctx(timestamp_override=None))

    assert envelope["timestamp"] == "2026-10-20T13:00:00Z"
    assert envelope["errors"][0] == _NAIVE_TIMESTAMP_NOTE
    assert "no UTC offset" in _NAIVE_TIMESTAMP_NOTE


def test_timestamp_with_offset_is_converted_to_utc_without_a_note(
    tau2_results_sample: dict[str, Any],
) -> None:
    raw = copy.deepcopy(tau2_results_sample)
    raw["timestamp"] = "2026-10-20T09:00:00-04:00"

    envelope = Tau2Converter().build_envelope(raw, _ctx(timestamp_override=None))

    assert envelope["timestamp"] == "2026-10-20T13:00:00Z"
    assert _NAIVE_TIMESTAMP_NOTE not in envelope["errors"]


def test_timestamp_override_wins_without_a_note(tau2_results_sample: dict[str, Any]) -> None:
    envelope = Tau2Converter().build_envelope(
        tau2_results_sample, _ctx(timestamp_override="2026-01-02T03:04:05Z")
    )

    assert envelope["timestamp"] == "2026-01-02T03:04:05Z"
    assert _NAIVE_TIMESTAMP_NOTE not in envelope["errors"]


def test_all_infrastructure_errors_is_an_error(tau2_results_sample: dict[str, Any]) -> None:
    raw = copy.deepcopy(tau2_results_sample)
    for sim in raw["simulations"]:
        sim["termination_reason"] = "infrastructure_error"
        sim["reward_info"] = None

    with pytest.raises(ValueError, match="nothing to publish"):
        Tau2Converter().build_envelope(raw, _ctx())


def test_unscored_simulation_that_is_not_infrastructure_is_refused(
    tau2_results_sample: dict[str, Any],
) -> None:
    raw = copy.deepcopy(tau2_results_sample)
    raw["simulations"][0] = _sim("task_a", 0, None, termination="unexpected_error")

    with pytest.raises(ValueError, match="has no reward and is not an infrastructure error"):
        Tau2Converter().build_envelope(raw, _ctx())


def _drop(*path: str) -> Any:
    def mutate(raw: dict[str, Any]) -> None:
        node = raw
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop("simulations"), "no simulations list"),
        (_drop("info", "num_trials"), "num_trials must be a positive integer"),
        (_drop("info", "environment_info", "domain_name"), "domain_name is missing"),
        (_drop("timestamp"), "has no timestamp"),
        (_drop("info", "agent_info"), "info.agent_info must be an object"),
    ],
)
def test_malformed_results_are_refused(
    tau2_results_sample: dict[str, Any], mutate: Any, message: str
) -> None:
    raw = copy.deepcopy(tau2_results_sample)
    mutate(raw)

    with pytest.raises(ValueError, match=message):
        Tau2Converter().build_envelope(raw, _ctx(timestamp_override=None))


def test_zero_trials_is_refused(tau2_results_sample: dict[str, Any]) -> None:
    raw = copy.deepcopy(tau2_results_sample)
    raw["info"]["num_trials"] = 0

    with pytest.raises(ValueError, match="num_trials must be a positive integer"):
        Tau2Converter().build_envelope(raw, _ctx())


def test_non_object_input_is_refused() -> None:
    not_an_object: Any = []
    with pytest.raises(ValueError, match=r"results\.json object"):
        Tau2Converter().build_envelope(not_an_object, _ctx())


def test_cli_accepts_tau2_kind_and_writes_a_valid_envelope(
    tmp_path: Path,
    tau2_results_sample: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mlx_benchmarks.cli.detect_system",
        lambda: {"os": "macOS", "chip": "Apple M4", "memory_gb": 64},
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(tau2_results_sample))
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            str(results_path),
            "--kind",
            "tau2",
            "--suite",
            "agentic",
            "--model",
            "mlx-community/example-served-model",
            "--git-sha",
            "deadbeef",
            "--timestamp",
            "2026-10-20T09:00:00Z",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    (written,) = output_dir.glob("run-*.json")
    envelope = json.loads(written.read_text())
    validate_envelope(envelope)
    assert envelope["suite"] == "agentic"
    assert envelope["model"] == "mlx-community/example-served-model"


def test_tau2_is_a_registered_cli_kind() -> None:
    kind_action = next(action for action in build_parser()._actions if action.dest == "kind")
    assert "tau2" in (kind_action.choices or ())
    assert isinstance(get_converter("tau2"), Tau2Converter)
