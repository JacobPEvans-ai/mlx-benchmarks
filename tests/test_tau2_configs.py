"""configs/tau2/tau2.toml: upstream tau2 run shape, one fixed user simulator, no literal hosts."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RUNBOOK = REPO / "configs" / "tau2" / "tau2.toml"


def _text() -> str:
    return RUNBOOK.read_text()


def _runbook() -> dict[str, Any]:
    return tomllib.loads(_text())


def test_runbook_covers_the_four_domains() -> None:
    assert _runbook()["matrix"]["domains"] == ["airline", "retail", "telecom", "banking_knowledge"]


def test_run_command_is_the_upstream_tau2_run_shape() -> None:
    runbook = _runbook()
    for command in (runbook["run"]["command"], runbook["run"]["smoke"]["command"]):
        assert command.lstrip().startswith("tau2 run")
        for fragment in (
            "--domain $DOMAIN",
            "--agent-llm openai/$MODEL",
            "--agent-llm-args",
            "--user-llm $USER_LLM",
            "--user-llm-args",
            "--save-to $RUN_NAME",
        ):
            assert fragment in command


def test_user_simulator_is_one_fixed_field_shared_by_every_arm() -> None:
    runbook = _runbook()
    assert list(runbook["user"]) == ["llm"]
    assert runbook["user"]["llm"].startswith("openai/")
    # Every run takes the user model from [user]; no per-domain or per-arm override exists.
    assert "--user-llm $USER_LLM " in runbook["run"]["command"]
    assert "--user-llm $USER_LLM " in runbook["run"]["smoke"]["command"]


def test_num_trials_is_above_one_so_pass_hat_k_is_defined() -> None:
    assert _runbook()["matrix"]["num_trials"] > 1


def test_publish_uses_the_tau2_converter() -> None:
    command = _runbook()["publish"]["command"]
    assert "--kind tau2" in command
    assert "mlx-bench-publish" in command


def test_runbook_holds_placeholders_not_hosts() -> None:
    text = _text()
    assert re.search(r"https?://", text) is None
    assert re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text) is None
    assert "$BASE_URL" in text
    assert "$MODEL" in text
