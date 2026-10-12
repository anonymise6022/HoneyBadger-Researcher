"""The desktop bridge for the Debate tab, on synthetic data with demo agents."""

from __future__ import annotations

import pathlib

import pytest

from research_cli.desktop import shell
from research_cli.desktop.debate_api import AppApi


@pytest.fixture(scope="module")
def api() -> AppApi:
    return AppApi(mock=True)


def test_window_uses_the_extended_bridge():
    assert "AppApi(mock=mock)" in pathlib.Path(shell.__file__).read_text("utf-8")


def test_future_questions_on_ask_are_handed_to_the_debate_tab(api):
    result = api.research("is Tesla going to go up")
    assert not result["ok"] and result["debate_query"] == "is Tesla going to go up"


def test_other_ask_questions_are_unchanged(api):
    assert api.research("why did SPY fall today")["ok"]


def test_offline_status_uses_demo_agents(api):
    status = api.debate_status()
    assert status["ok"] and status["mode"] == "demo"


@pytest.mark.parametrize("depth", ["kid", "beginner", "analyst"])
def test_debate_returns_the_validated_report(api, depth):
    result = api.debate("is Tesla going to go up", depth)
    assert result["ok"], result
    assert "THE DEBATE -- TSLA" in result["report"]
    assert result["validation_ok"] and result["outcome"] in ("clear", "several")
    assert result["synthetic"]
    assert result["usage"].startswith("LLM calls used:")
    assert any("demo agents" in note for note in result["notes"])


def test_debate_refuses_why_questions(api):
    result = api.debate("why did SPY fall today")
    assert not result["ok"] and "Ask tab" in result["suggestion"]


def test_plan_from_the_window(api):
    result = api.plan("TSLA", "$1000", "1 year", "$200")
    assert result["ok"], result
    assert "OPTION A" in result["report"] and result["validation"].startswith("PASS")
    assert not api.plan("TSLA", "lots", "1 year", "$200")["ok"]


def test_interface_loads_the_debate_tab():
    web = pathlib.Path(shell.__file__).resolve().parent / "web"
    html = (web / "index.html").read_text("utf-8")
    assert 'data-view="debate"' in html and 'id="view-debate"' in html
    assert '<script src="debate.js"></script>' in html
    script = (web / "debate.js").read_text("utf-8")
    assert "api().debate(" in script and "api().plan(" in script
