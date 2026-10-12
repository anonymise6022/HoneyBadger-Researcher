"""Debate rendering, the kid depth, the extended validator, and CLI wiring."""

from __future__ import annotations

import pathlib
from datetime import date

import pytest
from typer.testing import CliRunner

from research_cli import cli
from research_cli.debate import run_debate
from research_cli.debate.demo import scripted
from research_cli.debate.models import Claim
from research_cli.debate.runner import run_debate_flow
from research_cli.evidence.outlook import build_outlook_bundle
from research_cli.explain import DEPTH_LABELS, Lexicon, resolve_depth
from research_cli.query_parser import parse_query
from research_cli.synthesis.debate_report import build_debate_report, render_debate_report
from research_cli.synthesis.llm_budget import BudgetedProvider, CallBudget
from research_cli.synthesis.llm_provider import MockProvider
from research_cli.synthesis.validator import extract_numbers, validate_report

TODAY = date(2026, 9, 21)
DEPTHS = ("kid", "beginner", "intermediate", "analyst")


@pytest.fixture(scope="module")
def bundle():
    return build_outlook_bundle(parse_query("is Tesla going to go up", today=TODAY), source="mock")


@pytest.fixture()
def result(bundle):
    provider = BudgetedProvider(MockProvider(scripted()), CallBudget(12), cache_dir=None)
    return run_debate(bundle, provider)


# --- depth ---------------------------------------------------------------------

def test_kid_depth_and_aliases():
    assert resolve_depth("simple") == "beginner"
    assert resolve_depth("full") == "analyst"
    assert resolve_depth("kid") == "kid"
    with pytest.raises(ValueError):
        resolve_depth("expert")
    assert "waves" in Lexicon("kid").gloss("volatility")
    assert Lexicon("kid").is_beginner


def test_desktop_depths_are_unchanged():
    assert list(DEPTH_LABELS) == ["beginner", "intermediate", "analyst"]


# --- validator -----------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "TSLA will rise from here.", "Gains are guaranteed.", "It is going to climb.",
    "This is bound to work.",
])
def test_certainty_is_flagged_only_when_asked(bundle, phrase):
    flagged = validate_report(phrase, bundle, required_sections=(), check_numbers=False,
                              forbid_certainty=True)
    assert any(i.kind == "certainty_language" for i in flagged.issues)
    relaxed = validate_report(phrase, bundle, required_sections=(), check_numbers=False)
    assert not any(i.kind == "certainty_language" for i in relaxed.issues)


def test_stance_phrases_are_allowed(bundle):
    text = "The bear case says the price fell. A value investor would read this as expensive."
    report = validate_report(text, bundle, required_sections=(), check_numbers=False,
                             forbid_certainty=True)
    assert not [i for i in report.issues if i.kind.endswith("_language")]


def test_only_planner_code_passes_extra_numbers():
    package = pathlib.Path(cli.__file__).resolve().parent
    callers = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if "tests" not in path.parts and "extra_numbers=" in path.read_text("utf-8")
    ]
    assert all(c.startswith("planner/") for c in callers), callers


# --- the report ----------------------------------------------------------------

@pytest.mark.parametrize("depth", DEPTHS)
def test_debate_report_passes_at_every_depth(bundle, result, depth):
    report = build_debate_report(bundle, result, depth)
    assert report.used_debate, report.reason
    assert report.validation.ok


def test_every_depth_carries_the_same_figures(bundle, result):
    def figures(depth):
        return sorted(v for v, _, _ in extract_numbers(render_debate_report(bundle, result, depth)))

    assert len({tuple(figures(d)) for d in DEPTHS}) == 1


def test_no_clear_winner_is_said_plainly(bundle, result):
    text = render_debate_report(bundle, result)
    if result.outcome == "several":
        assert "SEVERAL STRONG VIEWS" in text and "No single case won" in text
    assert "What would prove it wrong" in text
    assert "This is not financial advice." in text and "YOU DECIDE" in text


def test_invented_number_in_debate_prose_falls_back_to_template(bundle, result):
    leader = result.leaders[0]
    leader.claims.append(Claim("x.c9", "Profit will reach 987.6% next year.", ("V1",)))
    report = build_debate_report(bundle, result)
    assert not report.used_debate
    assert "final evidence check" in report.reason
    assert "OUTLOOK" in report.text and report.validation.ok


def test_verdict_in_debate_prose_falls_back(bundle, result):
    result.leaders[0].headline = "This is a strong buy."
    assert not build_debate_report(bundle, result).used_debate


def test_failed_debate_falls_back(bundle):
    provider = BudgetedProvider(MockProvider("not json"), CallBudget(12), cache_dir=None)
    report = build_debate_report(bundle, run_debate(bundle, provider))
    assert not report.used_debate and report.validation.ok


def test_unavailable_provider_falls_back_without_calls(bundle, monkeypatch):
    monkeypatch.setenv("RESEARCH_CLI_CLAUDE_BIN", "/nonexistent/claude")
    run = run_debate_flow(bundle, provider_name="claude-cli")
    assert not run.report.used_debate and "no model" in run.report.reason
    assert run.usage == "LLM calls used: 0 of 12"


# --- CLI -----------------------------------------------------------------------

def test_cli_debate_with_demo_agents():
    outcome = CliRunner().invoke(cli.app, [
        "ask", "is Tesla going to go up", "--debate", "--mock", "--provider", "mock",
        "--plain", "--as-of", "2026-09-21", "--depth", "kid",
    ])
    assert outcome.exit_code == 0, outcome.output
    assert "THE DEBATE -- TSLA" in outcome.output
    assert "LLM calls used: 10 of 12" in outcome.output
    assert "demo agents" in outcome.output


def test_cli_without_debate_makes_no_model_calls(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("no provider may be resolved without --debate")

    monkeypatch.setattr("research_cli.debate.runner.resolve_provider", boom)
    outcome = CliRunner().invoke(cli.app, [
        "ask", "is Tesla going to go up", "--mock", "--plain", "--as-of", "2026-09-21",
    ])
    assert outcome.exit_code == 0, outcome.output
    assert "WHAT IF: SCENARIOS FROM HISTORY" in outcome.output
    assert "LLM calls used" not in outcome.output
