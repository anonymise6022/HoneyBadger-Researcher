"""Execution planner: inputs, sizing math, plan options, rendering, opt-in flow."""

from __future__ import annotations

import pathlib
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from research_cli import cli
from research_cli.backtest.engine import SealedWindow
from research_cli.evidence.outlook import build_outlook_bundle
from research_cli.planner.inputs import (
    PlanInputError,
    parse_amount,
    parse_horizon,
    parse_max_loss,
)
from research_cli.planner.interactive import build_plan_report, offer_plan
from research_cli.planner.plan import _StagedEntry
from research_cli.planner.sizing import horizon_risk, size_for_loss, stop_outcomes
from research_cli.query_parser import parse_query

TODAY = date(2026, 9, 21)


@pytest.fixture(scope="module")
def tsla_plan():
    return build_plan_report("TSLA", "TSLA", 1000.0, 252, "1 year", 200.0, "mock", as_of=TODAY)


# --- inputs ------------------------------------------------------------------

@pytest.mark.parametrize("text, value", [("$1,000", 1000.0), ("2.5k", 2500.0), ("750", 750.0)])
def test_amounts(text, value):
    assert parse_amount(text) == value


@pytest.mark.parametrize("text, days", [("6 months", 126), ("1 year", 252), ("3y", 756),
                                        ("90 days", 90)])
def test_horizons(text, days):
    assert parse_horizon(text)[0] == days


def test_loss_limit_in_dollars_or_percent_and_capped():
    assert parse_max_loss("20%", 1000) == 200
    assert parse_max_loss("$25", 1000) == 25
    assert parse_max_loss("$5000", 1000) == 1000


@pytest.mark.parametrize("call", [
    lambda: parse_amount("lots"), lambda: parse_amount("$10"),
    lambda: parse_horizon("a week"), lambda: parse_horizon("2 days"),
    lambda: parse_horizon("9 years"), lambda: parse_max_loss("0", 1000),
])
def test_bad_answers_are_refused(call):
    with pytest.raises(PlanInputError):
        call()


# --- sizing math ---------------------------------------------------------------

def test_max_loss_sizing():
    assert size_for_loss(200, 1000, 40.0) == pytest.approx(500.0)
    assert size_for_loss(200, 1000, 10.0) == 1000.0          # never more than the amount
    assert size_for_loss(200, 1000, 0.0) == 1000.0
    with pytest.raises(ValueError):
        size_for_loss(0, 1000, 40.0)


def test_horizon_risk_on_a_known_path():
    closes = pd.Series(100.0 * 1.001 ** np.arange(1200), index=pd.bdate_range("2015-01-01", periods=1200))
    risk = horizon_risk(closes, 21)
    assert risk.median_pct == pytest.approx((1.001 ** 21 - 1) * 100)
    assert risk.worst_pct == pytest.approx(risk.median_pct)
    assert risk.daily_vol_pct == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        horizon_risk(closes.head(300), 21)


def test_stop_replay_includes_gaps():
    # Flat, then a 40% overnight gap down: a 20% stop cannot save you.
    closes = pd.Series([100.0] * 5 + [60.0] * 5)
    result = stop_outcomes(closes, 6, 20.0)
    assert result.worst_pct == pytest.approx(-40.0)
    assert result.hit_share_pct > 0


def test_staged_entry_steps_up_and_fills_next_open():
    strategy = _StagedEntry(4, 2)
    frame = pd.DataFrame({"Close": np.arange(1.0, 11.0)}, index=pd.bdate_range("2020-01-01", periods=10))
    exposures = [strategy.signal(SealedWindow(frame, i)) for i in range(8)]
    assert exposures == [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]


# --- plan and report --------------------------------------------------------------

def test_plan_offers_three_labelled_options(tsla_plan):
    text, validation, plan = tsla_plan
    assert validation.ok, [str(i) for i in validation.issues]
    assert [o.key for o in plan.options] == ["A", "B", "C"]
    for heading in ("OPTION A", "OPTION B", "OPTION C", "PRACTICAL POINTS", "YOU DECIDE"):
        assert heading in text
    assert "Trade-off" in text


def test_option_a_respects_the_loss_budget(tsla_plan):
    _, _, plan = tsla_plan
    a = plan.options[0]
    assert a.invest == pytest.approx(min(1000.0, 200.0 / (plan.risk.bad_stretch_loss_pct / 100)), abs=0.01)
    assert a.invest + a.cash == pytest.approx(1000.0)


def test_plan_text_never_advises_or_promises(tsla_plan):
    text = tsla_plan[0].lower()
    for phrase in ("you should", "we recommend", "guarantee", "will rise", "best option"):
        assert phrase not in text
    assert "never places orders" in text


def test_no_exit_rule_when_the_whole_amount_may_be_lost():
    _, validation, plan = build_plan_report("TSLA", "TSLA", 1000.0, 126, "6 months", 1000.0,
                                            "mock", as_of=TODAY)
    assert validation.ok
    assert plan.options[1].stop is None and "no exit rule" in plan.options[1].title


def test_planner_has_no_network_or_broker_code():
    folder = pathlib.Path(cli.__file__).resolve().parent / "planner"
    for path in folder.glob("*.py"):
        imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", path.read_text("utf-8"), re.M)
        assert not [m for m in imports if m.split(".")[0] in ("requests", "urllib", "socket", "http")]


# --- opt-in flow --------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    return build_outlook_bundle(parse_query("is Tesla going to go up", today=TODAY), source="mock")


def _scripted(*answers):
    queue = list(answers)
    return lambda prompt: queue.pop(0)


def test_no_prompt_outside_a_terminal(bundle):
    def never(prompt):
        raise AssertionError("must not ask")

    assert offer_plan(bundle, input_fn=never, interactive=False) is None


def test_declining_ends_the_flow(bundle):
    shown = []
    assert offer_plan(bundle, input_fn=_scripted("n"), output=shown.append, interactive=True) is None
    assert any("decision is yours" in s for s in shown)


def test_yes_leads_to_three_questions_then_a_plan(bundle):
    shown = []
    text = offer_plan(bundle, source="mock", input_fn=_scripted("y", "$1000", "1 year", "20%"),
                      output=shown.append, interactive=True, as_of=TODAY)
    assert text is not None and "OPTION A" in text
    assert any("PASS" in s for s in shown)


def test_a_bad_answer_is_asked_again(bundle):
    shown = []
    text = offer_plan(bundle, source="mock",
                      input_fn=_scripted("yes", "lots", "$800", "6 months", "$100"),
                      output=shown.append, interactive=True, as_of=TODAY)
    assert text is not None
    assert any("not an amount" in s for s in shown)


def test_plan_command():
    outcome = CliRunner().invoke(cli.app, ["plan", "TSLA", "--amount", "$1000", "--horizon",
                                           "1 year", "--max-loss", "$200", "--mock",
                                           "--as-of", "2026-09-21"])
    assert outcome.exit_code == 0, outcome.output
    assert "A PLAN FOR YOUR DECISION -- TSLA" in outcome.output
    assert "[plan check] PASS" in outcome.output
