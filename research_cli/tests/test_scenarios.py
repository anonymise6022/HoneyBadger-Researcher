"""Scenario engine, outlook bundle and outlook report. Synthetic data only."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research_cli.evidence.outlook import build_outlook_bundle
from research_cli.evidence.scenario_schema import (
    ScenarioResult,
    confidence_for_episodes,
)
from research_cli.query_parser import parse_query
from research_cli.scenarios.engine import build_scenarios
from research_cli.scenarios.library import SCENARIOS
from research_cli.scenarios.measure import (
    count_episodes,
    episode_return,
    forward_returns,
    horizon_outcome,
    monthly_frequency,
    trigger_series,
)
from research_cli.synthesis.outlook_report import render_outlook_report, sections_for
from research_cli.synthesis.validator import validate_report

TODAY = date(2026, 9, 21)


@pytest.fixture(scope="module")
def tsla_outlook():
    return build_outlook_bundle(parse_query("is Tesla going to go up", today=TODAY), source="mock")


@pytest.fixture(scope="module")
def spy_outlook():
    return build_outlook_bundle(parse_query("is SPY going to go up", today=TODAY), source="mock")


# --- parser ------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "is Tesla going to go up", "will NVDA keep rising", "AAPL outlook",
    "where is MSFT heading", "should i buy TSLA, will it go up",
])
def test_forward_looking_questions_parse_as_outlook(query):
    assert parse_query(query, today=TODAY).question_type == "outlook"


def test_attribution_marker_first_still_wins():
    parsed = parse_query("why did TSLA drop, is it going to recover", today=TODAY)
    assert parsed.question_type == "attribution"


# --- pure measurements ------------------------------------------------------

def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_forward_returns_never_extend_past_the_data():
    closes = pd.Series([100.0, 110.0, 121.0], index=_days(3))
    forward = forward_returns(closes, 1)
    assert forward.iloc[0] == pytest.approx(10.0)
    assert np.isnan(forward.iloc[-1])


def test_trigger_unknown_is_not_false():
    series = pd.Series(np.linspace(100, 200, 10), index=_days(10))
    flags = trigger_series("change_pct", series, 5, 20.0)
    assert flags.iloc[:5].isna().all()
    assert (flags.iloc[5:] == 1.0).all()


def test_drawdown_and_level_triggers():
    series = pd.Series([100.0] * 30 + [70.0] * 5, index=_days(35))
    flags = trigger_series("drawdown_pct", series, 30, -20.0)
    assert flags.iloc[-1] == 1.0 and flags.iloc[25] == 0.0
    calm = trigger_series("level_below", pd.Series([12.0, 18.0], index=_days(2)), 1, 15.0)
    assert list(calm) == [1.0, 0.0]


def test_episode_counting_splits_on_gaps():
    mask = pd.Series([True, True, False, True] + [False] * 20 + [True], index=_days(25))
    assert count_episodes(mask, gap=10) == 2
    assert count_episodes(pd.Series([False] * 5, index=_days(5))) == 0


def test_monthly_frequency_counts_months_not_days():
    index = pd.bdate_range("2021-01-01", "2021-03-31")
    flags = pd.Series(0.0, index=index)
    flags.loc["2021-02-10"] = 1.0
    assert monthly_frequency(flags) == (3, 1)


def test_horizon_outcome_and_empty_sample():
    outcome = horizon_outcome(pd.Series([-10.0, 0.0, 5.0, 10.0]), 21)
    assert outcome.median_pct == pytest.approx(2.5)
    assert outcome.worst_pct == -10.0
    assert outcome.share_lower_pct == pytest.approx(25.0)
    empty = horizon_outcome(pd.Series(dtype=float), 21)
    assert empty.median_pct is None and empty.n_days == 0


def test_episode_before_listing_is_none_not_zero():
    closes = pd.Series([10.0, 11.0], index=pd.to_datetime(["2015-01-02", "2015-01-05"]))
    assert episode_return(closes, date(2008, 9, 15), date(2009, 3, 9)) is None


def test_confidence_comes_from_episodes():
    assert [confidence_for_episodes(n) for n in (0, 1, 2, 4, 10)] == [
        "none", "none", "low", "moderate", "high"]


# --- engine and bundle ----------------------------------------------------------

def test_every_library_scenario_is_present(tsla_outlook):
    assert [s.scenario_id for s in tsla_outlook.scenarios] == [s.scenario_id for s in SCENARIOS]
    assert tsla_outlook.question_type == "outlook"
    assert tsla_outlook.is_synthetic


def test_unmeasurable_scenarios_carry_no_number(tsla_outlook):
    for scenario in tsla_outlook.scenarios:
        if not scenario.quantified:
            assert scenario.frequency_pct is None
            assert scenario.outcomes == ()
            assert scenario.watch_for


def test_too_few_episodes_quotes_no_outcome(tsla_outlook):
    for scenario in tsla_outlook.scenarios:
        if scenario.quantified and scenario.episodes_detected < 2:
            assert scenario.outcomes == ()
            assert scenario.note


def test_company_shock_does_not_apply_to_funds(spy_outlook):
    shock = next(s for s in spy_outlook.scenarios if s.scenario_id == "company_shock")
    assert not shock.quantified and "fund" in shock.note


def test_scenario_numbers_are_bundle_fields(tsla_outlook):
    fields = tsla_outlook.numeric_fields()
    assert any(path.startswith("scenarios[") for path in fields)
    assert any(path.startswith("scenario_baseline.") for path in fields)


def test_history_ends_at_as_of_date():
    pack = build_scenarios("TSLA", "TSLA", date(2015, 6, 1), source="mock", years=6)
    assert pack.baseline.history_end <= date(2015, 6, 1)
    assert all(e.end <= date(2015, 6, 1) for s in pack.results for e in s.examples)


def test_scenario_result_is_frozen():
    result = ScenarioResult(scenario_id="x", label="x", plain_label="x",
                            description="x", quantified=False)
    with pytest.raises(Exception):
        result.label = "y"  # type: ignore[misc]


# --- report ------------------------------------------------------------------

@pytest.mark.parametrize("depth", ["beginner", "intermediate", "analyst"])
def test_outlook_report_passes_the_validator(tsla_outlook, depth):
    report = render_outlook_report(tsla_outlook, depth)
    result = validate_report(report, tsla_outlook, required_sections=sections_for("outlook"))
    assert result.ok, [str(issue) for issue in result.issues]
    assert "These are not predictions" in report


def test_outlook_report_never_claims_a_chance_of_the_future(tsla_outlook):
    report = render_outlook_report(tsla_outlook).lower()
    for phrase in ("will go up", "will rise", "chance it will", "probability of"):
        assert phrase not in report
