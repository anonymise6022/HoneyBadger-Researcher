"""Tests for the evidence schema and the attribution bundle builder."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research_cli.evidence.schema import (
    CONFIDENCE_WEIGHT,
    CandidateFactor,
    EvidenceBundle,
    confidence_for_sample,
)
from research_cli.evidence.why_moved import build_why_moved_bundle, quintile_hit_rate
from research_cli.query_parser import parse_query

TODAY = date(2026, 9, 21)


def _factor(**overrides) -> CandidateFactor:
    defaults: dict = {
        "factor_id": "f", "label": "Factor", "what_happened": "something happened",
        "observed_value": 1.0, "observed_units": "%", "condition_description": "days like today",
        "historical_hit_rate": 0.70, "base_rate": 0.55, "sample_size": 200,
    }
    defaults.update(overrides)
    return CandidateFactor(**defaults)


# --- schema invariants -----------------------------------------------------

@pytest.mark.parametrize(
    ("n", "label"),
    [(0, "none"), (9, "none"), (10, "low"), (29, "low"), (30, "moderate"),
     (99, "moderate"), (100, "high"), (5000, "high")],
)
def test_confidence_is_a_function_of_sample_size(n: int, label: str) -> None:
    assert confidence_for_sample(n) == label


def test_confidence_cannot_be_set_independently_of_the_sample() -> None:
    """It is a computed property, so it cannot drift from its evidence."""
    assert _factor(sample_size=12).confidence == "low"
    assert _factor(sample_size=400).confidence == "high"


def test_small_sample_hit_rate_is_discarded_not_merely_labelled() -> None:
    """'80% (4 observations)' invites believing the 80% and skimming the 4."""
    factor = _factor(historical_hit_rate=0.80, sample_size=4)
    assert factor.historical_hit_rate is None
    assert factor.note and "too few" in factor.note
    assert factor.rank_score == 0.0


def test_lift_is_hit_rate_minus_base_rate() -> None:
    assert _factor(historical_hit_rate=0.70, base_rate=0.55).lift == pytest.approx(0.15)


def test_a_factor_at_the_base_rate_does_not_beat_it() -> None:
    assert not _factor(historical_hit_rate=0.55, base_rate=0.55).beats_base_rate


def test_a_small_lift_on_a_small_sample_does_not_beat_the_base_rate() -> None:
    """The whole point of the standard-error bar."""
    assert not _factor(historical_hit_rate=0.60, base_rate=0.55, sample_size=20).beats_base_rate


def test_a_large_lift_on_a_large_sample_beats_the_base_rate() -> None:
    assert _factor(historical_hit_rate=0.75, base_rate=0.55, sample_size=400).beats_base_rate


def test_standard_error_shrinks_with_sample_size() -> None:
    small = _factor(sample_size=25).hit_rate_standard_error
    large = _factor(sample_size=2500).hit_rate_standard_error
    assert small > large
    assert large == pytest.approx(small / 10.0, rel=0.01)


def test_rank_score_is_hit_rate_weighted_by_confidence() -> None:
    factor = _factor(historical_hit_rate=0.70, sample_size=200)
    assert factor.rank_score == pytest.approx(0.70 * CONFIDENCE_WEIGHT["high"])


def test_high_confidence_outranks_a_better_rate_on_thin_evidence() -> None:
    thin = _factor(factor_id="thin", historical_hit_rate=0.90, sample_size=12)
    solid = _factor(factor_id="solid", historical_hit_rate=0.70, sample_size=500)
    bundle = EvidenceBundle(
        query="q", question_type="attribution", ticker="X",
        period_description="today", factors=[thin, solid],
    )
    assert next(f.factor_id for f in bundle.ranked_factors()) == "solid"


def test_mechanical_factors_are_excluded_from_supported_evidence() -> None:
    """An index moving with its own sectors is arithmetic, not evidence.

    Letting an identity top the evidence ranking would be the single most
    misleading thing this tool could do.
    """
    mechanical = _factor(
        factor_id="breadth", historical_hit_rate=0.95, sample_size=400,
        relationship="mechanical",
    )
    external = _factor(factor_id="vix", historical_hit_rate=0.68, sample_size=400)
    bundle = EvidenceBundle(
        query="q", question_type="attribution", ticker="SPY",
        period_description="today", factors=[mechanical, external],
    )
    assert [f.factor_id for f in bundle.supported_factors()] == ["vix"]
    assert [f.factor_id for f in bundle.mechanical_factors()] == ["breadth"]


def test_ranking_is_deterministic_regardless_of_insertion_order() -> None:
    a = _factor(factor_id="a", historical_hit_rate=0.70, sample_size=200)
    b = _factor(factor_id="b", historical_hit_rate=0.60, sample_size=200)
    forward = EvidenceBundle(
        query="q", question_type="attribution", ticker="X",
        period_description="t", factors=[a, b],
    ).ranked_factors()
    backward = EvidenceBundle(
        query="q", question_type="attribution", ticker="X",
        period_description="t", factors=[b, a],
    ).ranked_factors()
    assert [f.factor_id for f in forward] == [f.factor_id for f in backward]


def test_numeric_fields_flattens_every_number(spy_bundle) -> None:
    numbers = spy_bundle.numeric_fields()
    assert "observation.close" in numbers
    assert any(key.startswith("factors[") for key in numbers)
    assert all(isinstance(value, float) for value in numbers.values())


def test_numeric_fields_excludes_booleans(spy_bundle) -> None:
    """A boolean is not a figure a report quotes."""
    assert not any(key.endswith("is_unusual") for key in spy_bundle.numeric_fields())


# --- hit-rate machinery ----------------------------------------------------

def _series(values, start="2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_quintile_hit_rate_recovers_a_planted_relationship() -> None:
    """When the factor perfectly predicts direction, the hit rate must be 1."""
    rng = np.random.default_rng(0)
    factor_values = rng.standard_normal(1200)
    target_values = np.where(factor_values > 0, 1.0, -1.0) * rng.uniform(0.1, 2.0, 1200)
    hit, base, n, _ = quintile_hit_rate(
        _series(factor_values), _series(target_values), today_value=2.0, target_sign=1
    )
    assert hit == pytest.approx(1.0)
    assert base == pytest.approx(0.5, abs=0.05)
    assert n > 200


def test_quintile_hit_rate_finds_nothing_in_noise() -> None:
    """An unrelated factor must land at the base rate, not above it."""
    rng = np.random.default_rng(1)
    hit, base, n, _ = quintile_hit_rate(
        _series(rng.standard_normal(1500)), _series(rng.standard_normal(1500)),
        today_value=0.5, target_sign=1,
    )
    assert abs(hit - base) < 0.08
    assert n > 250


def test_quintile_hit_rate_refuses_a_short_history() -> None:
    hit, _base, _n, note = quintile_hit_rate(
        _series(np.arange(50.0)), _series(np.arange(50.0)), today_value=1.0, target_sign=1
    )
    assert hit is None
    assert "not enough history" in note


def test_quintile_hit_rate_handles_a_constant_factor() -> None:
    """A factor with one value cannot be binned; it must not crash."""
    hit, _, _, note = quintile_hit_rate(
        _series(np.ones(500)), _series(np.random.default_rng(2).standard_normal(500)),
        today_value=1.0, target_sign=1,
    )
    assert hit is None
    assert "distinct values" in note


def test_sample_size_is_about_one_fifth_of_history() -> None:
    """Quintiles exist to guarantee a usable sample; check they deliver one."""
    rng = np.random.default_rng(3)
    _, _, n, _ = quintile_hit_rate(
        _series(rng.standard_normal(1000)), _series(rng.standard_normal(1000)),
        today_value=0.0, target_sign=1,
    )
    assert 150 <= n <= 250


# --- the assembled bundle --------------------------------------------------

def test_bundle_has_an_observation_and_factors(spy_bundle) -> None:
    assert spy_bundle.observation is not None
    assert spy_bundle.factors
    assert spy_bundle.question_type == "attribution"


def test_bundle_always_records_limitations(spy_bundle) -> None:
    """An empty limitations list on a real bundle is a bug, not modesty."""
    assert spy_bundle.limitations
    assert any("coincid" in text.lower() for text in spy_bundle.limitations)


def test_index_bundle_marks_breadth_as_mechanical(spy_bundle) -> None:
    ids = {f.factor_id for f in spy_bundle.mechanical_factors()}
    assert "sector_breadth" in ids


def test_single_stock_bundle_decomposes_the_market_move(aapl_bundle) -> None:
    """The most useful thing a beginner can be told about a single stock."""
    ids = {f.factor_id for f in aapl_bundle.factors}
    assert "market_move" in ids
    assert "sector_breadth" not in {f.factor_id for f in aapl_bundle.mechanical_factors()}


def test_wrong_premise_is_contradicted(spy_bundle) -> None:
    """The query said 'fall'; if it rose, the report must say so first."""
    if spy_bundle.observation.direction == "up":
        assert spy_bundle.counterevidence
        assert "did not actually go down" in spy_bundle.counterevidence[0].label


def test_fx_bundle_omits_equity_only_factors() -> None:
    bundle = build_why_moved_bundle(
        parse_query("why did EURUSD rally today", today=TODAY), source="mock"
    )
    ids = {f.factor_id for f in bundle.factors}
    assert "sector_breadth" not in ids
    assert not any(i.startswith("megacap_") for i in ids)
    assert any("currency pair" in text for text in bundle.limitations)


def test_typical_move_is_flagged_as_needing_no_explanation() -> None:
    bundle = build_why_moved_bundle(
        parse_query("why did KO fall today", today=TODAY), source="mock"
    )
    if bundle.observation.outlier_label == "typical":
        labels = " ".join(c.label for c in bundle.counterevidence)
        assert "not unusual" in labels


def test_history_years_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_why_moved_bundle(
            parse_query("why did SPY fall today", today=TODAY),
            source="mock", history_years=0.5,
        )


def test_bundle_round_trips_through_json(spy_bundle) -> None:
    """The bundle is the LLM's input in Phase 2, so it must serialize."""
    restored = EvidenceBundle.model_validate_json(spy_bundle.model_dump_json())
    assert restored.ticker == spy_bundle.ticker
    assert len(restored.factors) == len(spy_bundle.factors)
    assert restored.observation.close == pytest.approx(spy_bundle.observation.close)
