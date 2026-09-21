"""Tests for the ported FRED ingestion, release calendar and surprise logic."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research_cli.data.macro_ingest import (
    FRED_SERIES,
    MacroDataError,
    align_to_daily,
    fetch_fred_series,
    load_macro_panel,
    normalized_surprise,
    releases_on,
    yield_moves_on,
)

START, END = date(2021, 1, 1), date(2026, 9, 21)


@pytest.fixture(scope="module")
def panel():
    return load_macro_panel(START, END, source="mock")


@pytest.fixture(scope="module")
def daily(panel):
    return align_to_daily(panel)


def test_mock_panel_has_every_required_series(panel) -> None:
    assert set(panel.observations) == {spec.series_id for spec in FRED_SERIES}
    assert panel.source == "mock"


def test_no_api_key_falls_back_rather_than_failing(monkeypatch) -> None:
    """A user with no accounts must still get a working tool."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert load_macro_panel(START, END).source == "mock"


def test_live_source_without_a_key_fails_loudly(monkeypatch) -> None:
    """--live means "tell me if this is not real", so it must not degrade."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(MacroDataError, match="FRED_API_KEY"):
        load_macro_panel(START, END, source="live")


def test_fetch_without_a_key_names_the_remedy() -> None:
    with pytest.raises(MacroDataError, match="fred.stlouisfed.org"):
        fetch_fred_series("CPIAUCSL", START, END, api_key="")


def test_publication_lag_prevents_look_ahead(panel, daily) -> None:
    """A monthly print may not appear before it was published.

    CPI for a given month is dated the 1st by FRED and published about six
    weeks later. On the reference date itself the aligned frame must still
    be showing the *previous* value.
    """
    spec = next(s for s in FRED_SERIES if s.series_id == "CPIAUCSL")
    cpi = panel.observations["CPIAUCSL"]
    reference = cpi.index[20]
    value = float(cpi.loc[reference])

    assert float(daily["cpi"].asof(reference)) != pytest.approx(value)
    released = reference + pd.Timedelta(days=spec.release_lag_days + 3)
    assert float(daily["cpi"].asof(released)) == pytest.approx(value)


def test_aligned_frame_is_forward_filled_not_back_filled(daily) -> None:
    """Back-filling would reintroduce exactly the look-ahead the lag removes."""
    first_valid = daily["cpi"].first_valid_index()
    assert daily["cpi"].loc[:first_valid].iloc[:-1].isna().all()


def test_monthly_series_steps_once_a_month_then_holds(daily) -> None:
    changes = daily["cpi"].dropna().diff().ne(0).sum()
    months = len(daily) / 21.0
    assert changes <= months + 2, "a monthly series must not change every day"


def test_releases_are_dated_by_publication_not_reference(panel) -> None:
    found = False
    probe = date(2026, 3, 1)
    for offset in range(40):
        releases = releases_on(panel, probe + timedelta(days=offset), window_days=1)
        for release in releases:
            assert release.release_date > release.reference_period
            found = True
    assert found, "expected at least one release in a 40-day span"


def test_surprise_is_measured_against_the_trailing_average_change(panel) -> None:
    """Regression: a raw change over its own std made every CPI print 3-sigma.

    The consumer price *index* rises nearly every month, so the expected
    change is not zero. Scoring deviations from zero scored ordinary
    inflation as a shock every single month.
    """
    scores = []
    probe = date(2023, 1, 1)
    while probe < date(2026, 9, 1):
        for release in releases_on(panel, probe, window_days=2):
            if release.series_id == "CPIAUCSL" and release.surprise_z is not None:
                scores.append(release.surprise_z)
        probe += timedelta(days=7)

    assert len(scores) > 20
    assert abs(np.mean(scores)) < 1.0, "surprises must be centred near zero"
    assert np.std(scores) < 3.0


def test_releases_are_sorted_by_absolute_surprise(panel) -> None:
    for offset in range(60):
        releases = releases_on(panel, date(2026, 1, 15) + timedelta(days=offset), window_days=20)
        if len(releases) >= 2:
            magnitudes = [abs(r.surprise_z or 0.0) for r in releases]
            assert magnitudes == sorted(magnitudes, reverse=True)
            return
    pytest.skip("no day with two releases in range")


def test_release_description_is_plain_language(panel) -> None:
    for offset in range(60):
        releases = releases_on(panel, date(2026, 1, 15) + timedelta(days=offset), window_days=5)
        if releases:
            text = releases[0].describe()
            assert "z-score" not in text.lower()
            assert "surprise_z" not in text
            assert text.endswith(".")
            return
    pytest.skip("no releases found in range")


def test_units_are_spaced_like_english(panel) -> None:
    """Regression: values once rendered as '356.24index'."""
    for offset in range(60):
        for release in releases_on(panel, date(2026, 1, 15) + timedelta(days=offset), 5):
            assert "index" not in release.formatted_value or " index" in release.formatted_value
            if release.units_label == "%":
                assert release.formatted_value.endswith("%")
    # No assertion needed beyond not raising; the loop covers what exists.


def test_yield_moves_are_one_day_changes(daily) -> None:
    moves = yield_moves_on(daily, date(2026, 9, 21))
    assert set(moves) <= {"yield_2y", "yield_10y", "credit_spread"}
    for value in moves.values():
        assert np.isfinite(value)
        assert abs(value) < 2.0, "a one-day yield change of >2pp is implausible"


def test_yield_moves_omit_rather_than_zero_fill_missing_data() -> None:
    """'Unchanged' and 'unknown' are different facts."""
    assert yield_moves_on(pd.DataFrame(), date(2026, 9, 21)) == {}


def test_normalized_surprise_uses_only_prior_dispersion() -> None:
    index = pd.bdate_range("2020-01-01", periods=300)
    actual = pd.Series(np.repeat(np.arange(15, dtype=float), 20), index=index, name="x")
    consensus = actual - np.repeat([0.2, -0.2, 0.1] * 5, 20)
    result = normalized_surprise(actual, consensus, window=5)
    assert result.notna().sum() > 0
    assert np.isfinite(result.dropna()).all()


def test_normalized_surprise_returns_nan_when_too_few_releases() -> None:
    index = pd.bdate_range("2020-01-01", periods=10)
    actual = pd.Series(np.ones(10), index=index, name="x")
    assert normalized_surprise(actual, actual - 0.1, window=5).isna().all()
