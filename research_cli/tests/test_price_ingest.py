"""Tests for price ingestion and the outlier measurement it exists to provide."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research_cli.data.mock_data import mock_price_history, symbol_seed
from research_cli.data.price_ingest import (
    OHLCV_COLUMNS,
    PriceDataError,
    UnknownSymbolError,
    build_observation,
    fetch_history,
    fetch_many,
)

TODAY = date(2026, 9, 21)


# --- mock generator --------------------------------------------------------

def test_mock_history_is_deterministic_across_calls() -> None:
    """Same symbol, same data -- otherwise no test could assert a value."""
    first = mock_price_history("AAPL", "2024-01-01", "2026-01-01")
    second = mock_price_history("AAPL", "2024-01-01", "2026-01-01")
    pd.testing.assert_frame_equal(first, second)


def test_mock_history_differs_between_symbols() -> None:
    a = mock_price_history("AAPL", "2024-01-01", "2026-01-01")
    b = mock_price_history("MSFT", "2024-01-01", "2026-01-01")
    assert not np.allclose(a["Close"].to_numpy(), b["Close"].to_numpy())


def test_symbol_seed_is_stable_across_processes() -> None:
    """Python salts hash() per process; the seed must not depend on it."""
    assert symbol_seed("AAPL") == symbol_seed("aapl")
    assert symbol_seed("AAPL") != symbol_seed("MSFT")


def test_mock_ohlc_bars_are_internally_consistent() -> None:
    frame = mock_price_history("TSLA", "2024-01-01", "2026-01-01")
    assert (frame["High"] >= frame[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (frame["Low"] <= frame[["Open", "Close"]].min(axis=1) + 1e-9).all()
    assert (frame["Low"] > 0).all()
    assert (frame["Volume"] >= 0).all()


def test_mock_volatility_tracks_the_symbol_profile() -> None:
    """A high-beta name must actually be more volatile than an index fund."""
    spy = mock_price_history("SPY", "2022-01-01", "2026-01-01")["Close"].pct_change().std()
    tsla = mock_price_history("TSLA", "2022-01-01", "2026-01-01")["Close"].pct_change().std()
    assert tsla > spy * 2.0


def test_mock_returns_have_fat_tails() -> None:
    """Outlier detection needs data where outliers actually occur."""
    returns = mock_price_history("SPY", "2018-01-01", "2026-01-01")["Close"].pct_change().dropna()
    beyond_two_sigma = (returns.abs() > 2 * returns.std()).mean()
    assert beyond_two_sigma > 0.046, "should exceed the 4.6% a normal distribution gives"


def test_mock_history_rejects_an_empty_range() -> None:
    with pytest.raises(ValueError, match="no business days"):
        mock_price_history("SPY", "2026-01-03", "2026-01-04")  # a weekend


# --- fetching --------------------------------------------------------------

def test_mock_source_never_touches_the_network() -> None:
    frame, source, warnings = fetch_history("SPY", date(2025, 1, 1), TODAY, source="mock")
    assert source == "mock"
    assert list(frame.columns) == list(OHLCV_COLUMNS)
    assert not warnings


def test_backwards_range_is_refused() -> None:
    with pytest.raises(PriceDataError, match="backwards"):
        fetch_history("SPY", TODAY, date(2025, 1, 1), source="mock")


def test_invalid_source_is_refused() -> None:
    with pytest.raises(ValueError, match="source must be"):
        fetch_history("SPY", date(2025, 1, 1), TODAY, source="nonsense")  # type: ignore[arg-type]


def test_fetch_many_drops_failures_rather_than_raising() -> None:
    """Losing one context symbol must cost that factor, not the report."""
    result = fetch_many(["SPY", "QQQ"], date(2025, 1, 1), TODAY, source="mock")
    assert set(result) == {"SPY", "QQQ"}


# --- the observation -------------------------------------------------------

def test_observation_measures_the_move_against_prior_volatility(today: date = TODAY) -> None:
    observation = build_observation("SPY", today, today, source="mock")
    expected = (observation.close / observation.prior_close - 1.0) * 100.0
    assert observation.return_pct == pytest.approx(expected)
    assert observation.sigma_multiple == pytest.approx(
        observation.period_return_pct / observation.daily_vol_pct
    )


def test_volatility_baseline_excludes_the_day_being_judged() -> None:
    """Otherwise a large move shrinks its own sigma multiple.

    The context window must end strictly before the session, so the last
    context row is the previous session's bar.
    """
    observation = build_observation("SPY", TODAY, TODAY, source="mock")
    assert observation.context.index.max() < pd.Timestamp(observation.session_date)
    assert len(observation.context) == 20


def test_annualized_volatility_is_the_daily_figure_scaled() -> None:
    observation = build_observation("AAPL", TODAY, TODAY, source="mock")
    assert observation.annualized_vol_pct == pytest.approx(
        observation.daily_vol_pct * np.sqrt(252.0)
    )


@pytest.mark.parametrize(
    ("sigma", "label"),
    [(0.5, "typical"), (1.5, "1-sigma"), (2.5, "2-sigma"), (4.0, "3-sigma+")],
)
def test_outlier_labels_match_their_thresholds(sigma: float, label: str) -> None:
    from research_cli.data.price_ingest import _classify_outlier

    assert _classify_outlier(sigma) == label
    assert _classify_outlier(-sigma) == label, "the label must not depend on direction"


def test_non_trading_day_steps_back_and_says_so() -> None:
    """A Sunday question must resolve to Friday, visibly."""
    sunday = date(2026, 9, 20)
    observation = build_observation("SPY", sunday, sunday, source="mock")
    assert observation.session_date < sunday
    assert observation.session_date.weekday() == 4
    assert any("not a trading day" in w for w in observation.warnings)


def test_weekend_query_does_not_report_a_zero_move() -> None:
    """Regression: the period baseline once collapsed onto the session itself.

    When the requested date is a non-trading day the session steps back, and
    measuring the period from the last session before the *requested* date
    made that session its own baseline -- every weekend query reported
    exactly 0.00%.
    """
    sunday = date(2026, 9, 20)
    observation = build_observation("GME", sunday, sunday, source="mock")
    assert observation.period_return_pct != 0.0
    assert observation.period_return_pct == pytest.approx(observation.return_pct)


def test_multi_day_period_includes_the_first_days_move() -> None:
    """The baseline is the close *before* the window, not its first close."""
    end = TODAY
    start = end - timedelta(days=7)
    observation = build_observation("SPY", start, end, source="mock")
    history = observation.history
    before = history.index[history.index < pd.Timestamp(start)]
    expected = (observation.close / float(history.loc[before[-1], "Close"]) - 1.0) * 100.0
    assert observation.period_return_pct == pytest.approx(expected)


def test_too_short_a_lookback_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 5 sessions"):
        build_observation("SPY", TODAY, TODAY, lookback=3, source="mock")


def test_too_little_history_is_refused_with_guidance(monkeypatch) -> None:
    """A newly listed symbol has no baseline, and must say so, not guess one.

    The mock generator will happily synthesize any date range, so this drives
    the guard directly with a short frame -- which is what a real freshly
    listed ticker returns.
    """
    from research_cli.data import price_ingest

    # Four sessions: the target plus three of context, below the five the
    # volatility baseline needs.
    short = mock_price_history("NEWCO", "2026-09-16", "2026-09-21")
    monkeypatch.setattr(
        price_ingest, "fetch_history", lambda *a, **k: (short, "mock", ())
    )
    with pytest.raises(PriceDataError) as excinfo:
        build_observation("NEWCO", TODAY, TODAY, source="mock", lookback=20)
    assert "at least 5" in str(excinfo.value)
    assert excinfo.value.suggestion


def test_single_session_of_history_is_refused(monkeypatch) -> None:
    """With one bar there is no previous close, so no return exists."""
    from research_cli.data import price_ingest

    one_bar = mock_price_history("NEWCO", "2026-09-14", "2026-09-21").tail(1)
    monkeypatch.setattr(
        price_ingest, "fetch_history", lambda *a, **k: (one_bar, "mock", ())
    )
    with pytest.raises(PriceDataError, match="one session"):
        build_observation("NEWCO", TODAY, TODAY, source="mock")


def test_describe_move_is_readable_without_a_glossary() -> None:
    text = build_observation("SPY", TODAY, TODAY, source="mock").describe_move()
    assert "standard deviations" in text
    assert "%" in text
    for jargon in ("sigma_multiple", "nan", "None"):
        assert jargon not in text


# --- unknown symbols must never be fabricated ------------------------------

def test_an_empty_vendor_response_is_an_unknown_symbol(monkeypatch) -> None:
    """The vendor answered and has no such ticker; that is not a fallback case."""
    from research_cli.data import price_ingest

    monkeypatch.setattr(
        price_ingest, "_fetch_live",
        lambda *a, **k: (_ for _ in ()).throw(
            UnknownSymbolError("'ZZZZQQ' is not real", "check the spelling")
        ),
    )
    with pytest.raises(UnknownSymbolError):
        price_ingest.fetch_history("ZZZZQQ", date(2025, 1, 1), TODAY, source="auto")


def test_unknown_symbol_does_not_degrade_to_synthetic_data(monkeypatch) -> None:
    """Regression: a typo once produced a full, confident report on invented data.

    A visible "synthetic" warning is not sufficient mitigation for inventing
    a company. The tool must refuse.
    """
    from research_cli.data import price_ingest

    monkeypatch.setattr(
        price_ingest, "_fetch_live",
        lambda *a, **k: (_ for _ in ()).throw(UnknownSymbolError("nope")),
    )
    with pytest.raises(UnknownSymbolError):
        build_observation("AAPI", TODAY, TODAY, source="auto")


def test_a_network_failure_still_falls_back(monkeypatch) -> None:
    """The opposite case: Yahoo being down must not stop the tool working."""
    from research_cli.data import price_ingest

    monkeypatch.setattr(
        price_ingest, "_fetch_live",
        lambda *a, **k: (_ for _ in ()).throw(PriceDataError("connection refused")),
    )
    frame, source, warnings = price_ingest.fetch_history(
        "SPY", date(2025, 1, 1), TODAY, source="auto", use_cache=False
    )
    assert source == "mock"
    assert any("NOT real" in w for w in warnings)
    assert len(frame) > 0


def test_live_source_refuses_to_fall_back(monkeypatch) -> None:
    from research_cli.data import price_ingest

    monkeypatch.setattr(
        price_ingest, "_fetch_live",
        lambda *a, **k: (_ for _ in ()).throw(PriceDataError("connection refused")),
    )
    with pytest.raises(PriceDataError):
        price_ingest.fetch_history(
            "SPY", date(2025, 1, 1), TODAY, source="live", use_cache=False
        )


def test_empty_frame_normalization_raises_unknown_symbol() -> None:
    from research_cli.data.price_ingest import _normalize_frame

    with pytest.raises(UnknownSymbolError, match="currently listed"):
        _normalize_frame(pd.DataFrame(), "ZZZZQQ")
