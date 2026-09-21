"""Synthetic macro series that imitate the FRED releases the pipeline uses.

Five series, at their real FRED frequencies and under their real FRED IDs,
so that swapping the mock for the live fetcher changes nothing downstream:

    CPIAUCSL      CPI-U, all items, index 1982-84 = 100   monthly, level
    UNRATE        civilian unemployment rate               monthly, percent
    DGS2          2-year Treasury constant maturity        daily, percent
    DGS10         10-year Treasury constant maturity       daily, percent
    BAMLH0A0HYM2  ICE BofA US high-yield option-adjusted   daily, percent

The series are not independent, and that is the point. A macro feature set
whose inputs are five unrelated random walks will look far more informative
in-sample than it deserves to, because the regression can spend degrees of
freedom on noise that is uncorrelated by construction. Here the short rate
responds to inflation and slack through a Taylor-type rule, the curve
inverts when policy tightens past neutral, and credit spreads widen when
unemployment rises -- so the features carry the collinearity that real
macro features carry.

Calibration is to *plausibility*, not to history: roughly 2-4% inflation,
unemployment near 4%, a 2-year rate in the low single digits, and
high-yield spreads oscillating around 400bp with stress spikes past 800bp.
Do not read anything into a backtest run on this data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["MOCK_SERIES_IDS", "mock_consensus", "mock_macro_panel"]

MOCK_SERIES_IDS = ("CPIAUCSL", "UNRATE", "DGS2", "DGS10", "BAMLH0A0HYM2")

# Typical one-release survey dispersion, in the units of each series. CPI is
# forecast in month-over-month percent (a 0.09 index-point error on a ~300
# level is about 3bp of m/m), unemployment to the nearest tenth.
_CONSENSUS_ERROR_SD = {"CPIAUCSL": 0.09, "UNRATE": 0.12}


def _ornstein_uhlenbeck(
    n: int, mean: float, reversion: float, vol: float, rng: np.random.Generator, x0: float | None = None
) -> np.ndarray:
    """Discretized OU path: x_t = x_{t-1} + kappa(mu - x_{t-1}) + sigma eps_t."""
    path = np.empty(n, dtype=np.float64)
    path[0] = mean if x0 is None else x0
    shocks = rng.standard_normal(n) * vol
    for t in range(1, n):
        path[t] = path[t - 1] + reversion * (mean - path[t - 1]) + shocks[t]
    return path


def _monthly_macro(
    months: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[pd.Series, pd.Series]:
    """CPI index level and unemployment rate on a month-start index.

    Inflation is modelled as an AR(1) in month-over-month log change rather
    than as i.i.d. noise, because inflation persistence is the single
    property that makes a CPI surprise informative about the next one.
    """
    n = len(months)
    # m/m log inflation, AR(1) around 0.21%/month (~2.5% annualized).
    persistence, target = 0.72, 0.0021
    inflation = np.empty(n, dtype=np.float64)
    inflation[0] = target
    innovations = rng.standard_normal(n) * 0.0011
    for t in range(1, n):
        inflation[t] = target + persistence * (inflation[t - 1] - target) + innovations[t]
    # One supply-shock surge in the middle third, of the kind that drives
    # the large consensus misses the surprise features are built to catch.
    surge_start = n // 3
    surge = np.zeros(n)
    surge_len = min(n - surge_start, max(6, n // 6))
    surge[surge_start : surge_start + surge_len] = 0.0032 * np.sin(
        np.linspace(0.0, np.pi, surge_len)
    )
    cpi = pd.Series(250.0 * np.exp(np.cumsum(inflation + surge)), index=months, name="CPIAUCSL")

    unemployment = _ornstein_uhlenbeck(n, 4.3, 0.06, 0.13, rng, x0=4.6)
    # Unemployment rises with a lag into the inflation surge, as a policy
    # response would push it; and FRED publishes it to one decimal place.
    lagged_surge = np.roll(surge, 6)
    lagged_surge[:6] = 0.0
    unemployment = np.round(np.clip(unemployment + 220.0 * lagged_surge, 3.2, 11.0), 1)
    return cpi, pd.Series(unemployment, index=months, name="UNRATE")


def mock_macro_panel(
    start: str = "2018-01-01", end: str = "2024-12-31", seed: int | None = None
) -> dict[str, pd.Series]:
    """Generate all five series, each on its own native frequency index.

    Returns a mapping from FRED series ID to a float Series named after that
    ID. Monthly series are indexed at month start (FRED's convention for
    CPIAUCSL and UNRATE); daily series are indexed on business days, which
    is when the Treasury and ICE curves actually print.

    The returned values are *observation* values dated by reference period.
    They carry no publication lag -- applying the lag, and so avoiding
    look-ahead bias, is `macro_model.data_ingest.align_to_daily`'s job.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(start=start, end=end, freq="MS")
    if len(months) < 24:
        raise ValueError(
            f"need at least 24 months between {start} and {end} for year-over-year "
            f"features to exist; got {len(months)}"
        )
    bdays = pd.bdate_range(start=start, end=end, name="date")
    if len(bdays) < 60:
        raise ValueError(f"need at least 60 business days, got {len(bdays)}")

    cpi, unemployment = _monthly_macro(months, rng)

    # Daily drivers: carry the monthly state forward to business days.
    yoy = (cpi / cpi.shift(12) - 1.0) * 100.0
    yoy_daily = yoy.reindex(bdays.union(months)).ffill().reindex(bdays).bfill()
    unrate_daily = unemployment.reindex(bdays.union(months)).ffill().reindex(bdays).bfill()

    # Taylor-type policy rule: r* + 1.4 * inflation gap + 0.5 * slack gap.
    neutral = 2.0
    policy = np.clip(
        neutral + 1.4 * (yoy_daily.to_numpy() - 2.0) + 0.5 * (4.3 - unrate_daily.to_numpy()), 0.0, 9.0
    )
    # The 2-year prices the expected path, so it tracks a smoothed rule.
    smoothed = pd.Series(policy, index=bdays).ewm(span=90, adjust=False).mean().to_numpy()
    dgs2 = np.clip(smoothed + _ornstein_uhlenbeck(len(bdays), 0.0, 0.03, 0.045, rng), 0.02, None)

    # Slope compresses and inverts as policy runs above neutral -- the
    # empirical regularity the 10y-2y feature exists to capture.
    slope_base = _ornstein_uhlenbeck(len(bdays), 1.05, 0.01, 0.035, rng)
    slope = slope_base - 0.55 * np.clip(smoothed - neutral, 0.0, None)
    dgs10 = np.clip(dgs2 + slope, 0.05, None)

    # High-yield OAS: log-OU around ~390bp, widening with unemployment.
    unrate_gap = unrate_daily.to_numpy() - 4.3
    log_oas = _ornstein_uhlenbeck(len(bdays), np.log(3.9), 0.012, 0.028, rng) + 0.30 * unrate_gap
    credit = np.exp(log_oas)

    return {
        "CPIAUCSL": cpi,
        "UNRATE": unemployment,
        "DGS2": pd.Series(np.round(dgs2, 2), index=bdays, name="DGS2"),
        "DGS10": pd.Series(np.round(dgs10, 2), index=bdays, name="DGS10"),
        "BAMLH0A0HYM2": pd.Series(np.round(credit, 2), index=bdays, name="BAMLH0A0HYM2"),
    }


def mock_consensus(
    observations: dict[str, pd.Series], seed: int | None = None
) -> dict[str, pd.Series]:
    """Fabricate survey-consensus forecasts for the monthly releases.

    Real consensus figures (Bloomberg ECO, Action Economics, Reuters polls)
    are licensed and have no free API, so a mock pipeline has to invent
    them. They are generated as the actual value plus mean-zero noise of
    realistic dispersion, which makes the resulting surprises genuinely
    unpredictable -- the honest null. Two consequences worth stating:

    1. A forecaster fitted on these surprises should find *no* edge in them
       beyond what the surprise's correlation with the other features
       gives it. If it does, something is leaking.
    2. Real consensus is biased and serially correlated (forecasters anchor
       and herd). Nothing here reproduces that, so a surprise feature that
       works on this data has not been shown to work on real data.

    Only the monthly releases get a consensus: daily market series (yields,
    spreads) are prices, not forecast releases, and have no survey.
    """
    rng = np.random.default_rng(seed)
    consensus: dict[str, pd.Series] = {}
    for series_id, error_sd in _CONSENSUS_ERROR_SD.items():
        if series_id not in observations:
            continue
        actual = observations[series_id]
        noise = rng.standard_normal(len(actual)) * error_sd
        consensus[series_id] = pd.Series(
            actual.to_numpy() - noise, index=actual.index, name=f"{series_id}_consensus"
        )
    return consensus
