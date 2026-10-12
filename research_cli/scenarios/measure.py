"""Pure measurements on price series: triggers, frequencies, forward outcomes.

No fetching and no I/O, so every function here can be tested on a series
whose answer is known.

**No lookahead.** A forward return is measured from a past day to a later
past day; the whole history ends at the as-of date the caller chose, so a
"forward" window never reaches beyond it. Days whose horizon would run past
the end of the data have no forward return and are dropped, not filled.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..evidence.scenario_schema import HorizonOutcome

__all__ = [
    "EPISODE_GAP_SESSIONS",
    "HORIZONS",
    "count_episodes",
    "episode_return",
    "forward_returns",
    "horizon_outcome",
    "monthly_frequency",
    "trigger_series",
]

#: Trading-day horizons: about a week, a month, a quarter.
HORIZONS: tuple[int, ...] = (5, 21, 63)

#: Trigger days more than this many sessions apart start a new episode.
EPISODE_GAP_SESSIONS = 10


def forward_returns(closes: pd.Series, horizon: int) -> pd.Series:
    """Percent change from each day's close to the close `horizon` sessions later."""
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    return (closes.shift(-horizon) / closes - 1.0) * 100.0


def trigger_series(
    kind: str,
    series: pd.Series,
    window: int,
    threshold: float,
    market: pd.Series | None = None,
) -> pd.Series:
    """1.0 where the condition holds, 0.0 where it does not, NaN where unknown.

    Unknown (not enough history yet for the window) is kept distinct from
    "did not hold", so a frequency is measured only over days that could
    have qualified.
    """
    values = series.astype(float)
    if kind == "change_pct":
        measure = values.pct_change(window, fill_method=None) * 100.0
    elif kind == "change_points":
        measure = values.diff(window)
    elif kind == "level_below":
        measure = values
        return _flag(measure, measure < threshold)
    elif kind == "drawdown_pct":
        peak = values.rolling(window, min_periods=max(20, window // 4)).max()
        measure = (values / peak - 1.0) * 100.0
        return _flag(measure, measure <= threshold)
    elif kind == "shock_day":
        returns = values.pct_change(fill_method=None)
        sigma = returns.rolling(window).std().shift(1)
        measure = returns / sigma.where(sigma > 1e-12)
        condition = measure <= threshold
        if market is not None:
            market_move = (market.astype(float).pct_change(fill_method=None) * 100.0).reindex(
                values.index
            )
            condition &= market_move > -1.0
        return _flag(measure, condition)
    else:
        raise ValueError(f"unknown trigger kind {kind!r}")
    condition = measure > threshold if threshold >= 0 else measure < threshold
    return _flag(measure, condition)


def _flag(measure: pd.Series, condition: pd.Series) -> pd.Series:
    flags = condition.astype(float)
    return flags.where(measure.notna())


def monthly_frequency(flags: pd.Series) -> tuple[int, int]:
    """(months with any known value, months in which the condition held at all)."""
    known = flags.dropna()
    if known.empty:
        return 0, 0
    by_month = known.groupby(known.index.to_period("M")).max()
    return int(len(by_month)), int((by_month >= 1.0).sum())


def count_episodes(mask: pd.Series, gap: int = EPISODE_GAP_SESSIONS) -> int:
    """Separate runs of True, where a gap longer than `gap` sessions splits runs."""
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))
    if len(positions) == 0:
        return 0
    return int(1 + np.sum(np.diff(positions) > gap))


def horizon_outcome(sample: pd.Series, horizon: int) -> HorizonOutcome:
    """Summary of forward returns at one horizon. Empty input gives Nones."""
    clean = sample.dropna()
    if clean.empty:
        return HorizonOutcome(horizon_days=horizon, n_days=0)
    return HorizonOutcome(
        horizon_days=horizon,
        median_pct=float(clean.median()),
        p10_pct=float(clean.quantile(0.10)),
        p90_pct=float(clean.quantile(0.90)),
        worst_pct=float(clean.min()),
        share_lower_pct=float((clean < 0).mean() * 100.0),
        share_higher_pct=float((clean > 0).mean() * 100.0),
        n_days=int(len(clean)),
    )


def episode_return(closes: pd.Series, start: date, end: date) -> float | None:
    """Return across a dated episode, from the close before it to its last close.

    None when the series does not reach back to the episode's start: the
    symbol was not listed yet, or the fetched history is shorter.
    """
    if closes.empty or closes.index.min() > pd.Timestamp(start):
        return None
    before = closes[closes.index < pd.Timestamp(start)]
    inside = closes[closes.index <= pd.Timestamp(end)]
    if before.empty or inside.empty:
        return None
    base, last = float(before.iloc[-1]), float(inside.iloc[-1])
    if base <= 0:
        return None
    return (last / base - 1.0) * 100.0
