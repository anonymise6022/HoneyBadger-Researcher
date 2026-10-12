"""The arithmetic behind a plan: risk over a horizon, and sizing to a loss limit.

Pure functions over a close-price series. Two ways of asking "how bad can a
stretch of this length get", used side by side because each covers the
other's blind spot:

**Volatility-based** (the sizing rule). Daily volatility scales with the
square root of time, so a horizon of H trading days has a typical swing of
sigma_daily * sqrt(H). A "bad stretch" is taken as two of those swings down,
capped at 95%. Sizing is then

    position = max_loss / bad_stretch_loss

so that a bad stretch costs about the loss limit. It reacts to the current
level of volatility, which is its strength.

**Empirical** (the reality check). Every past stretch of H days in the
history, measured: 10th percentile, median, 90th percentile, and the worst.
Real returns have fat tails, so the worst stretch is usually worse than two
swings -- which is exactly what a beginner needs to see next to the rule.

A stop ("exit if it falls X%") is simulated on the same stretches: exit at
the first close at or below the stop level. Exiting at that *close* rather
than at the stop price means gaps are included -- a stock that opens 30%
lower skips a 20% stop, and the simulation says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "BAD_STRETCH_SIGMAS",
    "HorizonRisk",
    "horizon_risk",
    "size_for_loss",
    "stop_outcomes",
]

BAD_STRETCH_SIGMAS = 2.0
_MAX_BAD_LOSS_PCT = 95.0
_VOL_LOOKBACK = 63
_LOW, _HIGH = 10, 90


@dataclass(frozen=True)
class HorizonRisk:
    """What stretches of `horizon_days` have looked like for this symbol."""

    horizon_days: int
    current_price: float
    daily_vol_pct: float
    horizon_vol_pct: float
    bad_stretch_loss_pct: float
    p10_pct: float
    median_pct: float
    p90_pct: float
    worst_pct: float
    n_windows: int
    #: The definition of "bad" and "good" stretches, carried as fields so
    #: a report that says "the worst 1 in 10" is quoting the plan.
    low_percentile: int = 10
    high_percentile: int = 90


def _window_returns(closes: pd.Series, horizon: int) -> np.ndarray:
    values = closes.to_numpy(dtype=float)
    if len(values) <= horizon:
        return np.array([])
    return (values[horizon:] / values[:-horizon] - 1.0) * 100.0


def horizon_risk(closes: pd.Series, horizon: int) -> HorizonRisk:
    """Volatility-based and empirical risk over one horizon.

    Raises ValueError when the history holds fewer than two years of
    stretches to look at, because a handful of windows is not a distribution.
    """
    closes = closes.dropna().astype(float)
    windows = _window_returns(closes, horizon)
    if len(windows) < 2 * 252:
        raise ValueError(
            f"only {len(windows)} past stretches of {horizon} trading days are available; "
            "a plan needs at least two years of them"
        )
    recent = closes.pct_change().dropna().tail(_VOL_LOOKBACK)
    daily = float(recent.std(ddof=1) * 100.0)
    horizon_vol = daily * float(np.sqrt(horizon))
    return HorizonRisk(
        horizon_days=horizon,
        current_price=float(closes.iloc[-1]),
        daily_vol_pct=daily,
        horizon_vol_pct=horizon_vol,
        bad_stretch_loss_pct=min(BAD_STRETCH_SIGMAS * horizon_vol, _MAX_BAD_LOSS_PCT),
        p10_pct=float(np.percentile(windows, _LOW)),
        median_pct=float(np.median(windows)),
        p90_pct=float(np.percentile(windows, _HIGH)),
        worst_pct=float(windows.min()),
        n_windows=int(len(windows)),
    )


def size_for_loss(max_loss: float, amount: float, bad_loss_pct: float) -> float:
    """Dollars to invest so a bad stretch costs about `max_loss`, at most `amount`."""
    if max_loss <= 0 or amount <= 0:
        raise ValueError("max_loss and amount must be positive")
    if bad_loss_pct <= 0:
        return amount
    return float(min(amount, max_loss / (bad_loss_pct / 100.0)))


@dataclass(frozen=True)
class StopOutcome:
    """Stretches replayed with a stop: how often it fired, and what followed."""

    stop_pct: float
    hit_share_pct: float
    p10_pct: float
    median_pct: float
    p90_pct: float
    worst_pct: float
    n_windows: int


def stop_outcomes(closes: pd.Series, horizon: int, stop_pct: float) -> StopOutcome:
    """Replay every past stretch with an exit at the first close below the stop."""
    values = closes.dropna().to_numpy(dtype=float)
    starts = len(values) - horizon
    if starts < 1:
        raise ValueError("not enough history to replay a stop")
    level = 1.0 - stop_pct / 100.0
    results, hits = np.empty(starts), 0
    for start in range(starts):
        path = values[start + 1 : start + horizon + 1] / values[start]
        breached = np.flatnonzero(path <= level)
        if len(breached):
            hits += 1
            results[start] = (path[breached[0]] - 1.0) * 100.0
        else:
            results[start] = (path[-1] - 1.0) * 100.0
    return StopOutcome(
        stop_pct=stop_pct,
        hit_share_pct=hits / starts * 100.0,
        p10_pct=float(np.percentile(results, 10)),
        median_pct=float(np.median(results)),
        p90_pct=float(np.percentile(results, 90)),
        worst_pct=float(results.min()),
        n_windows=int(starts),
    )
