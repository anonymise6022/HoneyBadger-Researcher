"""Performance and risk statistics for an equity curve.

Pure functions over a series of portfolio values. No strategy logic, no data
access, nothing that could see the future -- which is what makes these safe
to compute anywhere, including inside a walk-forward test fold.

Three things here are easy to get subtly wrong, so they are stated:

**Sharpe needs a risk-free rate and an annualization factor**, and omitting
either silently changes the answer. A "Sharpe of 1.4" computed against a
zero risk-free rate in a 4% rate environment is not comparable to one that
subtracts it. Both are explicit arguments with documented defaults.

**Maximum drawdown is measured on the equity curve, not on returns**, and it
is the worst peak-to-trough fall, not the worst single day. A strategy can
have a mild worst day and a catastrophic drawdown.

**Win rate is the least informative statistic on this page** and is included
because it is the one beginners ask for. A strategy that wins 90% of the
time and loses everything on the tenth trade has a 90% win rate. It is
reported next to the payoff ratio for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS",
    "PerformanceStats",
    "annualized_return",
    "annualized_volatility",
    "compute_stats",
    "drawdown_series",
    "max_drawdown",
    "sharpe_ratio",
]

TRADING_DAYS = 252
_MIN_POINTS = 3


@dataclass(frozen=True)
class PerformanceStats:
    """Headline statistics for one equity curve.

    Attributes
    ----------
    total_return_pct : end value over start value, minus one.
    annualized_return_pct : geometric, using the actual elapsed periods.
    annualized_volatility_pct : standard deviation of periodic returns,
        scaled by sqrt(periods per year).
    sharpe_ratio : excess return over volatility. None when volatility is
        zero -- a flat curve has no meaningful Sharpe, and reporting
        infinity would look like a spectacular result.
    max_drawdown_pct : worst peak-to-trough decline, as a negative number.
    max_drawdown_days : trading days from the prior peak to the worst trough.
        This is the duration of the *deepest* drawdown, which is not
        necessarily the longest one: SPY's worst fall was 34% in the 23
        sessions of February-March 2020, while its slowest recovery was the
        far shallower slide through 2022.
    win_rate_pct : share of *moving* periods that moved up. Periods with
        exactly zero return are excluded from both sides, because a day
        spent in cash is neither a win nor a loss -- counting it as a loss
        made a strategy that sits flat half the time report a 27% win rate
        while losing money on barely a quarter of days.
    payoff_ratio : mean winning period over mean losing period. Shown beside
        the win rate because neither means much alone.
    n_periods : observations the statistics rest on.
    n_trades : round trips, when the caller tracked them.
    """

    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float | None
    max_drawdown_pct: float
    max_drawdown_days: int
    win_rate_pct: float
    payoff_ratio: float | None
    n_periods: int
    n_trades: int = 0

    def summary_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for display, formatted once, in one place."""
        sharpe = "n/a" if self.sharpe_ratio is None else f"{self.sharpe_ratio:.2f}"
        payoff = "n/a" if self.payoff_ratio is None else f"{self.payoff_ratio:.2f}"
        return [
            ("Total return", f"{self.total_return_pct:+.2f}%"),
            ("Annualized return", f"{self.annualized_return_pct:+.2f}%"),
            ("Annualized volatility", f"{self.annualized_volatility_pct:.2f}%"),
            ("Sharpe ratio", sharpe),
            ("Maximum drawdown", f"{self.max_drawdown_pct:.2f}%"),
            ("Peak to trough", f"{self.max_drawdown_days} days"),
            ("Win rate", f"{self.win_rate_pct:.1f}%"),
            ("Payoff ratio", payoff),
            ("Trades", str(self.n_trades)),
            ("Periods", str(self.n_periods)),
        ]


def _validate(equity: pd.Series) -> pd.Series:
    """Coerce and check an equity curve before measuring anything."""
    if not isinstance(equity, pd.Series):
        raise TypeError(f"equity must be a Series, got {type(equity).__name__}")
    clean = equity.astype(float).dropna()
    if len(clean) < _MIN_POINTS:
        raise ValueError(
            f"need at least {_MIN_POINTS} points to compute statistics, got {len(clean)}"
        )
    if (clean <= 0).any():
        raise ValueError(
            "equity curve contains a non-positive value; a portfolio that reaches zero "
            "cannot have its returns measured multiplicatively"
        )
    return clean


def annualized_return(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualized return, in percent.

    Uses the number of periods actually observed, so a six-month backtest is
    annualized as a six-month backtest rather than assumed to be a year.
    """
    clean = _validate(equity)
    years = (len(clean) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    growth = float(clean.iloc[-1] / clean.iloc[0])
    return float((growth ** (1.0 / years) - 1.0) * 100.0)


def annualized_volatility(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Standard deviation of periodic returns, annualized, in percent."""
    returns = _validate(equity).pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year) * 100.0)


def sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float | None:
    """Excess return per unit of volatility.

    `risk_free_rate` is an annual decimal (0.04 for 4%) and is converted to
    a per-period rate geometrically. Returns None when volatility is zero:
    a flat curve has no meaningful Sharpe, and dividing by zero would print
    an infinity that reads as a triumph.
    """
    returns = _validate(equity).pct_change().dropna()
    if len(returns) < 2:
        return None
    volatility = float(returns.std(ddof=1))
    if volatility <= 1e-12:
        return None
    per_period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - per_period_rf
    return float(excess.mean() / volatility * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Percentage below the running peak at every point, as negative numbers."""
    clean = _validate(equity)
    return (clean / clean.cummax() - 1.0) * 100.0


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Worst peak-to-trough decline, and how many periods it took.

    Returns (percent, periods). The percent is negative or zero. Measured on
    the equity curve rather than on returns: the worst single day and the
    worst drawdown are different numbers, and the second is the one that
    decides whether somebody abandons a strategy.
    """
    drawdowns = drawdown_series(equity)
    trough = int(np.argmin(drawdowns.to_numpy()))
    worst = float(drawdowns.iloc[trough])
    if worst >= 0.0:
        return 0.0, 0
    before = drawdowns.iloc[: trough + 1]
    peak_positions = np.flatnonzero(before.to_numpy() >= -1e-12)
    peak = int(peak_positions[-1]) if len(peak_positions) else 0
    return worst, int(trough - peak)


def compute_stats(
    equity: pd.Series,
    n_trades: int = 0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceStats:
    """Compute every headline statistic for an equity curve."""
    clean = _validate(equity)
    returns = clean.pct_change().dropna()

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    # Flat periods are excluded from the denominator; see PerformanceStats.
    moving = len(wins) + len(losses)
    payoff = (
        float(wins.mean() / abs(losses.mean()))
        if len(wins) and len(losses) and abs(losses.mean()) > 1e-12
        else None
    )
    worst_drawdown, drawdown_length = max_drawdown(clean)

    return PerformanceStats(
        total_return_pct=float((clean.iloc[-1] / clean.iloc[0] - 1.0) * 100.0),
        annualized_return_pct=annualized_return(clean, periods_per_year),
        annualized_volatility_pct=annualized_volatility(clean, periods_per_year),
        sharpe_ratio=sharpe_ratio(clean, risk_free_rate, periods_per_year),
        max_drawdown_pct=worst_drawdown,
        max_drawdown_days=drawdown_length,
        win_rate_pct=float(len(wins) / moving * 100.0) if moving else 0.0,
        payoff_ratio=payoff,
        n_periods=len(clean),
        n_trades=int(n_trades),
    )
