"""Volatility estimation and forecasting.

Three estimators, in increasing order of how much they assume:

**Close-to-close** is the textbook standard deviation of daily returns. It
throws away the high and the low, so it uses roughly a quarter of the
information in each bar and is correspondingly noisy.

**Yang-Zhang** (2000) uses the full OHLC bar and decomposes variance into
overnight, open-to-close and intraday components. It is the most efficient
of the drift-independent estimators -- roughly 7-14x the efficiency of
close-to-close, meaning a given accuracy needs that much less data. It is
the right default when OHLC bars are available, which here they always are.

**EWMA** weights recent observations more heavily, with a decay that makes
it react to a volatility change within days rather than the ~month a
rolling window takes. RiskMetrics' lambda of 0.94 is the convention and is
the default here.

Forecasting uses **GARCH(1,1)** when the `arch` package is available. Its
appeal is not accuracy on any single day -- it is that it produces a term
structure, so a one-day and a one-month forecast differ in the direction
they should, converging toward the long-run level. When `arch` is missing
or the fit fails, the forecast degrades to EWMA and says so, because a
silently different model is worse than a stated simpler one.

**On alpha status.** These are standard, well-documented estimators and the
implementations are tested against closed-form cases. What is *not*
established is that any of them helps you make money, which is why the
containing package is labelled experimental and why nothing here feeds the
attribution or snapshot reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS",
    "VolatilityEstimate",
    "close_to_close_volatility",
    "ewma_volatility",
    "forecast_volatility",
    "yang_zhang_volatility",
]

TRADING_DAYS = 252
_MIN_OBSERVATIONS = 20
_RISKMETRICS_LAMBDA = 0.94


@dataclass(frozen=True)
class VolatilityEstimate:
    """Current volatility by several methods, plus a forecast.

    All figures are annualized percentages. `forecast_model` names what
    actually produced the forecast, which may not be what was requested --
    a GARCH fit that fails degrades to EWMA and records it here rather than
    pretending.
    """

    close_to_close_pct: float
    yang_zhang_pct: float
    ewma_pct: float
    forecast_pct: float
    forecast_horizon_days: int
    forecast_model: str
    long_run_pct: float | None
    persistence: float | None
    n_observations: int
    notes: tuple[str, ...] = ()

    @property
    def regime(self) -> str:
        """Where current volatility sits against its own long-run level."""
        if self.long_run_pct is None or self.long_run_pct <= 0:
            return "unknown"
        ratio = self.ewma_pct / self.long_run_pct
        if ratio > 1.35:
            return "elevated"
        if ratio < 0.75:
            return "subdued"
        return "normal"


def _returns(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame["Close"] / frame["Close"].shift(1)).dropna()


def close_to_close_volatility(frame: pd.DataFrame, window: int = 21) -> float:
    """Annualized standard deviation of log returns over the last `window` bars."""
    returns = _returns(frame).tail(window)
    if len(returns) < 3:
        raise ValueError(f"need at least 3 returns, got {len(returns)}")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0)


def yang_zhang_volatility(frame: pd.DataFrame, window: int = 21) -> float:
    """Yang-Zhang volatility: the OHLC estimator, annualized percent.

    sigma^2 = sigma_overnight^2 + k*sigma_open_to_close^2 + (1-k)*sigma_rs^2

    where sigma_rs is Rogers-Satchell (which handles drift) and

        k = 0.34 / (1.34 + (n+1)/(n-1))

    is chosen to minimize variance of the combined estimator. Requires
    Open, High, Low and Close; raises if any is missing rather than
    silently falling back, since a caller asking for Yang-Zhang wants the
    efficiency and should be told if it is unavailable.
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Yang-Zhang needs full OHLC bars; missing {sorted(missing)}"
        )

    bars = frame.tail(window + 1)
    if len(bars) < _MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {_MIN_OBSERVATIONS} bars for a stable Yang-Zhang "
            f"estimate, got {len(bars)}"
        )

    open_, high = bars["Open"], bars["High"]
    low, close = bars["Low"], bars["Close"]
    previous_close = close.shift(1)

    overnight = np.log(open_ / previous_close).dropna()
    open_to_close = np.log(close / open_).iloc[1:]
    # Rogers-Satchell: independent of drift, which is why it is the
    # component that carries the intraday range.
    rogers_satchell = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    ).iloc[1:]

    n = len(overnight)
    if n < 3:
        raise ValueError(f"need at least 3 complete bars, got {n}")

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    variance = (
        float(overnight.var(ddof=1))
        + k * float(open_to_close.var(ddof=1))
        + (1.0 - k) * float(rogers_satchell.mean())
    )
    # Rogers-Satchell can go slightly negative on a tiny sample; clamp at
    # the close-to-close estimate rather than returning a NaN from sqrt.
    if variance <= 0:
        return close_to_close_volatility(frame, window)
    return float(np.sqrt(variance * TRADING_DAYS) * 100.0)


def ewma_volatility(frame: pd.DataFrame, decay: float = _RISKMETRICS_LAMBDA) -> float:
    """Exponentially weighted volatility, annualized percent.

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2

    `decay` is RiskMetrics' 0.94 by default: a half-life of about 11 days,
    so the estimate responds to a volatility shift inside a fortnight.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay must lie strictly in (0, 1), got {decay}")
    returns = _returns(frame)
    if len(returns) < _MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {_MIN_OBSERVATIONS} returns, got {len(returns)}"
        )
    weighted = returns.pow(2).ewm(alpha=1.0 - decay, adjust=False).mean()
    return float(np.sqrt(weighted.iloc[-1] * TRADING_DAYS) * 100.0)


def _garch_forecast(
    returns: pd.Series, horizon: int
) -> tuple[float, float | None, float | None, str, tuple[str, ...]]:
    """Fit GARCH(1,1) and forecast. Falls back to EWMA on any failure."""
    try:
        from arch import arch_model
    except ImportError:
        return (float("nan"), None, None, "unavailable", ("the arch package is not installed",))

    # arch wants percentage returns; fitting on decimals makes the
    # optimizer struggle and silently return poor parameters.
    scaled = returns * 100.0
    try:
        model = arch_model(scaled, vol="GARCH", p=1, q=1, mean="Zero", dist="t")
        fitted = model.fit(disp="off", show_warning=False)
        forecast = fitted.forecast(horizon=horizon, reindex=False)
        # Average variance across the horizon, then annualize.
        path_variance = float(forecast.variance.to_numpy()[-1].mean())
        annualized = float(np.sqrt(path_variance * TRADING_DAYS))

        params = fitted.params
        alpha = float(params.get("alpha[1]", np.nan))
        beta = float(params.get("beta[1]", np.nan))
        omega = float(params.get("omega", np.nan))
        persistence = alpha + beta
        notes: tuple[str, ...] = ()
        long_run = None
        if np.isfinite(persistence) and persistence < 1.0 and np.isfinite(omega):
            long_run = float(np.sqrt(omega / (1.0 - persistence) * TRADING_DAYS))
        elif np.isfinite(persistence):
            notes = (
                (
                    f"GARCH persistence is {persistence:.3f}, at or above 1: the "
                    "fitted process has no finite long-run variance, so no long-run "
                    "level is shown"
                ),
            )
        return annualized, long_run, persistence, "GARCH(1,1)", notes
    except Exception as exc:  # noqa: BLE001 - arch raises many untyped errors
        return (
            float("nan"), None, None, "failed",
            (f"GARCH fit failed ({type(exc).__name__}); using EWMA instead",),
        )


def forecast_volatility(
    frame: pd.DataFrame, horizon_days: int = 21, window: int = 21
) -> VolatilityEstimate:
    """Estimate current volatility three ways and forecast it forward.

    Raises ValueError for too little history or a non-positive horizon.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")
    returns = _returns(frame)
    if len(returns) < 60:
        raise ValueError(
            f"need at least 60 returns to fit a volatility model, got {len(returns)}"
        )

    close_to_close = close_to_close_volatility(frame, window)
    try:
        yang_zhang = yang_zhang_volatility(frame, window)
    except ValueError:
        yang_zhang = close_to_close
    ewma = ewma_volatility(frame)

    forecast, long_run, persistence, model, notes = _garch_forecast(returns, horizon_days)
    if not np.isfinite(forecast):
        forecast, model = ewma, "EWMA (GARCH unavailable)"

    return VolatilityEstimate(
        close_to_close_pct=close_to_close,
        yang_zhang_pct=yang_zhang,
        ewma_pct=ewma,
        forecast_pct=forecast,
        forecast_horizon_days=horizon_days,
        forecast_model=model,
        long_run_pct=long_run,
        persistence=persistence,
        n_observations=len(returns),
        notes=notes,
    )
