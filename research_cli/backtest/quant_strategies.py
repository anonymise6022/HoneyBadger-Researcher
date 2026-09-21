"""Strategies built on the quant models, tested the same sceptical way.

The quant package is labelled alpha because computing a GARCH forecast
correctly is not evidence that it helps. This module is where that question
gets an actual answer: each rule below is run walk-forward -- parameters
chosen on past data, scored on data the method has never seen -- against the
only benchmark that matters, which is buying and holding.

Four rules, chosen because they test genuinely different claims:

**Volatility targeting** claims that holding *constant risk* beats holding a
constant position: lever up when the market is calm, cut when it is wild.
This is the one with the best academic support -- volatility is far more
forecastable than returns, and the effect shows up across asset classes --
so it is the fairest test of whether a quant model adds anything.

**Mean reversion on a z-score** claims prices return to a recent average.
The quant module already reports that daily equity prices mostly do not, so
this rule is expected to disappoint, and watching it disappoint on real data
is more convincing than being told.

**Time-series momentum** claims a security that has risen over the past year
keeps rising. Twelve-month-minus-one-month is the canonical specification
from Moskowitz, Ooi & Pedersen (2012), used unchanged here rather than
tuned, so a poor result cannot be blamed on a bad parameter choice.

**Regime-filtered trend** claims the problem with trend-following is not the
trend rule but *when* it is applied -- that it bleeds in choppy markets and
should stand aside in them. It runs a moving-average rule only while the
Hurst exponent says the series is trending.

**On speed.** The engine calls `signal` once per bar, so anything expensive
has to be cached. Re-estimating a Hurst exponent daily would take minutes
and would also be unrealistic -- nobody re-reads a regime every morning. The
regime filter recomputes every `refresh_bars` and holds its reading in
between, which is both faster and closer to how such a filter would actually
be used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..quant.regime import hurst_exponent
from .engine import SealedWindow

__all__ = [
    "QUANT_TEMPLATES",
    "MeanReversionZScore",
    "RegimeFilteredTrend",
    "TimeSeriesMomentum",
    "VolatilityTargeted",
]

_TRADING_DAYS = 252


def _log_returns(window: SealedWindow) -> np.ndarray:
    closes = window.close.to_numpy(dtype=float)
    return np.diff(np.log(closes))


def _ewma_volatility(returns: np.ndarray, decay: float = 0.94) -> float:
    """Annualized EWMA volatility, in percent.

    EWMA rather than the GARCH used elsewhere: the engine calls this once
    per bar, and a GARCH refit per bar would take minutes for a result that
    tracks EWMA closely at a one-day horizon.
    """
    if len(returns) < 2:
        return float("nan")
    weights = (1.0 - decay) * decay ** np.arange(len(returns) - 1, -1, -1)
    weights /= weights.sum()
    variance = float(np.sum(weights * returns**2))
    return float(np.sqrt(variance * _TRADING_DAYS) * 100.0)


@dataclass
class VolatilityTargeted:
    """Hold constant *risk* rather than a constant position.

    Exposure is set to target_vol / forecast_vol, so a calm market gets a
    larger position and a turbulent one a smaller. Capped at `max_exposure`
    because the arithmetic asks for enormous leverage when volatility is very
    low, and the engine's own cap would silently bind anyway.

    The honest caveat: this improves risk-adjusted return far more reliably
    than raw return, and in a long bull run it will underperform simply
    holding, because it spends that run partly in cash.
    """

    target_vol: float = 15.0
    lookback: int = 60
    max_exposure: float = 1.0
    name: str = "Volatility targeted"

    def __post_init__(self) -> None:
        if self.target_vol <= 0:
            raise ValueError(f"target_vol must be positive, got {self.target_vol}")
        if self.lookback < 20:
            raise ValueError(
                f"lookback must be at least 20 bars for a stable volatility "
                f"estimate, got {self.lookback}"
            )
        self.name = f"Volatility targeted ({self.target_vol:g}%)"

    @property
    def warmup_bars(self) -> int:
        return self.lookback + 1

    def signal(self, window: SealedWindow) -> float:
        returns = _log_returns(window)[-self.lookback :]
        volatility = _ewma_volatility(returns)
        if not np.isfinite(volatility) or volatility <= 1e-6:
            return 0.0
        return float(np.clip(self.target_vol / volatility, 0.0, self.max_exposure))


@dataclass
class MeanReversionZScore:
    """Buy when the price is unusually low against its own recent average.

    Exposure ramps linearly between the thresholds rather than switching, so
    the rule scales into a position instead of flipping on a single bar --
    which halves the turnover and therefore the costs.
    """

    window: int = 63
    entry_z: float = -1.0
    exit_z: float = 0.5
    name: str = "Mean reversion"

    def __post_init__(self) -> None:
        if self.window < 20:
            raise ValueError(f"window must be at least 20 bars, got {self.window}")
        if self.entry_z >= self.exit_z:
            raise ValueError(
                f"entry_z ({self.entry_z}) must be below exit_z ({self.exit_z}); "
                "the rule buys weakness and sells back into strength"
            )
        self.name = f"Mean reversion (z {self.entry_z:g} to {self.exit_z:g})"

    @property
    def warmup_bars(self) -> int:
        return self.window + 1

    def signal(self, window: SealedWindow) -> float:
        closes = window.close.to_numpy(dtype=float)[-self.window :]
        mean, std = float(closes.mean()), float(closes.std(ddof=1))
        if std <= 1e-9:
            return 0.0
        z = (float(closes[-1]) - mean) / std
        if z <= self.entry_z:
            return 1.0
        if z >= self.exit_z:
            return 0.0
        # Linear ramp between the two thresholds.
        return float((self.exit_z - z) / (self.exit_z - self.entry_z))


@dataclass
class TimeSeriesMomentum:
    """Long while the past year's return excluding the last month is positive.

    The canonical 12-1 specification (Moskowitz, Ooi & Pedersen 2012). The
    most recent month is skipped because short-horizon returns tend to
    reverse, and including them dilutes the momentum signal with that noise.
    Parameters are left at the published values rather than tuned, so a poor
    result here cannot be explained away as a bad choice of window.
    """

    lookback: int = 252
    skip: int = 21
    name: str = "Time-series momentum"

    def __post_init__(self) -> None:
        if self.skip < 0 or self.skip >= self.lookback:
            raise ValueError(
                f"skip ({self.skip}) must be non-negative and below lookback "
                f"({self.lookback})"
            )
        self.name = f"Momentum ({self.lookback}-{self.skip})"

    @property
    def warmup_bars(self) -> int:
        return self.lookback + 2

    def signal(self, window: SealedWindow) -> float:
        closes = window.close.to_numpy(dtype=float)
        if len(closes) < self.lookback + 1:
            return 0.0
        recent = closes[-1 - self.skip] if self.skip else closes[-1]
        past = closes[-1 - self.lookback]
        if past <= 0:
            return 0.0
        return 1.0 if recent > past else 0.0


@dataclass
class RegimeFilteredTrend:
    """A moving-average rule that stands aside when the series is not trending.

    The claim being tested: trend-following does not fail because the trend
    rule is wrong, but because it keeps trading in choppy markets where there
    is no trend to follow. The Hurst exponent decides -- above the threshold
    the moving-average rule runs, below it the position goes flat.

    Hurst is re-estimated every `refresh_bars` and held in between. That is a
    performance necessity (rescaled-range analysis is far too slow to run on
    every bar) and also the more realistic choice: a regime reading is a
    slow-moving judgement, not a daily one.

    The cache is keyed on the window's *length*, which increases by exactly
    one each bar, so it cannot serve a stale value from a different point in
    the backtest.
    """

    fast: int = 20
    slow: int = 100
    hurst_threshold: float = 0.5
    hurst_window: int = 400
    refresh_bars: int = 21
    name: str = "Regime-filtered trend"
    _cache: dict[int, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError(
                f"fast ({self.fast}) must be shorter than slow ({self.slow})"
            )
        if self.hurst_window < 150:
            raise ValueError(
                f"hurst_window must be at least 150 bars for a usable rescaled-range "
                f"estimate, got {self.hurst_window}"
            )
        if self.refresh_bars < 1:
            raise ValueError(f"refresh_bars must be at least 1, got {self.refresh_bars}")
        self.name = f"Regime-filtered trend ({self.fast}/{self.slow}, H>{self.hurst_threshold:g})"

    @property
    def warmup_bars(self) -> int:
        return max(self.slow, self.hurst_window) + 1

    def _hurst(self, window: SealedWindow) -> float:
        """Current Hurst, recomputed only every `refresh_bars`."""
        length = len(window)
        bucket = length // self.refresh_bars
        cached = self._cache.get(bucket)
        if cached is not None:
            return cached

        returns = _log_returns(window)[-self.hurst_window :]
        try:
            # Hurst takes increments, never a price level: rescaled range
            # cumulates internally, so passing prices would return H + 1.
            import pandas as pd

            value, _, _ = hurst_exponent(pd.Series(returns))
        except ValueError:
            value = 0.5  # not enough data to judge: do not filter
        self._cache[bucket] = value
        return value

    def signal(self, window: SealedWindow) -> float:
        if self._hurst(window) < self.hurst_threshold:
            return 0.0
        closes = window.close.to_numpy(dtype=float)
        fast_mean = float(closes[-self.fast :].mean())
        slow_mean = float(closes[-self.slow :].mean())
        return 1.0 if fast_mean > slow_mean else 0.0


#: Quant templates, in the same shape as the built-in ones so the engine,
#: the walk-forward harness and the UI need no special cases.
QUANT_TEMPLATES: dict[str, dict] = {
    "vol-target": {
        "factory": lambda **kwargs: VolatilityTargeted(**kwargs),
        "grid": [{"target_vol": v, "lookback": w} for v in (10.0, 15.0, 20.0) for w in (40, 60)],
        "description": "Hold constant risk: bigger when calm, smaller when wild.",
        "alpha": False,
    },
    "mean-reversion": {
        "factory": lambda **kwargs: MeanReversionZScore(**kwargs),
        "grid": [
            {"window": w, "entry_z": e, "exit_z": x}
            for w in (21, 63) for e, x in ((-1.0, 0.5), (-1.5, 0.0), (-2.0, 0.5))
        ],
        "description": "Buy when the price is unusually low against its recent average.",
        "alpha": False,
    },
    "momentum": {
        "factory": lambda **kwargs: TimeSeriesMomentum(**kwargs),
        "grid": [{"lookback": lb, "skip": s} for lb in (126, 252) for s in (0, 21)],
        "description": "Long while the past year's return has been positive.",
        "alpha": False,
    },
    "regime-trend": {
        "factory": lambda **kwargs: RegimeFilteredTrend(**kwargs),
        "grid": [
            {"fast": f, "slow": s, "hurst_threshold": h}
            for f, s in ((20, 100), (50, 200)) for h in (0.5, 0.55)
        ],
        "description": "Trend-follow, but only while the series is actually trending.",
        "alpha": True,
    },
}
