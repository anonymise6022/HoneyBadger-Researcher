"""Quantitative analysis -- ALPHA, and deliberately kept to one side.

These are standard, well-documented models: GARCH for volatility, rescaled
range and the Lo-MacKinlay variance ratio for regime, an Ornstein-Uhlenbeck
fit for mean reversion. Each implementation is tested against cases whose
answers are known analytically -- a simulated random walk must score a Hurst
of 0.5 and a variance ratio of 1, and they do.

**Why it is labelled alpha anyway.** Testing that a model computes what it
claims is a much weaker statement than testing that the model tells you
anything useful about markets. Run on real daily equity data these
diagnostics mostly report "random walk", which is the honest answer and also
a reminder that the interesting-sounding machinery is not finding much. So:

* Nothing here feeds the attribution or snapshot reports. Those stay on
  observable facts with measured hit rates.
* The desktop app shows this under a section explicitly marked experimental.
* Where these models genuinely earn their place is the **backtest**, where a
  volatility forecast or a regime filter can be tested properly -- fitted on
  past data, scored on data it has never seen, against buy-and-hold. That is
  in `research_cli.backtest.quant_strategies`.

    volatility.py      close-to-close, Yang-Zhang, EWMA, GARCH(1,1) forecast
    regime.py          Hurst exponent, variance ratio, trend/reversion call
    mean_reversion.py  z-score and Ornstein-Uhlenbeck half-life
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .mean_reversion import MeanReversionEstimate, estimate_mean_reversion, half_life
from .regime import RegimeEstimate, estimate_regime, hurst_exponent, variance_ratio
from .volatility import (
    VolatilityEstimate,
    close_to_close_volatility,
    ewma_volatility,
    forecast_volatility,
    yang_zhang_volatility,
)

__all__ = [
    "ALPHA_NOTICE",
    "MeanReversionEstimate",
    "QuantAnalysis",
    "RegimeEstimate",
    "VolatilityEstimate",
    "analyse",
    "close_to_close_volatility",
    "estimate_mean_reversion",
    "estimate_regime",
    "ewma_volatility",
    "forecast_volatility",
    "half_life",
    "hurst_exponent",
    "variance_ratio",
    "yang_zhang_volatility",
]

ALPHA_NOTICE = (
    "Experimental. These models are implemented correctly and tested against "
    "cases with known answers, but that is not the same as being useful for "
    "decisions. On real daily equity data they usually report 'random walk', "
    "which is the honest result. Treat this section as a description of the "
    "series, not as a signal."
)


@dataclass(frozen=True)
class QuantAnalysis:
    """Every quant diagnostic for one symbol, with whatever failed recorded.

    A failure in one model costs that model, not the section: a GARCH fit
    that will not converge should not take the regime estimate down with it.
    """

    symbol: str
    volatility: VolatilityEstimate | None = None
    regime: RegimeEstimate | None = None
    mean_reversion: MeanReversionEstimate | None = None
    failures: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any((self.volatility, self.regime, self.mean_reversion))


def analyse(frame: pd.DataFrame, symbol: str, horizon_days: int = 21) -> QuantAnalysis:
    """Run every diagnostic, degrading gracefully model by model."""
    results: dict[str, object] = {}
    failures: list[str] = []

    for name, call in (
        ("volatility", lambda: forecast_volatility(frame, horizon_days)),
        ("regime", lambda: estimate_regime(frame)),
        ("mean_reversion", lambda: estimate_mean_reversion(frame)),
    ):
        try:
            results[name] = call()
        except Exception as exc:  # noqa: BLE001 - one model must not sink the rest
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    return QuantAnalysis(
        symbol=symbol,
        volatility=results.get("volatility"),  # type: ignore[arg-type]
        regime=results.get("regime"),  # type: ignore[arg-type]
        mean_reversion=results.get("mean_reversion"),  # type: ignore[arg-type]
        failures=tuple(failures),
    )
