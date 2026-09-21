"""Mean-reversion statistics: how far from normal, and how fast it returns.

Two numbers, and the second is the one people forget.

**The z-score** says how far the price sits from its recent average, in
units of its own variability. Minus two means "unusually cheap relative to
the last few months", which is the easy half.

**The half-life** says how long it has historically taken to close half that
gap. This is what decides whether a z-score is tradeable. A half-life of six
days is a trade; a half-life of nine months is a way to have your money tied
up while being technically right. It comes from fitting an
Ornstein-Uhlenbeck process, whose discrete form is an AR(1):

    dP = theta * (mu - P) dt + sigma dW      =>      P_t = a + b*P_{t-1} + e

with b = exp(-theta), so theta = -ln(b) and the half-life is ln(2)/theta.

**Deciding whether there is any reversion at all is the hard part, and the
obvious test is wrong.** It is tempting to fit the AR(1), look at how many
standard errors b sits below 1, and call anything past the usual bar
mean-reverting. That fails badly, for two compounding reasons:

* The OLS estimate of b is *biased downward* near a unit root (Hurwicz
  bias), so a true random walk tends to fit at b slightly under 1.
* Under a unit root the t-statistic does not follow a normal distribution.
  It follows the Dickey-Fuller distribution, whose 5% critical value is
  near -2.86 rather than -1.96, with a much fatter left tail.

Together these mean a plain random walk routinely fits at what looks like
three standard errors below 1. An earlier version of this module used that
naive bar and confidently reported simulated random walks as
"mean-reverting" with a 137-day half-life. `is_mean_reverting` now rests on
an Augmented Dickey-Fuller test, which is built for exactly this null and
uses the right critical values.

**When the half-life is meaningless.** If the ADF test cannot reject a unit
root, there is no reversion to have a half-life, and `half_life_days` is
reported as None rather than as a large number that reads like slow
reversion.

Reference: Dickey & Fuller (1979), JASA 74(366) 427-431.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "MeanReversionEstimate",
    "estimate_mean_reversion",
    "half_life",
    "unit_root_pvalue",
]

_MIN_OBSERVATIONS = 60


@dataclass(frozen=True)
class MeanReversionEstimate:
    """Distance from the recent mean, and how quickly that gap has closed.

    Attributes
    ----------
    z_score : current price minus rolling mean, over rolling std.
    half_life_days : trading days to close half the gap, or None when the
        series shows no reversion.
    ar_coefficient : the fitted b. 1.0 is a random walk.
    ar_standard_error : its standard error. Kept for reference only -- it is
        *not* what decides whether the series reverts, because it is not
        valid under a unit root.
    adf_pvalue : Augmented Dickey-Fuller p-value for the null that the
        series has a unit root. Below 0.05 rejects it. None when the test
        could not be run.
    current_price, rolling_mean, rolling_std : the inputs to the z-score.
    window : bars in the rolling statistics.
    """

    z_score: float
    half_life_days: float | None
    ar_coefficient: float
    ar_standard_error: float
    current_price: float
    rolling_mean: float
    rolling_std: float
    window: int
    n_observations: int
    adf_pvalue: float | None = None

    @property
    def is_mean_reverting(self) -> bool:
        """Whether a unit root can actually be rejected.

        Uses the ADF p-value, never the AR(1) t-statistic: see the module
        docstring on why the latter calls random walks mean-reverting.
        When the test could not be run the answer is False, because
        "unknown" must not be presented as reversion.
        """
        return self.adf_pvalue is not None and self.adf_pvalue < 0.05

    @property
    def stretch(self) -> str:
        """A plain reading of the z-score."""
        magnitude = abs(self.z_score)
        if magnitude < 1.0:
            return "near its recent average"
        if magnitude < 2.0:
            direction = "above" if self.z_score > 0 else "below"
            return f"moderately {direction} its recent average"
        direction = "above" if self.z_score > 0 else "below"
        return f"unusually far {direction} its recent average"

    def describe(self) -> str:
        """One sentence a non-specialist can act on."""
        base = (
            f"The price is {self.stretch} "
            f"({self.z_score:+.1f} standard deviations)."
        )
        if not self.is_mean_reverting:
            return (
                base + " It shows no reliable tendency to return to that average, "
                "so being far from it does not by itself imply a move back."
            )
        if self.half_life_days is None:
            return base
        return (
            base + f" Historically it has taken about {self.half_life_days:.0f} "
            "trading days to close half of a gap like this."
        )


def unit_root_pvalue(series: pd.Series) -> float | None:
    """Augmented Dickey-Fuller p-value for the null of a unit root.

    Low values reject the null, i.e. support stationarity and therefore
    reversion. Returns None when statsmodels is unavailable or the test
    cannot be computed, and callers treat None as "no reversion" rather
    than guessing.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < _MIN_OBSERVATIONS:
        return None
    # A constant but no trend: the question is whether the level reverts to
    # a mean, not whether it reverts around a drift.
    #
    # `result_object=True` is the form statsmodels is migrating to. The
    # older tuple form still works but emits a FutureWarning, and a caller
    # running with warnings-as-errors would see that surface as a failed
    # test rather than a deprecation -- which is exactly how this path was
    # first found to be swallowing errors.
    try:
        try:
            result = adfuller(values, regression="c", autolag="AIC", result_object=True)
            return float(result.pvalue)
        except TypeError:  # statsmodels older than the result-object API
            return float(adfuller(values, regression="c", autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        # Too few distinct values, or a singular design matrix.
        return None


def half_life(series: pd.Series) -> tuple[float | None, float, float]:
    """Ornstein-Uhlenbeck half-life. Returns (half-life or None, b, se(b)).

    Fits P_t = a + b*P_{t-1} by least squares. The half-life is
    ln(2) / -ln(b), defined only for 0 < b < 1; anything else means no
    reversion and returns None rather than a misleading number.
    """
    values = series.dropna()
    if len(values) < _MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {_MIN_OBSERVATIONS} observations to fit an AR(1), "
            f"got {len(values)}"
        )

    lagged = values.shift(1).dropna()
    current = values.loc[lagged.index]
    result = stats.linregress(lagged.to_numpy(), current.to_numpy())
    b, error = float(result.slope), float(result.stderr)

    if not 0.0 < b < 1.0:
        return None, b, error
    theta = -np.log(b)
    if theta <= 1e-12:
        return None, b, error
    return float(np.log(2.0) / theta), b, error


def estimate_mean_reversion(
    frame: pd.DataFrame, window: int = 63
) -> MeanReversionEstimate:
    """Z-score against a rolling window, plus the fitted reversion speed.

    `window` defaults to 63 bars, about a quarter -- long enough that the
    mean is not dragged around by the last week, short enough to describe
    the current regime rather than a two-year average.
    """
    if window < 10:
        raise ValueError(f"window must be at least 10 bars, got {window}")
    closes = frame["Close"].dropna().astype(float)
    if len(closes) < max(window + 10, _MIN_OBSERVATIONS):
        raise ValueError(
            f"need at least {max(window + 10, _MIN_OBSERVATIONS)} bars, got {len(closes)}"
        )

    recent = closes.tail(window)
    mean, std = float(recent.mean()), float(recent.std(ddof=1))
    current = float(closes.iloc[-1])
    z = (current - mean) / std if std > 1e-12 else 0.0

    life, b, error = half_life(closes)
    pvalue = unit_root_pvalue(closes)
    # A half-life is only meaningful once a unit root has been rejected.
    if pvalue is None or pvalue >= 0.05:
        life = None
    return MeanReversionEstimate(
        z_score=float(z),
        half_life_days=life,
        ar_coefficient=b,
        ar_standard_error=error,
        current_price=current,
        rolling_mean=mean,
        rolling_std=std,
        window=window,
        n_observations=len(closes),
        adf_pvalue=pvalue,
    )
