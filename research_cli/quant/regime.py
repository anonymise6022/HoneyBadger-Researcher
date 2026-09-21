"""Regime diagnostics: is this series trending, mean-reverting, or neither?

Two independent readings, deliberately chosen because they can disagree.

**The Hurst exponent** describes how a series scales with time. H = 0.5 is a
random walk; H > 0.5 means moves tend to continue (trending, persistent);
H < 0.5 means they tend to reverse. Estimated here by rescaled-range
analysis over a spread of lags, then regressing log(R/S) on log(lag) -- the
slope is H. The Anis-Lloyd null expectation is subtracted first, without
which an independent series scores around 0.56 and is confidently
misreported as trending. It is computed on **returns**, never prices: R/S cumulates the
series it is given, so passing prices integrates twice and returns roughly
H + 1, which reports every series as strongly trending. The estimator is
noisy on short samples, so a standard error from the regression is reported
alongside, and the classification refuses to call anything a trend unless H
is more than one standard error clear of 0.5.

**Variance ratio** (Lo & MacKinlay 1988) asks a sharper question: if this
were a random walk, the variance of a q-period return would be exactly q
times the variance of a one-period return. The ratio of those two is 1
under the null. Above 1 means positive autocorrelation (trending), below 1
means reversal. Its advantage over Hurst is a proper test statistic with a
known distribution under heteroskedasticity, so "is this significant" has
an actual answer rather than a rule of thumb.

They disagree often, and that is informative rather than a defect: Hurst
looks at long-memory scaling while the variance ratio looks at one specific
horizon. When they conflict, the honest reading is "no clear regime", which
is what `classify` returns.

References: Hurst (1951); Lo (1991), Econometrica 59(5); Lo & MacKinlay
(1988), Review of Financial Studies 1(1) 41-66.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "RegimeEstimate",
    "estimate_regime",
    "expected_rescaled_range",
    "hurst_exponent",
    "variance_ratio",
]

_MIN_OBSERVATIONS = 120


def expected_rescaled_range(n: int) -> float:
    """E[R/S] for `n` independent observations, per Anis & Lloyd (1976).

    Raw rescaled-range analysis is biased upward on finite samples: a truly
    independent series scores about 0.56 rather than 0.50 at a few thousand
    points, and with a regression standard error near 0.01 that reads as a
    five-sigma trend. Subtracting the theoretical null recenters it.

        E[R/S]_n = ((n - 0.5)/n) * (n*pi/2)^(-1/2) * sum_{i=1}^{n-1} sqrt((n-i)/i)

    Reference: Anis & Lloyd (1976), Biometrika 63(1) 111-116; the (n-0.5)/n
    prefactor is Peters' (1994) small-sample refinement.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    i = np.arange(1, n, dtype=float)
    return float(
        ((n - 0.5) / n) * (n * np.pi / 2.0) ** -0.5 * np.sum(np.sqrt((n - i) / i))
    )


@dataclass(frozen=True)
class RegimeEstimate:
    """What the two diagnostics say, and whether they agree.

    Attributes
    ----------
    hurst : the exponent. 0.5 is a random walk.
    hurst_standard_error : from the log-log regression. Treat any H within
        one of these of 0.5 as indistinguishable from a random walk.
    hurst_r_squared : fit quality of the scaling regression. A low value
        means the series does not scale as a power law at all, and H is
        then not describing anything.
    variance_ratio : VR(q). 1.0 under a random walk.
    variance_ratio_z : Lo-MacKinlay heteroskedasticity-robust statistic.
        Beyond about +/-1.96 is significant at 5%.
    variance_ratio_lag : the q used.
    classification : "trending", "mean-reverting", "random walk", or
        "unclear" when the two disagree.
    """

    hurst: float
    hurst_standard_error: float
    hurst_r_squared: float
    variance_ratio: float
    variance_ratio_z: float
    variance_ratio_lag: int
    classification: str
    n_observations: int

    @property
    def agrees(self) -> bool:
        """Whether both diagnostics point the same way."""
        return self.classification not in ("unclear",)

    def describe(self) -> str:
        """One sentence, without assuming the reader knows either statistic."""
        if self.classification == "trending":
            return (
                f"Moves have tended to continue rather than reverse "
                f"(Hurst {self.hurst:.2f}, variance ratio {self.variance_ratio:.2f})."
            )
        if self.classification == "mean-reverting":
            return (
                f"Moves have tended to reverse rather than continue "
                f"(Hurst {self.hurst:.2f}, variance ratio {self.variance_ratio:.2f})."
            )
        if self.classification == "random walk":
            return (
                f"Behaves like a random walk over this window; neither continuation "
                f"nor reversal is detectable (Hurst {self.hurst:.2f} +/- "
                f"{self.hurst_standard_error:.2f})."
            )
        return (
            f"The two diagnostics disagree (Hurst {self.hurst:.2f} suggests one thing, "
            f"variance ratio {self.variance_ratio:.2f} another), so no regime is claimed."
        )


def hurst_exponent(
    increments: pd.Series, min_lag: int = 8, max_lag: int | None = None
) -> tuple[float, float, float]:
    """Rescaled-range Hurst exponent. Returns (H, standard error, R^2).

    **Takes increments -- returns, not prices.** Rescaled-range analysis
    cumulates the series internally: within each window it takes the
    cumulative deviation from that window's mean, measures the range R of
    that cumulative path, and divides by the window's standard deviation S.
    Feeding it an already-cumulated level therefore integrates twice and
    inflates H by roughly 1.0 -- a random walk's *prices* score about 1.01
    instead of its *returns'* correct 0.50. That mistake produces a
    confident claim of extreme trending on any series at all, so the
    parameter is named for what it must receive.

    Averaging R/S across the windows at each lag and regressing its log on
    log(lag) gives H as the slope.

    Raises ValueError for too little data or a constant series, both of
    which make R/S undefined rather than merely imprecise.
    """
    values = np.asarray(increments.dropna(), dtype=float)
    if len(values) < _MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {_MIN_OBSERVATIONS} observations for a rescaled-range "
            f"estimate, got {len(values)}"
        )
    if np.std(values) <= 1e-12:
        raise ValueError("series is constant; rescaled range is undefined")

    if max_lag is None:
        max_lag = len(values) // 4
    if max_lag <= min_lag:
        raise ValueError(f"max_lag ({max_lag}) must exceed min_lag ({min_lag})")

    lags = np.unique(np.geomspace(min_lag, max_lag, 12).astype(int))
    log_lags: list[float] = []
    log_rs: list[float] = []

    for lag in lags:
        n_windows = len(values) // lag
        if n_windows < 2:
            continue
        ratios: list[float] = []
        for w in range(n_windows):
            window = values[w * lag : (w + 1) * lag]
            deviations = np.cumsum(window - window.mean())
            spread = float(deviations.max() - deviations.min())
            scale = float(window.std(ddof=1))
            if scale > 1e-12 and spread > 0:
                ratios.append(spread / scale)
        if ratios:
            # Subtract the null expectation so the regression measures the
            # departure from independence rather than the estimator's own
            # small-sample bias.
            log_lags.append(float(np.log(lag)))
            log_rs.append(
                float(np.log(np.mean(ratios)) - np.log(expected_rescaled_range(int(lag))))
            )

    if len(log_lags) < 4:
        raise ValueError(
            f"only {len(log_lags)} usable lags; the series is too short or too flat "
            "to estimate a scaling exponent"
        )

    # With the null subtracted the regression slope is the *excess* over a
    # random walk, so 0.5 is added back to return H on its usual scale.
    result = stats.linregress(log_lags, log_rs)
    return float(result.slope + 0.5), float(result.stderr), float(result.rvalue**2)


def variance_ratio(returns: pd.Series, lag: int = 5) -> tuple[float, float]:
    """Lo-MacKinlay variance ratio and its robust z-statistic.

    VR(q) = Var(q-period return) / (q * Var(1-period return)), which is 1
    under a random walk. The z-statistic uses the heteroskedasticity-robust
    variance of Lo & MacKinlay (1988), because financial returns are
    emphatically not homoskedastic and the simpler statistic over-rejects.
    """
    values = np.asarray(returns.dropna(), dtype=float)
    n = len(values)
    if lag < 2:
        raise ValueError(f"lag must be at least 2, got {lag}")
    if n < lag * 10:
        raise ValueError(
            f"need at least {lag * 10} returns for a lag-{lag} variance ratio, got {n}"
        )

    mean = values.mean()
    centered = values - mean
    variance_1 = float(np.sum(centered**2) / (n - 1))
    if variance_1 <= 1e-16:
        raise ValueError("returns have no variance; the ratio is undefined")

    # Overlapping q-period sums, with the standard small-sample correction.
    aggregated = np.convolve(centered, np.ones(lag), mode="valid")
    m = lag * (n - lag + 1) * (1.0 - lag / n)
    variance_q = float(np.sum(aggregated**2) / m)
    ratio = variance_q / variance_1

    # Heteroskedasticity-robust standard error: a weighted sum of the
    # autocorrelation-of-squares terms.
    theta = 0.0
    for j in range(1, lag):
        weight = 2.0 * (lag - j) / lag
        numerator = float(np.sum((centered[j:] ** 2) * (centered[:-j] ** 2)))
        denominator = float(np.sum(centered**2) ** 2 / n)
        if denominator > 0:
            theta += weight**2 * (numerator / denominator)

    z = (ratio - 1.0) / np.sqrt(theta / n) if theta > 0 else 0.0
    return float(ratio), float(z)


def estimate_regime(frame: pd.DataFrame, lag: int = 5) -> RegimeEstimate:
    """Run both diagnostics on a price frame and reconcile them.

    Both diagnostics take **log returns**. Rescaled range cumulates
    internally, so handing it prices would double-integrate and report
    near-1.0 Hurst values for everything; the variance ratio is defined
    over increments in the first place.
    """
    log_price = np.log(frame["Close"].dropna())
    returns = log_price.diff().dropna()

    h, h_error, h_r2 = hurst_exponent(returns)
    ratio, z = variance_ratio(returns, lag)

    # The regression's own standard error is optimistic: the log-log points
    # are correlated across lags, so it understates how much H moves between
    # samples. Measured on independent noise, H lands within about 0.03 of
    # 0.5 but the reported error is near 0.01, which would read as a 3-sigma
    # trend on pure noise. The band is therefore the wider of twice the
    # regression error and a floor of 0.05, which admits no false positives
    # across the seeds tested while still separating an AR(1) at +/-0.5.
    band = max(2.0 * h_error, 0.05)
    hurst_says = (
        "trending" if h > 0.5 + band
        else "mean-reverting" if h < 0.5 - band
        else "random walk"
    )
    vr_says = (
        "trending" if z > 1.96
        else "mean-reverting" if z < -1.96
        else "random walk"
    )

    if hurst_says == vr_says:
        classification = hurst_says
    elif "random walk" in (hurst_says, vr_says):
        # One detects nothing and the other does: report the weaker claim.
        classification = "random walk"
    else:
        classification = "unclear"

    return RegimeEstimate(
        hurst=h,
        hurst_standard_error=h_error,
        hurst_r_squared=h_r2,
        variance_ratio=ratio,
        variance_ratio_z=z,
        variance_ratio_lag=lag,
        classification=classification,
        n_observations=len(returns),
    )
