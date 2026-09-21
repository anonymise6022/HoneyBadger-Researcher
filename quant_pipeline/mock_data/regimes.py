"""Multi-asset return panels with explicitly injected regime shifts.

A regime detector tested only on i.i.d. noise learns nothing, and one
tested only on real crashes cannot be debugged, because the truth is
unknown. This module sits in between: it produces a panel whose regime at
every date is *known by construction*, so `regime_detection` can be checked
against ground truth.

The generator drives two quantities through a schedule -- the common
volatility level and the cross-asset correlation -- because those are the
two things that actually move together in a real stress episode, and both
are what the topological and rough-volatility diagnostics respond to:

    calm      low vol, low correlation. Points fill the cube; few loops.
    buildup   vol and correlation both climbing. The cloud collapses toward
              its diagonal and starts to show the coordinated oscillation
              that Gidea & Katz (2018) report as a rising H1 landscape norm.
    crash     high vol, correlation near 1, negative drift.
    recovery  vol decaying back to calm, correlation still elevated.

Returns are drawn from a multivariate Student-t rather than a Gaussian so
that tail behaviour is realistic; kurtosis matters to anything estimating
volatility from squared returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["REGIME_LABELS", "RegimeEpisode", "default_episodes", "regime_shift_return_panel"]

REGIME_LABELS = ("calm", "buildup", "crash", "recovery")

_DEFAULT_TAIL_DF = 5.0  # Student-t degrees of freedom; 5 gives finite kurtosis


@dataclass(frozen=True)
class RegimeEpisode:
    """One contiguous stretch of days sharing a regime label and parameters.

    `vol` and `correlation` are given as (start, end) pairs and interpolated
    linearly across the episode, which is what makes `buildup` a build-up
    rather than a step change. `drift` is the annualized expected return.
    """

    label: str
    days: int
    vol: tuple[float, float]
    correlation: tuple[float, float]
    drift: float = 0.0

    def __post_init__(self) -> None:
        if self.label not in REGIME_LABELS:
            raise ValueError(f"label must be one of {REGIME_LABELS}, got {self.label!r}")
        if self.days < 1:
            raise ValueError(f"episode must span at least one day, got {self.days}")
        if any(v <= 0.0 for v in self.vol):
            raise ValueError(f"episode volatilities must be positive, got {self.vol}")
        for rho in self.correlation:
            if not -0.5 < rho < 1.0:
                raise ValueError(
                    f"equicorrelation must lie in (-0.5, 1) to stay positive-definite for "
                    f"several assets, got {rho}"
                )


def default_episodes(n_days: int) -> list[RegimeEpisode]:
    """A three-crisis schedule stretched to fill `n_days`.

    Proportions (roughly 48% calm, 15% buildup, 10% crash, 27% recovery)
    are set so a rolling detector sees plenty of unstressed data to
    calibrate its z-scores against, which is the realistic ratio and the
    one that makes false-positive rates meaningful.

    The third crisis is deliberately placed so that the last 5% of the
    sample -- about 90 business days in a seven-year panel, which is what
    `demo_results.py` reports by default -- contains a complete arc:
    buildup, crash, then the start of a recovery. A schedule whose final
    quarter was uneventful would leave any short evaluation window sitting
    inside one regime, which shows nothing about whether the regime layer
    discriminates between regimes.

    Say plainly what that means: the evaluation window is positioned to
    contain the event. That makes the demo a legible test of whether the
    machinery responds to a regime change it is *known* to be facing. It
    is not a measure of how often the detector fires when nothing happens,
    which is the number that decides whether a detector is useful and which
    this schedule is not designed to estimate. For that, run over the whole
    panel, where roughly three quarters of days are calm or recovering, and
    count the flags raised outside the buildups.

    Episode weights are percentages of `n_days` and may be fractional; each
    is rounded to at least one day.
    """
    if n_days < 40:
        raise ValueError(f"need at least 40 days to lay out a regime schedule, got {n_days}")
    unit = n_days / 100.0
    plan = [
        ("calm", 23, (0.11, 0.13), (0.15, 0.25), 0.08),
        ("buildup", 7, (0.13, 0.26), (0.25, 0.70), -0.02),
        ("crash", 4, (0.26, 0.55), (0.70, 0.92), -1.10),
        ("recovery", 14, (0.55, 0.18), (0.92, 0.40), 0.30),
        ("calm", 17, (0.14, 0.12), (0.30, 0.20), 0.10),
        ("buildup", 6, (0.12, 0.24), (0.20, 0.65), -0.01),
        ("crash", 4, (0.24, 0.48), (0.65, 0.90), -0.95),
        ("recovery", 12, (0.48, 0.16), (0.90, 0.35), 0.25),
        ("calm", 8, (0.15, 0.13), (0.28, 0.22), 0.09),
        ("buildup", 2.5, (0.13, 0.25), (0.22, 0.68), -0.02),
        ("crash", 1.5, (0.25, 0.46), (0.68, 0.90), -1.00),
        ("recovery", 1.0, (0.46, 0.26), (0.90, 0.48), 0.28),
    ]
    episodes = [
        RegimeEpisode(label, max(1, round(weight * unit)), vol, corr, drift)
        for label, weight, vol, corr, drift in plan
    ]
    # Rounding rarely lands on exactly n_days; absorb the difference in the
    # opening calm stretch so every crisis keeps its intended length.
    shortfall = n_days - sum(e.days for e in episodes)
    first = episodes[0]
    episodes[0] = RegimeEpisode(
        first.label, max(1, first.days + shortfall), first.vol, first.correlation, first.drift
    )
    return episodes


def _equicorrelated_draw(
    rng: np.random.Generator, n_assets: int, rho: float, tail_df: float
) -> np.ndarray:
    """One unit-variance multivariate-t draw with equicorrelation rho.

    Equicorrelation factorizes: X_i = sqrt(rho) * F + sqrt(1 - rho) * E_i
    with F and E_i independent standard normals gives Corr(X_i, X_j) = rho
    exactly, for any number of assets, with no Cholesky factorization and
    no chance of an indefinite matrix. The whole vector is then divided by
    sqrt(chi2_df / df), the standard normal variance-mixture construction
    of the multivariate t, and rescaled to unit variance.
    """
    common = rng.standard_normal()
    idiosyncratic = rng.standard_normal(n_assets)
    gaussian = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idiosyncratic
    mixing = np.sqrt(rng.chisquare(tail_df) / tail_df)
    # Var(t_df) = df/(df-2); divide it out so `vol` means what it says.
    return gaussian / mixing * np.sqrt((tail_df - 2.0) / tail_df)


def regime_shift_return_panel(
    n_days: int = 1500,
    n_assets: int = 4,
    episodes: list[RegimeEpisode] | None = None,
    start: str = "2019-01-02",
    tail_df: float = _DEFAULT_TAIL_DF,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate a labelled multi-asset daily return panel.

    Parameters
    ----------
    n_days : total business days to generate.
    n_assets : number of correlated series. Gidea & Katz (2018) use four
        equity indices, which is the default here, and the TDA point cloud
        inherits that dimension.
    episodes : explicit schedule; `default_episodes(n_days)` when omitted.
        A schedule shorter or longer than `n_days` is truncated or the last
        episode is extended, so callers can pass a partial plan.
    start : first business day of the index.
    tail_df : Student-t degrees of freedom, > 2 for finite variance.
    seed : seed for `numpy.random.default_rng`.

    Returns
    -------
    DataFrame indexed by business day with one column per asset
    (`asset_0` ...), plus the ground truth used to generate them:
    `regime` (label), `true_vol` (annualized), `true_corr`.
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be at least 1, got {n_assets}")
    if tail_df <= 2.0:
        raise ValueError(
            f"tail_df must exceed 2 for the t-distribution to have finite variance, got {tail_df}"
        )
    schedule = episodes if episodes is not None else default_episodes(n_days)
    if not schedule:
        raise ValueError("episodes must contain at least one RegimeEpisode")

    labels: list[str] = []
    vols: list[float] = []
    corrs: list[float] = []
    drifts: list[float] = []
    for episode in schedule:
        ramp = np.linspace(0.0, 1.0, episode.days)
        vols.extend(episode.vol[0] + ramp * (episode.vol[1] - episode.vol[0]))
        corrs.extend(episode.correlation[0] + ramp * (episode.correlation[1] - episode.correlation[0]))
        labels.extend([episode.label] * episode.days)
        drifts.extend([episode.drift] * episode.days)

    if len(labels) < n_days:  # extend the final episode rather than wrapping
        pad = n_days - len(labels)
        labels.extend([labels[-1]] * pad)
        vols.extend([vols[-1]] * pad)
        corrs.extend([corrs[-1]] * pad)
        drifts.extend([drifts[-1]] * pad)

    labels, vols, corrs, drifts = labels[:n_days], vols[:n_days], corrs[:n_days], drifts[:n_days]

    rng = np.random.default_rng(seed)
    returns = np.empty((n_days, n_assets), dtype=np.float64)
    for t in range(n_days):
        shock = _equicorrelated_draw(rng, n_assets, float(corrs[t]), tail_df)
        returns[t] = drifts[t] / 252.0 + vols[t] / np.sqrt(252.0) * shock

    index = pd.bdate_range(start=start, periods=n_days, name="date")
    frame = pd.DataFrame(returns, index=index, columns=[f"asset_{i}" for i in range(n_assets)])
    frame["regime"] = labels
    frame["true_vol"] = vols
    frame["true_corr"] = corrs
    return frame
