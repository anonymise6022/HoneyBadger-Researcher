"""Rough volatility: Hurst estimation and the RFSV forecast.

Gatheral, Jaisson & Rosenbaum (2018) observed that for essentially every
liquid asset, the log of realized volatility behaves like fractional
Brownian motion with a Hurst exponent near 0.1 -- far below the 0.5 of a
standard diffusion, and far below the 0.5+ that long-memory volatility
models had assumed for two decades. Volatility is *rough*: its paths are
much more jagged than Brownian motion, while its autocovariance still
decays slowly enough to look like long memory over any finite sample.

**Estimating H.** The estimator is a scaling law, not a likelihood. Define

    m(q, Delta) = mean_t | log sigma_{t+Delta} - log sigma_t |^q

If log-volatility is fBm with Hurst H and scale nu, its increments over a
gap Delta are N(0, nu^2 Delta^{2H}), so

    log m(q, Delta) = q log nu + q H log Delta + log K_q,
    K_q = E|N(0,1)|^q = 2^{q/2} Gamma((q+1)/2) / sqrt(pi)

Two nested regressions follow. For each q, regress log m(q, Delta) on
log Delta; the slope is zeta_q. Then regress zeta_q on q through the
origin; that slope is H. The scale nu comes from the intercepts, with K_q
divided out. Running several q and checking that zeta_q is linear in q is
the real test -- if it is not, log-volatility is not self-similar and the
single number H is not describing anything.

**Forecasting.** Under the RFSV model, the conditional expectation of
future log-volatility has a closed form (their equation 5.1):

    E[log sigma_{t+Delta} | F_t]
        = (cos(H pi)/pi) Delta^{H+1/2}
          integral_{-inf}^{t} log sigma_s / ((t-s+Delta)(t-s)^{H+1/2}) ds

a weighted average of the entire past with a power-law kernel -- not an
exponential one, which is what separates this from a GARCH or EWMA
forecast. The kernel is exactly normalized: feeding it a constant path
returns that constant. Discretizing truncates the tail, so the weights
here are renormalized to sum to one, which restores that property.

The forecast is of *log* volatility. Exponentiating gives a median, not a
mean; the variance correction that converts one to the other is

    Var[log sigma_{t+Delta} | F_t] = c nu^2 Delta^{2H},
    c = Gamma(3/2 - H) / (Gamma(H + 1/2) Gamma(2 - 2H))

and is applied in `RoughVolEstimate.forecast_vol`.

**How this is used here.** Not as a signal. H is estimated on a trailing
window and compared to its own recent median; a fall in H means volatility
has become rougher than usual, which means volatility forecasts -- and
therefore the variance leg of any divergence against an options market --
have become less reliable. That is a reason to size down, not a reason to
trade, which is why `ensemble` consumes it as a multiplier.

**What you feed it matters more than the estimator.** H is estimated from
an observable proxy for latent volatility, and the choice of proxy biases
the answer in a direction that depends on the proxy, by more than the
estimator's own error:

* *Estimation noise biases H down.* Independent noise in each day's
  realized-volatility estimate is white, so it inflates short-lag
  increments relative to long-lag ones and the scaling regression reads a
  rougher path. On this repository's generator with a true H of 0.14,
  5-minute realized volatility reads about 0.11 and a noisy daily proxy
  built from a single absolute return reads about 0.01 -- the estimate
  becomes almost pure noise.
* *Rolling-window smoothing biases H up, much harder.* A trailing standard
  deviation of daily returns is a moving average, and it removes precisely
  the high-frequency variation the estimator measures. The same generator
  read through a 21-day trailing window gives H around 0.48, i.e. reports
  that volatility is an ordinary diffusion. This is the more dangerous
  error, because 21-day trailing volatility is the proxy closest to hand.

So this module wants an intraday realized measure -- the Oxford-Man
realized library, as Gatheral et al. use -- or the daily proxy from
`mock_data.fbm.fbm_vol_series`, and not a rolling window of daily returns.

Finally, a rolling 250-day estimate of a scaling exponent is noisy in its
own right: on data generated with a constant H the estimate moves by
several hundredths between windows, so small changes should not be
over-read.

References: Gatheral, Jaisson & Rosenbaum (2018), "Volatility is rough",
Quantitative Finance 18(6) 933-949; Bennedsen, Lunde & Pakkanen (2022),
Journal of Econometrics 226(2) 233-251 (on the measurement-error bias).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma as gamma_function

__all__ = [
    "HurstEstimate",
    "RoughVolConfig",
    "RoughVolEstimate",
    "estimate_hurst",
    "rfsv_forecast",
    "rolling_rough_vol",
]

_DEFAULT_MOMENTS = (0.5, 1.0, 1.5, 2.0, 3.0)
_DEFAULT_LAGS = (1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48)
_MIN_VOL = 1e-8


@dataclass(frozen=True)
class RoughVolConfig:
    """Windows and thresholds for the rough-volatility diagnostic.

    Attributes
    ----------
    window : trailing observations of realized volatility used for one
        Hurst estimate. 250 is about a year; much shorter and the scaling
        regression has too few increments at the longer lags.
    moments : the q values in m(q, Delta).
    lags : the Delta values, in days. They must stay well inside `window`,
        since m(q, Delta) at lag 48 on a 250-day window averages only about
        200 increments.
    baseline_window : trailing Hurst estimates used for the median that the
        current estimate is compared against.
    elevated_drop, unstable_drop : how far H must fall below that median
        before the diagnostic reports stress. The defaults are set against
        the estimator's own sampling noise, not against theory: rolled
        through a constant-H simulation, `hurst_drop` has a standard
        deviation near 0.012, so 0.03 and 0.06 are roughly 2.5 and 5 of
        those. Re-derive them on real data before relying on them; the
        dispersion of a rolling Hurst estimate depends on the window, the
        lag set and the volatility proxy.
    forecast_horizon : Delta, in days, for the RFSV forecast.
    """

    window: int = 250
    moments: tuple[float, ...] = _DEFAULT_MOMENTS
    lags: tuple[int, ...] = _DEFAULT_LAGS
    baseline_window: int = 120
    elevated_drop: float = 0.03
    unstable_drop: float = 0.06
    forecast_horizon: int = 21

    def __post_init__(self) -> None:
        if self.window < 60:
            raise ValueError(
                f"window must be at least 60 observations for the scaling regression to "
                f"have any power, got {self.window}"
            )
        if len(self.lags) < 3:
            raise ValueError(f"need at least 3 lags to regress on, got {len(self.lags)}")
        if max(self.lags) >= self.window // 3:
            raise ValueError(
                f"largest lag {max(self.lags)} is too close to window {self.window}; keep "
                f"lags below window/3 so each m(q, Delta) averages enough increments"
            )
        if any(q <= 0.0 for q in self.moments):
            raise ValueError(f"moments q must be positive, got {self.moments}")
        if not 0.0 < self.elevated_drop < self.unstable_drop:
            raise ValueError(
                f"require 0 < elevated_drop ({self.elevated_drop}) < unstable_drop "
                f"({self.unstable_drop}); the flags are ordered by severity"
            )
        if self.forecast_horizon < 1:
            raise ValueError(f"forecast_horizon must be at least 1 day, got {self.forecast_horizon}")


@dataclass(frozen=True)
class HurstEstimate:
    """Result of the scaling-law regression.

    Attributes
    ----------
    hurst : H, the slope of zeta_q against q through the origin.
    nu : volatility of log-volatility, per day^H.
    zeta : the per-q slopes, one per entry of `moments`.
    moments : the q values used.
    r_squared : R^2 of the zeta_q-on-q regression. This is the
        self-similarity check: values well below 1 mean zeta_q is not
        linear in q and a single H is not an adequate description.
    n_observations : log-volatility points the estimate was built from.
    """

    hurst: float
    nu: float
    zeta: NDArray[np.float64]
    moments: tuple[float, ...]
    r_squared: float
    n_observations: int


@dataclass(frozen=True)
class RoughVolEstimate:
    """A Hurst estimate plus the RFSV forecast built from it.

    Attributes
    ----------
    hurst_estimate : the underlying scaling regression.
    forecast_log_vol : E[log sigma_{t+Delta} | F_t].
    forecast_vol : exp(forecast_log_vol + variance/2), the conditional
        *mean* volatility rather than its median.
    forecast_variance : Var[log sigma_{t+Delta} | F_t] = c nu^2 Delta^{2H}.
    horizon_days : Delta.
    current_log_vol : the most recent observed log-volatility, so the
        forecast can be read as a move from where volatility is now.
    """

    hurst_estimate: HurstEstimate
    forecast_log_vol: float
    forecast_vol: float
    forecast_variance: float
    horizon_days: int
    current_log_vol: float

    @property
    def hurst(self) -> float:
        return self.hurst_estimate.hurst

    @property
    def nu(self) -> float:
        return self.hurst_estimate.nu


def _absolute_moment_of_normal(q: float) -> float:
    """E|N(0,1)|^q = 2^{q/2} Gamma((q+1)/2) / sqrt(pi)."""
    return float(2.0 ** (q / 2.0) * gamma_function((q + 1.0) / 2.0) / np.sqrt(np.pi))


def _ols(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Least-squares slope and intercept, guarding a degenerate x."""
    x_centered = x - x.mean()
    denominator = float(x_centered @ x_centered)
    if denominator <= 1e-15:
        raise ValueError("regressor has no variation; cannot fit a slope")
    slope = float(x_centered @ (y - y.mean()) / denominator)
    return slope, float(y.mean() - slope * x.mean())


def estimate_hurst(
    log_vol: ArrayLike,
    moments: tuple[float, ...] = _DEFAULT_MOMENTS,
    lags: tuple[int, ...] = _DEFAULT_LAGS,
) -> HurstEstimate:
    """Estimate H and nu from the scaling of log-volatility increments.

    Parameters
    ----------
    log_vol : series of log realized volatility, finite, no NaN.
    moments : q values for m(q, Delta).
    lags : Delta values in observation steps; all must be smaller than the
        series length.

    Raises
    ------
    ValueError : non-finite input, too few observations, a lag at or beyond
        the series length, or a q for which every increment is zero (a
        perfectly flat volatility path, where no scaling exponent exists).
    """
    series = np.asarray(log_vol, dtype=np.float64).ravel()
    if not np.all(np.isfinite(series)):
        raise ValueError("log_vol contains NaN or infinite values; drop them before estimating")
    if len(lags) < 3:
        raise ValueError(f"need at least 3 lags for the scaling regression, got {len(lags)}")
    if len(moments) < 2:
        raise ValueError(f"need at least 2 moments to test self-similarity, got {len(moments)}")
    if max(lags) >= len(series):
        raise ValueError(
            f"largest lag {max(lags)} exceeds the {len(series)}-observation series"
        )
    if len(series) < 3 * max(lags):
        raise ValueError(
            f"need at least {3 * max(lags)} observations for lags up to {max(lags)}, "
            f"got {len(series)}"
        )

    log_lags = np.log(np.asarray(lags, dtype=np.float64))
    zeta = np.empty(len(moments), dtype=np.float64)
    log_nu = np.empty(len(moments), dtype=np.float64)

    for index, q in enumerate(moments):
        log_m = np.empty(len(lags), dtype=np.float64)
        for position, lag in enumerate(lags):
            increments = np.abs(series[lag:] - series[:-lag])
            moment = float(np.mean(increments**q))
            if moment <= 0.0:
                raise ValueError(
                    f"m({q}, {lag}) is zero: the log-volatility path is exactly flat over "
                    "this window, so it has no scaling exponent"
                )
            log_m[position] = np.log(moment)
        slope, intercept = _ols(log_lags, log_m)
        zeta[index] = slope
        # intercept = q log nu + log K_q  =>  log nu = (intercept - log K_q)/q
        log_nu[index] = (intercept - np.log(_absolute_moment_of_normal(q))) / q

    # zeta_q = q H, forced through the origin: H = sum(q zeta_q)/sum(q^2).
    q_array = np.asarray(moments, dtype=np.float64)
    hurst = float(q_array @ zeta / (q_array @ q_array))
    residual = zeta - hurst * q_array
    total = zeta - zeta.mean()
    r_squared = float(1.0 - (residual @ residual) / max(float(total @ total), 1e-15))

    return HurstEstimate(
        hurst=hurst,
        nu=float(np.exp(np.mean(log_nu))),
        zeta=zeta,
        moments=tuple(float(q) for q in moments),
        r_squared=r_squared,
        n_observations=int(series.size),
    )


def _rfsv_weights(n_history: int, hurst: float, horizon: float) -> NDArray[np.float64]:
    """Discretized, renormalized kernel of the RFSV prediction formula.

    Weight on the observation u days back is proportional to
    1 / ((u + Delta) u^{H + 1/2}), evaluated at cell midpoints. The u -> 0
    singularity is integrable for H < 1/2 and is handled analytically over
    the first cell: integral_0^1 u^{-(H+1/2)} du = 1/(1/2 - H).

    Weights are returned most-recent-first and normalized to sum to one.
    The continuous kernel already integrates to exactly one against a
    constant path; truncating it at `n_history` days does not, so
    renormalizing restores the property that a flat volatility history
    forecasts itself. The cost is that the residual tail weight is
    redistributed over the observed history rather than dropped, which
    slightly over-weights the recent past.
    """
    if not 0.0 < hurst < 0.5:
        raise ValueError(
            f"the RFSV prediction kernel requires H in (0, 1/2) for the u -> 0 singularity "
            f"to be integrable; got H={hurst}. An estimate at or above 1/2 means the series "
            "is not rough and this forecast does not apply."
        )
    if n_history < 2:
        raise ValueError(f"need at least 2 history points, got {n_history}")

    exponent = hurst + 0.5
    weights = np.empty(n_history, dtype=np.float64)
    weights[0] = (1.0 / (0.5 + horizon)) * (1.0 / (0.5 - hurst))
    u = np.arange(1, n_history, dtype=np.float64) + 0.5
    weights[1:] = 1.0 / ((u + horizon) * u**exponent)
    return weights / weights.sum()


def rfsv_forecast(
    log_vol: ArrayLike, hurst: float, nu: float, horizon: int = 21
) -> tuple[float, float]:
    """Forecast log-volatility `horizon` days ahead, with its variance.

    Returns (expected log-volatility, conditional variance of log-volatility).
    The variance uses the RFSV constant
    c = Gamma(3/2 - H) / (Gamma(H + 1/2) Gamma(2 - 2H)), so that
    Var = c nu^2 Delta^{2H}.

    Raises ValueError for a non-rough H, a non-positive nu, a non-positive
    horizon, or non-finite history.
    """
    series = np.asarray(log_vol, dtype=np.float64).ravel()
    if not np.all(np.isfinite(series)):
        raise ValueError("log_vol contains NaN or infinite values")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 day, got {horizon}")
    if nu <= 0.0:
        raise ValueError(f"nu must be positive, got {nu}")

    weights = _rfsv_weights(series.size, hurst, float(horizon))
    expected = float(weights @ series[::-1])  # weights are most-recent-first
    constant = float(
        gamma_function(1.5 - hurst)
        / (gamma_function(hurst + 0.5) * gamma_function(2.0 - 2.0 * hurst))
    )
    return expected, float(constant * nu**2 * horizon ** (2.0 * hurst))


def rough_vol_estimate(
    realized_vol: ArrayLike, config: RoughVolConfig | None = None
) -> RoughVolEstimate:
    """Estimate H and nu on a realized-volatility window, then forecast.

    `realized_vol` is in volatility units (0.20 = 20% annualized), not
    variance and not logs; the log is taken here. Non-positive entries are
    rejected rather than floored, since a zero realized volatility is a
    data error and flooring it would silently distort the increments the
    whole estimate rests on.
    """
    cfg = config or RoughVolConfig()
    vol = np.asarray(realized_vol, dtype=np.float64).ravel()
    if not np.all(np.isfinite(vol)):
        raise ValueError("realized_vol contains NaN or infinite values; drop them first")
    if np.any(vol <= _MIN_VOL):
        raise ValueError(
            f"realized_vol must be strictly positive (minimum observed {vol.min():.3e}); "
            "a zero or negative volatility is a data error, not a low-volatility day"
        )

    log_vol = np.log(vol)
    estimate = estimate_hurst(log_vol, cfg.moments, cfg.lags)
    # The prediction kernel is only defined for a rough path. When the
    # estimate lands at or above 1/2 -- which happens on short or noisy
    # windows -- fall back to the current level rather than inventing one.
    if 0.0 < estimate.hurst < 0.5:
        forecast_log, forecast_variance = rfsv_forecast(
            log_vol, estimate.hurst, estimate.nu, cfg.forecast_horizon
        )
    else:
        forecast_log, forecast_variance = float(log_vol[-1]), 0.0

    return RoughVolEstimate(
        hurst_estimate=estimate,
        forecast_log_vol=forecast_log,
        forecast_vol=float(np.exp(forecast_log + 0.5 * forecast_variance)),
        forecast_variance=forecast_variance,
        horizon_days=cfg.forecast_horizon,
        current_log_vol=float(log_vol[-1]),
    )


def rolling_rough_vol(
    realized_vol: pd.Series, config: RoughVolConfig | None = None
) -> pd.DataFrame:
    """Roll `rough_vol_estimate` through a realized-volatility series.

    Returns a DataFrame indexed like `realized_vol` with columns:

        hurst              the trailing estimate
        nu                 volatility of log-volatility
        hurst_r2           self-similarity check, see `HurstEstimate`
        rv_forecast        RFSV forecast of volatility, `horizon` days out
        rv_forecast_ratio  forecast divided by current realized volatility
        hurst_baseline     rolling median of past Hurst estimates
        hurst_drop         baseline minus current H; positive means the
                           path has turned rougher than its own recent norm

    Rows before the first full window, and rows where the window contains a
    non-positive or missing volatility, are NaN. A window that raises inside
    the estimator (a flat path, for example) is also recorded as NaN rather
    than aborting the whole series -- one unusable window should not cost
    the other several hundred.
    """
    cfg = config or RoughVolConfig()
    if not isinstance(realized_vol, pd.Series):
        raise TypeError(f"realized_vol must be a Series, got {type(realized_vol).__name__}")
    if len(realized_vol) < cfg.window:
        raise ValueError(
            f"need at least window={cfg.window} observations, got {len(realized_vol)}"
        )

    values = realized_vol.to_numpy(dtype=np.float64)
    records: list[dict[str, float]] = []
    blank = {"hurst": np.nan, "nu": np.nan, "hurst_r2": np.nan, "rv_forecast": np.nan}

    for end in range(len(values)):
        start = end - cfg.window + 1
        window = values[start : end + 1] if start >= 0 else None
        if window is None or not np.all(np.isfinite(window)) or np.any(window <= _MIN_VOL):
            records.append(dict(blank))
            continue
        try:
            estimate = rough_vol_estimate(window, cfg)
        except ValueError:
            records.append(dict(blank))
            continue
        records.append(
            {
                "hurst": estimate.hurst,
                "nu": estimate.nu,
                "hurst_r2": estimate.hurst_estimate.r_squared,
                "rv_forecast": estimate.forecast_vol,
            }
        )

    frame = pd.DataFrame(records, index=realized_vol.index)
    frame["rv_forecast_ratio"] = frame["rv_forecast"] / realized_vol.where(realized_vol > _MIN_VOL)
    # shift(1) so the baseline is built only from estimates that predate the
    # one being compared against it.
    frame["hurst_baseline"] = (
        frame["hurst"].shift(1).rolling(cfg.baseline_window, min_periods=cfg.baseline_window // 3).median()
    )
    frame["hurst_drop"] = frame["hurst_baseline"] - frame["hurst"]
    return frame
