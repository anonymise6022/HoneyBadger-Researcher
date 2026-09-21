"""Exact fractional Brownian motion and fractional-OU log-volatility paths.

Fractional Gaussian noise with Hurst parameter H is the stationary
increment process with autocovariance

    gamma(k) = 1/2 (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H})

H = 1/2 gives white noise (gamma(k) = 0 for k != 0); H < 1/2 gives negative
one-lag correlation and *rougher* paths than Brownian motion; H > 1/2 gives
long-memory, persistent paths. Gatheral, Jaisson & Rosenbaum (2018) find
H ~ 0.1 for log-volatility across essentially every liquid asset, which is
why `fbm_vol_series` defaults there.

Sampling uses the Davies-Harte circulant-embedding method (Davies & Harte
1987): embed the n x n covariance Toeplitz matrix in a 2n x 2n circulant
one, whose eigenvalues are the FFT of its first row, and synthesize the
path in the Fourier domain. It is *exact* -- the sample has the target
covariance to machine precision, not asymptotically -- and costs
O(n log n). The embedding's eigenvalues are provably non-negative for
fractional Gaussian noise on this grid, so a negative eigenvalue means a
numerical problem and is raised rather than clipped.

References: Davies & Harte (1987), Biometrika 74(1) 95-101; Dieker (2004),
*Simulation of Fractional Brownian Motion*, ch. 2; Gatheral, Jaisson &
Rosenbaum (2018), Quantitative Finance 18(6) 933-949.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "fbm_vol_series",
    "fractional_brownian_motion",
    "fractional_gaussian_noise",
]

_EIGENVALUE_TOLERANCE = -1e-10  # allow round-off, reject genuine negatives


def _check_hurst(hurst: float) -> float:
    if not 0.0 < hurst < 1.0:
        raise ValueError(
            f"hurst must lie strictly in (0, 1); H <= 0 and H >= 1 are not valid "
            f"fractional Brownian motions, got {hurst}"
        )
    return float(hurst)


def fractional_gaussian_noise(
    n: int, hurst: float = 0.14, seed: int | None = None
) -> NDArray[np.float64]:
    """Draw n increments of unit-variance fractional Gaussian noise.

    Exact in distribution: Cov(X_i, X_j) = gamma(|i-j|) to machine
    precision, and Var(X_i) = 1.

    Parameters
    ----------
    n : number of increments, >= 2.
    hurst : Hurst parameter in (0, 1).
    seed : seed for `numpy.random.default_rng`.

    Raises
    ------
    ValueError : n < 2, hurst outside (0, 1), or -- indicating a numerical
        failure rather than a modelling one -- a negative circulant
        eigenvalue.
    """
    if n < 2:
        raise ValueError(f"need at least 2 increments to have any covariance structure, got {n}")
    h = _check_hurst(hurst)
    rng = np.random.default_rng(seed)

    k = np.arange(n + 1, dtype=np.float64)
    two_h = 2.0 * h
    gamma = 0.5 * (np.abs(k - 1.0) ** two_h - 2.0 * k**two_h + (k + 1.0) ** two_h)

    # First row of the circulant embedding: gamma(0..n) then gamma(n-1..1).
    first_row = np.concatenate([gamma, gamma[-2:0:-1]])
    m = first_row.size  # == 2n
    eigenvalues = np.fft.fft(first_row).real
    if eigenvalues.min() < _EIGENVALUE_TOLERANCE:
        raise ValueError(
            f"circulant embedding produced a negative eigenvalue "
            f"({eigenvalues.min():.3e}) for H={h}, n={n}; this indicates loss of "
            "precision rather than an invalid Hurst parameter"
        )
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # Build a Hermitian spectral vector so the inverse transform is real.
    # Scaling: the zero and Nyquist bins carry one real degree of freedom
    # each (variance lam/m); interior bins carry two (variance lam/(2m) per
    # component) and are mirrored into the upper half.
    half = m // 2
    spectrum = np.zeros(m, dtype=np.complex128)
    normal = rng.standard_normal(m)
    spectrum[0] = np.sqrt(eigenvalues[0] / m) * normal[0]
    spectrum[half] = np.sqrt(eigenvalues[half] / m) * normal[half]
    interior = np.arange(1, half)
    scale = np.sqrt(eigenvalues[interior] / (2.0 * m))
    spectrum[interior] = scale * (normal[interior] + 1j * normal[half + 1 :][::-1])
    spectrum[half + 1 :] = np.conj(spectrum[1:half][::-1])

    return np.fft.fft(spectrum).real[:n]


def fractional_brownian_motion(
    n: int, hurst: float = 0.14, length: float = 1.0, seed: int | None = None
) -> NDArray[np.float64]:
    """Sample fBm on [0, length] at n+1 equally spaced points, starting at 0.

    Self-similarity B_H(at) =d= a^H B_H(t) fixes the increment scale: on a
    grid of step dt = length/n each increment carries standard deviation
    dt^H, so Var(B_H(length)) = length^{2H}.
    """
    increments = fractional_gaussian_noise(n, hurst, seed)
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    dt = length / n
    return np.concatenate([[0.0], np.cumsum(increments) * dt**hurst])


def fbm_vol_series(
    n_days: int = 1500,
    hurst: float = 0.14,
    nu: float = 0.30,
    long_run_vol: float = 0.20,
    mean_reversion: float = 0.02,
    start: str = "2019-01-02",
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate an RFSV-style log-volatility path and returns driven by it.

    The rough fractional stochastic volatility model of Gatheral, Jaisson &
    Rosenbaum (2018) takes log-volatility to be fBm with small H. Raw fBm
    is non-stationary, so -- as they do for simulation over long horizons --
    this uses the fractional Ornstein-Uhlenbeck form

        d log sigma_t = -kappa (log sigma_t - log sigma_bar) dt + nu dB^H_t

    discretized daily. With kappa small (default 0.02/day, a half-life of
    about 35 days) the path is statistically indistinguishable from fBm at
    the horizons the Hurst estimator looks at, which is the point: the
    estimator in `regime_detection.rough_vol` should recover `hurst` from
    this series. Returns are r_t = sigma_t/sqrt(252) * Z_t, with Z
    correlated to the volatility innovation at `leverage` to produce the
    negative return/vol correlation real equity data shows.

    Returns a DataFrame indexed by business day with columns:

        log_vol          the latent driver, log of annualized volatility
        vol              exp(log_vol)
        ret              daily simple return
        realized_vol     a daily realized-volatility estimate of the kind
                         built from 5-minute intraday returns
        realized_vol_21d 21-day trailing standard deviation of `ret`

    The two volatility observables behave very differently under a Hurst
    estimator, and the difference is not a detail. `realized_vol` carries
    independent estimation noise, log-scale standard deviation
    sqrt(1/(2M)) for M intraday observations, which biases H *downward* --
    noise is rougher than the signal. `realized_vol_21d` is a moving
    average, which biases H sharply *upward* by smoothing away exactly the
    high-frequency roughness the estimator measures: on this generator,
    with a true H of 0.14, the 21-day series reads about 0.48, which is not
    a rough path at all. `regime_detection.rough_vol` should therefore be
    fed `realized_vol`, and real deployments should feed it an intraday
    realized measure (the Oxford-Man library, as Gatheral et al. use), not
    a rolling window of daily returns.
    """
    if n_days < 2:
        raise ValueError(f"n_days must be at least 2, got {n_days}")
    if nu <= 0.0:
        raise ValueError(f"nu (vol-of-log-vol) must be positive, got {nu}")
    if long_run_vol <= 0.0:
        raise ValueError(f"long_run_vol must be positive, got {long_run_vol}")
    if not 0.0 <= mean_reversion < 1.0:
        raise ValueError(
            f"mean_reversion is a per-day rate and must lie in [0, 1), got {mean_reversion}"
        )
    h = _check_hurst(hurst)

    rng = np.random.default_rng(seed)
    # The fBm driver is sampled on the unit interval and rescaled so its
    # per-day increment standard deviation is exactly nu, independent of n.
    driver = np.diff(fractional_brownian_motion(n_days, h, length=1.0, seed=seed))
    driver = nu * driver / (1.0 / n_days) ** h

    log_bar = float(np.log(long_run_vol))
    log_vol = np.empty(n_days, dtype=np.float64)
    log_vol[0] = log_bar
    for t in range(1, n_days):
        log_vol[t] = log_vol[t - 1] - mean_reversion * (log_vol[t - 1] - log_bar) + driver[t]

    vol = np.exp(log_vol)
    leverage = -0.4
    shock = rng.standard_normal(n_days)
    innovation = driver / max(float(np.std(driver)), 1e-12)
    shock = leverage * innovation + np.sqrt(1.0 - leverage**2) * shock
    returns = vol / np.sqrt(252.0) * shock

    index = pd.bdate_range(start=start, periods=n_days, name="date")
    frame = pd.DataFrame({"log_vol": log_vol, "vol": vol, "ret": returns}, index=index)
    # A realized-variance estimator built from M intraday returns has
    # asymptotic variance 2/M in logs, so the volatility estimate carries
    # log-scale noise of sqrt(1/(2M)). M = 78 is 5-minute sampling over a
    # 6.5-hour session, the standard choice for realized-volatility work.
    intraday_observations = 78
    measurement_noise = rng.standard_normal(n_days) * np.sqrt(0.5 / intraday_observations)
    frame["realized_vol"] = np.exp(log_vol + measurement_noise)
    frame["realized_vol_21d"] = (
        frame["ret"].rolling(21, min_periods=10).std() * np.sqrt(252.0)
    )
    return frame
