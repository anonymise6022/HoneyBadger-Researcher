"""Breeden-Litzenberger extraction of the risk-neutral density.

For a European call,  C(K) = exp(-rT) * integral_K^inf (S - K) q(S) dS.
Differentiating under the integral once gives

    dC/dK = -exp(-rT) * (1 - Q(K)),

and again gives the Breeden-Litzenberger identity (1978):

    d2C/dK2 = exp(-rT) q(K)      =>      q(K) = exp(rT) * d2C/dK2

So the density is the *curvature* of the call curve in strike. That is a
second derivative of noisy market data: quote noise eps on a grid of
spacing h is amplified by eps/h^2. Feeding in SVI-smoothed prices rather
than raw quotes is the standard remedy.

References: Breeden & Litzenberger (1978), J. Business 51(4) 621-651;
Figlewski (2010), "Estimating the Implied Risk-Neutral Density for the U.S.
Market Portfolio".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ImpliedDistribution", "extract_density"]

_MIN_MASS = 1e-8  # below this, the integral is rounding noise, not a density


@dataclass(frozen=True)
class ImpliedDistribution:
    """Risk-neutral density on a strike grid plus its first three moments.

    `total_mass` is the raw integral before normalization; far from 1.0 it
    signals a truncated or noisy grid.
    """

    strikes: NDArray[np.float64]
    density: NDArray[np.float64]
    mean: float
    variance: float
    skew: float
    total_mass: float

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


def extract_density(
    K: ArrayLike, C: ArrayLike, T: float, r: float, normalize: bool = True
) -> ImpliedDistribution:
    """Recover q(K) = exp(rT) * d2C/dK2 by finite differences.

    Uses the three-point central second difference generalized to a
    non-uniform grid, with h- = K_i - K_{i-1} and h+ = K_{i+1} - K_i:

        C''(K_i) ~= 2[h- C_{i+1} - (h- + h+) C_i + h+ C_{i-1}]
                    / (h- h+ (h- + h+))

    which collapses to (C_{i+1} - 2C_i + C_{i-1})/h^2 when h- = h+ = h.
    Endpoints are dropped since a central difference needs both neighbours.

    Negatives are clipped to zero: a density cannot be negative, and small
    negatives here are finite-difference noise or a static-arbitrage
    violation (non-convex call curve), not signal.

    Moments use trapezoidal quadrature against the normalized density:
    mean = int K q dK, var = int (K-mean)^2 q dK, skew = int (K-mean)^3 q dK
    / var^1.5. These are moments of the *truncated* distribution — strikes
    outside the quoted range contribute nothing — so compare `mean` against
    the forward S*exp(rT); a large gap means the grid is too narrow.

    Parameters
    ----------
    K : strictly increasing strikes, >= 3 points, need not be uniform.
    C : call prices at each strike, len == len(K).
    T : years to expiry, positive.
    r : continuously compounded rate.
    normalize : rescale the density to unit mass before taking moments.

    Raises
    ------
    ValueError : mismatched lengths, fewer than 3 strikes, unsorted grid,
        non-finite inputs, T <= 0, or a call curve with no convexity.
    """
    K_arr = np.asarray(K, dtype=np.float64).ravel()
    C_arr = np.asarray(C, dtype=np.float64).ravel()

    if K_arr.shape != C_arr.shape:
        raise ValueError(f"K and C must be the same length; got {K_arr.size} and {C_arr.size}")
    if K_arr.size < 3:
        raise ValueError(f"need >= 3 strikes for a central second difference, got {K_arr.size}")
    if not (np.all(np.isfinite(K_arr)) and np.all(np.isfinite(C_arr))):
        raise ValueError("K and C must be finite; drop NaN quotes before extraction")
    spacings = np.diff(K_arr)
    if np.any(spacings <= 0.0):
        raise ValueError("K must be strictly increasing; sort the chain by strike first")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if not np.isfinite(r):
        raise ValueError(f"r must be finite, got {r}")

    h_minus, h_plus = spacings[:-1], spacings[1:]
    second_derivative = (
        2.0
        * (h_minus * C_arr[2:] - (h_minus + h_plus) * C_arr[1:-1] + h_plus * C_arr[:-2])
        / (h_minus * h_plus * (h_minus + h_plus))
    )
    strikes = K_arr[1:-1]
    density = np.clip(np.exp(r * T) * second_derivative, 0.0, None)

    total_mass = float(np.trapezoid(density, strikes))
    # A genuine density integrates to ~1, and even a badly truncated grid
    # retains an appreciable fraction of that. Anything below _MIN_MASS is
    # rounding noise off a curve with no real convexity (e.g. an exactly
    # linear one), which would otherwise yield confident-looking nonsense
    # moments after normalization.
    if total_mass <= _MIN_MASS:
        raise ValueError(
            f"recovered density has negligible mass ({total_mass:.3e}); the call curve "
            "has no convexity in strike, so no distribution can be implied from it"
        )

    pdf = density / total_mass if normalize else density
    mean = float(np.trapezoid(strikes * pdf, strikes))
    centered = strikes - mean
    variance = float(np.trapezoid(centered**2 * pdf, strikes))
    if variance <= 0.0:
        raise ValueError(
            f"implied variance is non-positive ({variance:.3e}); the density collapsed "
            "onto a single grid point"
        )

    return ImpliedDistribution(
        strikes=strikes,
        density=density,
        mean=mean,
        variance=variance,
        skew=float(np.trapezoid(centered**3 * pdf, strikes) / variance**1.5),
        total_mass=total_mass,
    )
