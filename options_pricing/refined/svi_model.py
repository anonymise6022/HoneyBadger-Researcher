"""Raw SVI parametrization of one expiry's implied volatility smile.

Total implied variance w(k) = sigma_BS(k)^2 * T as a function of
log-moneyness k = ln(K/F), F = forward (Gatheral 2006, ch. 3):

    w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

A hyperbola: linear asymptotes of slope b(rho +/- 1) joined by a smooth
vertex of width sigma. Parameters: a = variance level, b = wing angle,
rho = skew (rho < 0 lifts the downside wing, the usual equity shape),
m = vertex location, sigma = vertex curvature scale.

No-arbitrage constraints enforced by the fit (Gatheral & Jacquier 2014, §2):

    1. b >= 0        wings open upward; b < 0 sends variance to -inf.
    2. |rho| < 1     rho is correlation-like; at |rho| = 1 one asymptote
                     goes flat and the slice degenerates.
    3. sigma > 0     sigma = 0 makes w non-differentiable at k = m.
    4. a + b*sigma*sqrt(1 - rho^2) >= 0
                     This expression is exactly min_k w(k), attained at
                     k = m - rho*sigma/sqrt(1-rho^2). Total variance is a
                     variance, so it cannot go negative anywhere.

These are necessary, not sufficient, for absence of butterfly arbitrage.
The sufficient condition also caps the wing slope at b(1+|rho|) <= 4/T
(Lee 2004); exposed as the optional `max_wing_slope_T` rather than always
imposed, since it needs T and over-constrains short-dated fits.

References: Gatheral (2006), *The Volatility Surface*, ch. 3; Gatheral &
Jacquier (2014), Quantitative Finance 14(1) 59-71; Lee (2004), Math.
Finance 14(3) 469-480.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

__all__ = ["SVIParams", "SVIFit", "svi_total_variance", "fit_svi"]

_RHO_BOUND = 0.999  # keep |rho| strictly inside (-1, 1) for the optimizer
_MIN_POSITIVE = 1e-8


@dataclass(frozen=True)
class SVIParams:
    """The five raw-SVI parameters (a, b, rho, m, sigma)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma], dtype=np.float64)

    def min_total_variance(self) -> float:
        """min_k w(k) = a + b*sigma*sqrt(1 - rho^2).

        From w'(k) = 0, whose root is k* = m - rho*sigma/sqrt(1-rho^2).
        """
        return float(self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2))

    def wing_slopes(self) -> tuple[float, float]:
        """Asymptotic slopes (left, right): w(k) ~ b(rho +/- 1)k as k -> +/-inf."""
        return (-self.b * (1.0 - self.rho), self.b * (1.0 + self.rho))


@dataclass(frozen=True)
class SVIFit:
    """Calibration result: parameters, residuals (w - w_model), diagnostics."""

    params: SVIParams
    residuals: NDArray[np.float64]
    sse: float
    rmse: float
    n_points: int
    n_iterations: int


def _unpack(params: SVIParams | Sequence[float]) -> tuple[float, float, float, float, float]:
    if isinstance(params, SVIParams):
        return (params.a, params.b, params.rho, params.m, params.sigma)
    values = tuple(float(v) for v in params)
    if len(values) != 5:
        raise ValueError(f"expected 5 parameters (a, b, rho, m, sigma), got {len(values)}")
    return values  # type: ignore[return-value]


def svi_total_variance(
    k: ArrayLike, params: SVIParams | Sequence[float]
) -> NDArray[np.float64]:
    """Evaluate w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + sigma^2)).

    k is log-moneyness ln(K/F); params is an SVIParams or a 5-sequence
    ordered (a, b, rho, m, sigma). Returns total variance sigma_BS^2 * T.
    """
    a, b, rho, m, sigma = _unpack(params)
    centered = np.asarray(k, dtype=np.float64) - m
    # hypot avoids overflow in the far wings where (k-m)^2 would blow up.
    return a + b * (rho * centered + np.hypot(centered, sigma))


def _initial_guess(k: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    """Data-driven start: a below the variance floor, m at the observed min.

    The SVI objective is non-convex, so landing in the right basin matters
    far more than the precision of the starting point.
    """
    return np.array(
        [
            max(0.5 * float(np.min(w)), _MIN_POSITIVE),
            0.1,
            0.0,
            float(k[np.argmin(w)]),
            0.25 * (float(np.ptp(k)) or 1.0),
        ]
    )


def fit_svi(
    k: ArrayLike,
    w: ArrayLike,
    initial_guess: Sequence[float] | None = None,
    max_wing_slope_T: float | None = None,
    max_iterations: int = 500,
) -> SVIFit:
    """Calibrate raw SVI by constrained least squares:

        min over (a,b,rho,m,sigma) of  sum_i [w_model(k_i) - w_i]^2

    subject to the four no-arbitrage constraints in the module docstring.
    SLSQP is used because the problem mixes box bounds with the nonlinear
    inequality a + b*sigma*sqrt(1-rho^2) >= 0. Fitting in total variance
    rather than vol weights the smile the way option prices do and keeps
    the objective polynomial in the parameters.

    Parameters
    ----------
    k : log-moneyness ln(K/F); at least 5 points (one per free parameter).
    w : observed total variance iv^2 * T, strictly positive, len == len(k).
    initial_guess : optional (a, b, rho, m, sigma) start point.
    max_wing_slope_T : if given, the expiry T in years, adding Lee's
        sufficient bound b(1 + |rho|) <= 4/T.
    max_iterations : SLSQP iteration cap.

    Raises
    ------
    ValueError : mismatched, too short, non-finite, or non-positive inputs.
    RuntimeError : optimizer failed to converge, or converged to a slice
        violating the variance floor beyond numerical tolerance.
    """
    k_arr = np.asarray(k, dtype=np.float64).ravel()
    w_arr = np.asarray(w, dtype=np.float64).ravel()

    if k_arr.shape != w_arr.shape:
        raise ValueError(f"k and w must be the same length; got {k_arr.size} and {w_arr.size}")
    if k_arr.size < 5:
        raise ValueError(f"SVI has 5 free parameters; need at least 5 points, got {k_arr.size}")
    if not (np.all(np.isfinite(k_arr)) and np.all(np.isfinite(w_arr))):
        raise ValueError("k and w must be finite; drop NaN quotes before fitting")
    if np.any(w_arr <= 0.0):
        raise ValueError(
            f"total variance w must be strictly positive (min: {np.min(w_arr)}); "
            "check that w = iv**2 * T"
        )

    x0 = (
        _initial_guess(k_arr, w_arr)
        if initial_guess is None
        else np.asarray(_unpack(initial_guess), dtype=np.float64)
    )
    # Box bounds cover constraints 1-3; the variance floor (4) is nonlinear.
    bounds = [(0.0, None), (0.0, None), (-_RHO_BOUND, _RHO_BOUND), (None, None), (_MIN_POSITIVE, None)]
    constraints: list[dict] = [
        {"type": "ineq", "fun": lambda x: x[0] + x[1] * x[4] * np.sqrt(1.0 - x[2] ** 2)}
    ]
    if max_wing_slope_T is not None:
        if max_wing_slope_T <= 0.0:
            raise ValueError(f"max_wing_slope_T must be positive, got {max_wing_slope_T}")
        lim = 4.0 / max_wing_slope_T
        constraints.append({"type": "ineq", "fun": lambda x, L=lim: L - x[1] * (1.0 + abs(x[2]))})

    result = minimize(
        lambda x: float(np.sum((svi_total_variance(k_arr, x) - w_arr) ** 2)),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iterations, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(
            f"SVI calibration did not converge after {result.nit} iterations: "
            f"{result.message}. Try another initial_guess or screen the smile for bad quotes."
        )

    params = SVIParams(*(float(v) for v in result.x))
    if params.min_total_variance() < -1e-10:
        raise RuntimeError(
            f"SVI converged to an arbitrageable slice: min total variance "
            f"{params.min_total_variance():.3e} < 0; the smile likely holds inconsistent quotes."
        )

    residuals = w_arr - svi_total_variance(k_arr, params)
    sse = float(np.sum(residuals**2))
    return SVIFit(
        params=params,
        residuals=residuals,
        sse=sse,
        rmse=float(np.sqrt(sse / residuals.size)),
        n_points=int(residuals.size),
        n_iterations=int(result.nit),
    )
