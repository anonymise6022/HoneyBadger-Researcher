"""
SVI (Stochastic Volatility Inspired) raw parametrization, fit to a single
expiry's implied total-variance smile.

Gatheral's raw SVI formula, total variance w(k) as a function of
log-moneyness k = ln(K/F) (F = forward price):

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

where w(k) = sigma_BS(k)^2 * T (total implied variance, not annualized vol).

Parameters: theta = [a, b, rho, m, sigma]
    a     : overall level of variance (vertical shift)
    b     : angle between the left/right wings (controls slope magnitude)
    rho   : rotation / skew of the smile, counter-clockwise for rho<0
    m     : horizontal shift of the smile (location of the "kink")
    sigma : curvature at the vertex k = m (ATM smoothness)

No-static-arbitrage constraints on a single slice (necessary conditions,
Gatheral & Jacquier 2014):
    b >= 0                          -- wings must open upward, not downward
    0 <= |rho| < 1                  -- rho is a correlation-like parameter
    sigma > 0                       -- strictly positive curvature scale
    a + b*sigma*sqrt(1 - rho^2) >= 0  -- minimum of w(k) over all k must be
                                         non-negative (total variance can't
                                         be negative anywhere on the curve)
These are necessary (not sufficient) for absence of butterfly arbitrage on
the slice; they are enforced here as fit constraints, not the full
Gatheral-Jacquier sufficient condition (which also bounds b*(1+|rho|) <= 4/T).
"""

import numpy as np
from scipy.optimize import minimize


def svi_total_variance(k, params):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _min_w_constraint(params):
    a, b, rho, m, sigma = params
    return a + b * sigma * np.sqrt(1.0 - rho ** 2)


def _sse(params, k, w):
    model = svi_total_variance(k, params)
    return np.sum((model - w) ** 2)


def fit_svi(k, w, initial_guess=None):
    """
    Constrained least-squares fit of raw SVI to observed (k, w) points.

    k : array of log-moneyness ln(K/F)
    w : array of observed total variance (implied_vol^2 * T)

    Returns dict with fitted params, residuals, and SSE.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)

    if initial_guess is None:
        a0 = max(np.min(w) * 0.5, 1e-4)
        b0 = 0.1
        rho0 = 0.0
        m0 = k[np.argmin(np.abs(k))]
        sigma0 = 0.1
        initial_guess = [a0, b0, rho0, m0, sigma0]

    # bounds: a>=0, b>=0, -1<rho<1, m free, sigma>0
    bounds = [
        (0.0, None),
        (1e-6, None),
        (-0.999, 0.999),
        (None, None),
        (1e-6, None),
    ]

    constraints = [
        {"type": "ineq", "fun": _min_w_constraint},  # a + b*sigma*sqrt(1-rho^2) >= 0
    ]

    result = minimize(
        _sse,
        initial_guess,
        args=(k, w),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    fitted_params = result.x
    residuals = w - svi_total_variance(k, fitted_params)

    return {
        "a": fitted_params[0],
        "b": fitted_params[1],
        "rho": fitted_params[2],
        "m": fitted_params[3],
        "sigma": fitted_params[4],
        "residuals": residuals,
        "sse": float(np.sum(residuals ** 2)),
        "success": result.success,
    }


if __name__ == "__main__":
    # synthetic smile: true params, plus small noise
    true_params = [0.04, 0.4, -0.3, 0.0, 0.1]
    k = np.linspace(-0.5, 0.5, 25)
    w_true = svi_total_variance(k, true_params)
    rng = np.random.default_rng(0)
    w_obs = w_true + rng.normal(0, 0.001, size=k.shape)

    fit = fit_svi(k, w_obs)
    print("fitted params:", {key: fit[key] for key in ("a", "b", "rho", "m", "sigma")})
    print("SSE:", fit["sse"])
