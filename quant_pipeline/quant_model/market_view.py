"""Compose chain -> smile -> density -> return moments into one market view.

The three steps each exist for a reason:

1. **Fit SVI to the smile.** The Breeden-Litzenberger identity needs a
   second derivative of the call curve in strike. Differencing raw quotes
   twice amplifies quote noise by 1/h^2 and produces negative densities.
   Fitting an arbitrage-constrained smile first and re-pricing from it is
   the standard remedy (Figlewski 2010).

2. **Re-price on a dense, wider grid.** The density is only recovered where
   there are strikes, and the moments are then moments of the *truncated*
   distribution. Evaluating the fitted smile outside the quoted range
   recovers the tails -- at the cost of trusting SVI's linear wings, which
   are a modelling assumption and not data. `MarketView.wing_extrapolation`
   records how far past the quotes the grid reaches so this is visible
   rather than silent.

3. **Change variables from strike to return.** The macro forecast predicts
   the distribution of R = S_T/S_0 - 1; the density is over S_T. The map
   is affine, so the moments transform exactly:

       E[R]    = E[S_T]/S_0 - 1
       Var[R]  = Var[S_T]/S_0^2
       Skew[R] = Skew[S_T]                (scale- and shift-invariant)

   No numerical integration is repeated, and no approximation enters.

A standing caveat the divergence layer must not forget: this density is
risk-neutral, the macro forecast is real-world. Their means differ by the
variance risk premium and the equity risk premium even when nobody is
wrong. See `ensemble.divergence_signal` for how that is handled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .black_scholes import call_price
from .implied_density import ImpliedDistribution, extract_density
from .svi_model import SVIFit, SVIParams, fit_svi, svi_total_variance

__all__ = ["MarketView", "implied_return_moments", "market_view_from_chain"]

_MIN_QUOTES = 5  # SVI has five free parameters
_DEFAULT_GRID_POINTS = 401


@dataclass(frozen=True)
class MarketView:
    """The option market's terminal distribution, in strike and return space.

    Attributes
    ----------
    spot, expiry_years, rate : the inputs the view was built at.
    forward : S exp(rT), the risk-neutral mean of S_T. The recovered
        `distribution.mean` should match it; `forward_error` reports the gap.
    svi_fit : the calibrated smile, including RMSE in total-variance units.
    distribution : the Breeden-Litzenberger density over strikes.
    mean_return, variance_return, skew_return : moments of R = S_T/S - 1.
    implied_vol_atm : sqrt(w(0)/T) from the fitted smile, the at-the-money
        forward volatility.
    forward_error : (distribution.mean - forward) / forward. Large values
        mean the grid is too narrow or the smile fit is poor.
    wing_extrapolation : how far outside the quoted log-moneyness range the
        pricing grid was extended, as a fraction of the quoted range.
    n_quotes : usable quotes the smile was fitted to.
    """

    spot: float
    expiry_years: float
    rate: float
    forward: float
    svi_fit: SVIFit
    distribution: ImpliedDistribution
    mean_return: float
    variance_return: float
    skew_return: float
    implied_vol_atm: float
    forward_error: float
    wing_extrapolation: float
    n_quotes: int

    @property
    def std_return(self) -> float:
        """Standard deviation of R over the option's life."""
        return float(np.sqrt(self.variance_return))

    @property
    def annualized_vol(self) -> float:
        """Return standard deviation rescaled to one year by sqrt(T)."""
        return float(self.std_return / np.sqrt(self.expiry_years))


def implied_return_moments(
    distribution: ImpliedDistribution, spot: float
) -> tuple[float, float, float]:
    """Map (mean, variance, skew) of S_T to those of R = S_T/spot - 1.

    Exact: the transformation is affine, so variance scales by 1/spot^2 and
    skew -- being standardized -- is unchanged.

    Raises ValueError if spot is not strictly positive.
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be strictly positive, got {spot}")
    return (
        float(distribution.mean / spot - 1.0),
        float(distribution.variance / spot**2),
        float(distribution.skew),
    )


def _initial_guesses(
    k: NDArray[np.float64], w: NDArray[np.float64]
) -> list[SVIParams | None]:
    """Candidate SVI starting points, best-guess first.

    `fit_svi`'s own default start is deliberately generic, and the SVI
    objective is not convex: on a one-month smile, whose total variance is
    of order 0.003, a start with b = 0.1 and sigma = 0.15 sits far enough
    from the optimum that SLSQP stalls in a local minimum several percent
    off the at-the-money level -- measurably, even on a chain generated
    from an exact SVI slice. Since that error would flow straight into the
    divergence signal as fake disagreement, the fit is multi-started here.

    The first candidate is quasi-explicit, read off the data. Matching the
    asymptotic wing slopes w'(k) -> b(rho +/- 1) against the observed outer
    slopes gives

        b   = (slope_right - slope_left) / 2
        rho = (slope_right + slope_left) / (slope_right - slope_left)

    with m placed at the observed minimum and a set from the variance
    floor a = w_min - b*sigma*sqrt(1 - rho^2). The remaining candidates
    vary the vertex width sigma, which is the parameter the slope identity
    says nothing about, plus `None` for the library default.
    """
    span = float(k[-1] - k[0])
    half = max(1, len(k) // 3)
    slope_left = float((w[half] - w[0]) / max(k[half] - k[0], 1e-12))
    slope_right = float((w[-1] - w[-1 - half]) / max(k[-1] - k[-1 - half], 1e-12))

    b = max((slope_right - slope_left) / 2.0, 1e-6)
    denominator = slope_right - slope_left
    rho = float(np.clip((slope_right + slope_left) / denominator, -0.95, 0.95)) if abs(
        denominator
    ) > 1e-12 else 0.0
    m = float(k[int(np.argmin(w))])
    w_min = float(np.min(w))

    candidates: list[SVIParams | None] = []
    for sigma in (0.05 * span, 0.15 * span, 0.40 * span, 1.0 * span):
        sigma = max(sigma, 1e-4)
        a = w_min - b * sigma * np.sqrt(1.0 - rho**2)
        candidates.append(SVIParams(a=float(max(a, 0.0)), b=b, rho=rho, m=m, sigma=float(sigma)))
    candidates.append(None)
    return candidates


def _best_svi_fit(k: NDArray[np.float64], w: NDArray[np.float64]) -> SVIFit:
    """Fit from every candidate start and keep the lowest sum of squares.

    Raises RuntimeError only if *every* start fails, quoting the last
    failure so the caller sees why.
    """
    best: SVIFit | None = None
    last_error: Exception | None = None
    for guess in _initial_guesses(k, w):
        try:
            fit = fit_svi(k, w) if guess is None else fit_svi(k, w, initial_guess=guess)
        except (RuntimeError, ValueError) as exc:  # a stalled start, not a bad chain
            last_error = exc
            continue
        if best is None or fit.sse < best.sse:
            best = fit
    if best is None:
        raise RuntimeError(
            f"SVI calibration failed from every starting point; last error: {last_error}"
        )
    return best


def _usable_quotes(chain: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the smile fit cannot use, and explain what is left.

    A quote is usable when its strike and implied volatility are both finite
    and strictly positive. Call prices are *not* required: the density is
    re-priced from the fitted smile, so a missing call mid only costs a
    quote if its IV is missing too.
    """
    missing = [c for c in ("strike", "iv") if c not in chain.columns]
    if missing:
        raise ValueError(
            f"chain is missing required column(s) {missing}; expected the schema from "
            "options_pricing/refined/data_scraper.py (strike, call_price, put_price, iv, ...)"
        )
    usable = chain[np.isfinite(chain["strike"]) & np.isfinite(chain["iv"])]
    usable = usable[(usable["strike"] > 0.0) & (usable["iv"] > 0.0)]
    return usable.drop_duplicates(subset="strike").sort_values("strike")


def market_view_from_chain(
    chain: pd.DataFrame,
    spot: float,
    expiry_years: float,
    rate: float = 0.0,
    grid_points: int = _DEFAULT_GRID_POINTS,
    wing_extrapolation: float = 0.25,
) -> MarketView:
    """Build a `MarketView` from one expiry's option chain.

    Parameters
    ----------
    chain : DataFrame with at least `strike` and `iv` columns, one expiry.
        `iv` is a decimal (0.25 = 25%), not a percentage.
    spot : underlying spot price.
    expiry_years : time to expiry in years, > 0.
    rate : continuously compounded risk-free rate.
    grid_points : strikes on the dense re-pricing grid. The finite-difference
        error in the density falls as h^2, so more points help until quote
        noise in the *fit* dominates; a few hundred is the usual plateau.
    wing_extrapolation : how far beyond the quoted log-moneyness range to
        extend the grid, as a fraction of that range. 0.0 keeps strictly to
        quoted strikes and truncates the tails; the default 0.25 buys tail
        mass at the price of trusting the SVI wings.

    Raises
    ------
    ValueError : fewer than five usable quotes, non-positive spot or expiry,
        or a degenerate grid.
    RuntimeError : propagated from `fit_svi` when calibration fails, or from
        `extract_density` when the re-priced curve has no convexity.
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be strictly positive, got {spot}")
    if expiry_years <= 0.0:
        raise ValueError(f"expiry_years must be strictly positive, got {expiry_years}")
    if grid_points < 3:
        raise ValueError(f"need at least 3 grid points for a second difference, got {grid_points}")
    if wing_extrapolation < 0.0:
        raise ValueError(
            f"wing_extrapolation is a non-negative fraction of the quoted range, got "
            f"{wing_extrapolation}"
        )

    quotes = _usable_quotes(chain)
    if len(quotes) < _MIN_QUOTES:
        raise ValueError(
            f"SVI needs at least {_MIN_QUOTES} usable quotes (finite, positive strike and iv); "
            f"the chain has {len(quotes)} of {len(chain)} rows"
        )

    forward = float(spot * np.exp(rate * expiry_years))
    strikes = quotes["strike"].to_numpy(dtype=np.float64)
    iv = quotes["iv"].to_numpy(dtype=np.float64)
    k = np.log(strikes / forward)
    total_variance = iv**2 * expiry_years

    svi_fit = _best_svi_fit(k, total_variance)

    span = float(k[-1] - k[0])
    if span <= 0.0:
        raise ValueError("quoted strikes collapse to a single log-moneyness point")
    pad = wing_extrapolation * span
    dense_k: NDArray[np.float64] = np.linspace(k[0] - pad, k[-1] + pad, grid_points)
    dense_strikes = forward * np.exp(dense_k)
    dense_w = np.clip(svi_total_variance(dense_k, svi_fit.params), 1e-12, None)
    dense_iv = np.sqrt(dense_w / expiry_years)
    dense_calls = call_price(spot, dense_strikes, expiry_years, rate, dense_iv)

    distribution = extract_density(dense_strikes, dense_calls, expiry_years, rate)
    mean_return, variance_return, skew_return = implied_return_moments(distribution, spot)

    atm_w = float(svi_total_variance(np.array([0.0]), svi_fit.params)[0])
    return MarketView(
        spot=float(spot),
        expiry_years=float(expiry_years),
        rate=float(rate),
        forward=forward,
        svi_fit=svi_fit,
        distribution=distribution,
        mean_return=mean_return,
        variance_return=variance_return,
        skew_return=skew_return,
        implied_vol_atm=float(np.sqrt(max(atm_w, 0.0) / expiry_years)),
        forward_error=float(distribution.mean / forward - 1.0),
        wing_extrapolation=float(wing_extrapolation),
        n_quotes=len(quotes),
    )
