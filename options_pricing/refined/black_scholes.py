"""Black-Scholes-Merton pricing and Greeks, vectorized over all arguments.

Risk-neutral dynamics, constant vol, no dividends:

    dS = r S dt + sigma S dW   =>   ln(S_T/S) ~ N((r - sigma^2/2)T, sigma^2 T)

    d1 = [ln(S/K) + (r + sigma^2/2)T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

N(d2) is the risk-neutral probability of expiring in the money; N(d1) is the
same probability under the stock numeraire, which is why it is also delta.

References: Black & Scholes (1973), JPE 81(3); Merton (1973), Bell J. Econ
4(1); Hull (2018), *Options, Futures and Other Derivatives*, ch. 15, 19.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

OptionType = Literal["call", "put"]

__all__ = ["d1_d2", "call_price", "put_price", "delta", "gamma", "theta", "vega"]


class _Prepared(NamedTuple):
    """Validated broadcast inputs bundled with d1/d2, computed once per call."""

    S: NDArray[np.float64]
    K: NDArray[np.float64]
    T: NDArray[np.float64]
    r: NDArray[np.float64]
    sigma: NDArray[np.float64]
    d1: NDArray[np.float64]
    d2: NDArray[np.float64]


def _prepare(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike) -> _Prepared:
    """Broadcast, validate the model domain, and derive d1/d2.

    S, K, T, sigma must be strictly positive: each zero/negative case is a
    division by zero or log of a non-positive number. The T -> 0 and
    sigma -> 0 limits are genuinely discontinuous (price collapses to
    intrinsic max(S-K,0), whose gamma is a Dirac delta), so they are
    rejected rather than silently returning a limit the Greeks cannot show.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in (S, K, T, r, sigma)]
    try:
        S_b, K_b, T_b, r_b, sig_b = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        raise ValueError(
            f"S, K, T, r, sigma must be broadcast-compatible; got shapes "
            f"{[np.shape(a) for a in arrays]}"
        ) from exc

    if S_b.size == 0:
        raise ValueError("inputs are empty; expected at least one contract")
    for name, arr in (("S", S_b), ("K", K_b), ("T", T_b), ("sigma", sig_b)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")
        if np.any(arr <= 0.0):
            raise ValueError(
                f"{name} must be strictly positive; Black-Scholes is undefined at "
                f"{name} <= 0 (minimum observed: {np.min(arr)})"
            )
    if not np.all(np.isfinite(r_b)):
        raise ValueError("r contains NaN or infinite values")

    vol_sqrt_T = sig_b * np.sqrt(T_b)
    d1 = (np.log(S_b / K_b) + (r_b + 0.5 * sig_b**2) * T_b) / vol_sqrt_T
    return _Prepared(S_b, K_b, T_b, r_b, sig_b, d1, d1 - vol_sqrt_T)


def _check_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def d1_d2(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """d1 = [ln(S/K) + (r + sigma^2/2)T]/(sigma sqrt(T)); d2 = d1 - sigma sqrt(T).

    S=spot, K=strike, T=years to expiry, r=continuously compounded rate,
    sigma=annualized vol. All arguments broadcast. Raises ValueError on
    incompatible shapes, empty input, or non-positive S/K/T/sigma.
    """
    p = _prepare(S, K, T, r, sigma)
    return p.d1, p.d2


def call_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> NDArray[np.float64]:
    """European call:  C = S N(d1) - K exp(-rT) N(d2)."""
    p = _prepare(S, K, T, r, sigma)
    return p.S * norm.cdf(p.d1) - p.K * np.exp(-p.r * p.T) * norm.cdf(p.d2)


def put_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> NDArray[np.float64]:
    """European put:  P = K exp(-rT) N(-d2) - S N(-d1).

    Equal to C - S + K exp(-rT) by parity, but evaluated directly so deep
    ITM puts do not lose precision cancelling two large terms.
    """
    p = _prepare(S, K, T, r, sigma)
    return p.K * np.exp(-p.r * p.T) * norm.cdf(-p.d2) - p.S * norm.cdf(-p.d1)


def delta(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
    option_type: OptionType = "call",
) -> NDArray[np.float64]:
    """dV/dS:  delta_call = N(d1),  delta_put = N(d1) - 1.

    Differentiating parity in S gives delta_call - delta_put = 1 exactly.
    """
    _check_type(option_type)
    nd1 = norm.cdf(_prepare(S, K, T, r, sigma).d1)
    return nd1 if option_type == "call" else nd1 - 1.0


def gamma(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> NDArray[np.float64]:
    """d2V/dS2 = phi(d1) / (S sigma sqrt(T)).

    Same for calls and puts: they differ by the linear payoff S - K exp(-rT),
    whose second derivative in S vanishes.
    """
    p = _prepare(S, K, T, r, sigma)
    return norm.pdf(p.d1) / (p.S * p.sigma * np.sqrt(p.T))


def vega(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> NDArray[np.float64]:
    """dV/dsigma = S phi(d1) sqrt(T), per unit vol (1.00 = 100 vol points).

    Divide by 100 for the per-vol-point desk convention. Call and put vega
    coincide, since they differ only by a vol-independent term.
    """
    p = _prepare(S, K, T, r, sigma)
    return p.S * norm.pdf(p.d1) * np.sqrt(p.T)


def theta(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
    option_type: OptionType = "call",
) -> NDArray[np.float64]:
    """Time decay per year, -dV/dT:

        theta_call = -S phi(d1) sigma / (2 sqrt(T)) - r K exp(-rT) N(d2)
        theta_put  = -S phi(d1) sigma / (2 sqrt(T)) + r K exp(-rT) N(-d2)

    Shared first term is the cost of carrying gamma; the second is interest
    on the discounted strike, which flips sign between calls and puts.
    Divide by 365 for the per-calendar-day figure brokers quote.
    """
    _check_type(option_type)
    p = _prepare(S, K, T, r, sigma)
    decay = -p.S * norm.pdf(p.d1) * p.sigma / (2.0 * np.sqrt(p.T))
    carry = p.r * p.K * np.exp(-p.r * p.T)
    return decay - carry * norm.cdf(p.d2) if option_type == "call" else decay + carry * norm.cdf(-p.d2)
