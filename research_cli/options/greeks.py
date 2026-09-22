"""Black-Scholes prices and greeks, in the units a broker screen shows them.

The formulas are the textbook ones. What is worth stating is the *units*,
because that is where a options screen either reads like a trading platform
or like a homework answer, and the difference is entirely cosmetic
arithmetic applied consistently:

* **Delta** per share, so 0.62 means the contract behaves like 62 shares of
  the underlying. Multiply by 100 for the contract.
* **Gamma** per share, per one dollar of underlying movement.
* **Theta per day**, not per year. Nobody holds an option for a year and
  thinks in annualized decay; every platform divides by 365, so this does
  too. The number is what the position loses overnight, all else equal.
* **Vega per one point of implied volatility** -- the move from 24% to 25%
  -- not per 1.00 (that is, per 100 points), which is what the raw partial
  derivative gives.
* **Rho per one point of interest rate**, for the same reason.

Everything here is pure standard library on purpose. The paper-trading desk
has to price a whole chain on every refresh, and a dependency-free module
can be imported and tested without the scientific stack behind it.

No dividends. A dividend before expiry lowers a call and raises a put, and
for a single-name stock with a fat yield the error is real -- it is reported
as a caveat on screen rather than silently absorbed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Greeks",
    "black_scholes_price",
    "greeks",
    "implied_volatility",
    "intrinsic_value",
]

#: Shares per contract. One hundred everywhere in US listed options, and the
#: reason a $2.40 quote costs $240.
CONTRACT_SIZE = 100

_DAYS_PER_YEAR = 365.0
_MIN_VOL = 0.005
_MAX_VOL = 5.0
#: Below this, one point of implied volatility moves the price by less than
#: half a cent -- finer than the chain quotes -- so the price simply does not
#: carry the information. See `implied_volatility`.
_MIN_VEGA = 0.005


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function -- exact, not tabulated."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot: float, strike: float, years: float, rate: float, vol: float) -> tuple[float, float]:
    vol_sqrt_t = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def intrinsic_value(spot: float, strike: float, kind: str) -> float:
    """What the contract is worth if it settled right now."""
    if kind == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def black_scholes_price(
    spot: float, strike: float, years: float, rate: float, vol: float, kind: str
) -> float:
    """European option price. At or past expiry, the intrinsic payoff."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if years <= 0 or vol <= 0:
        return intrinsic_value(spot, strike, kind)

    d1, d2 = _d1_d2(spot, strike, years, rate, vol)
    discount = math.exp(-rate * years)
    if kind == "call":
        return spot * _normal_cdf(d1) - strike * discount * _normal_cdf(d2)
    return strike * discount * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    """One contract's sensitivities, per share, in trading-desk units."""

    price: float
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega: float    # per one point of implied volatility
    rho: float     # per one point of interest rate

    def per_contract(self) -> "Greeks":
        """The same numbers for one contract of 100 shares.

        Gamma is *not* scaled the way the others are: it is already a rate
        of change of delta, and a desk quotes it per share so that the
        position's gamma times a one dollar move reads directly as a change
        in share-equivalent exposure.
        """
        return Greeks(
            price=self.price * CONTRACT_SIZE,
            delta=self.delta * CONTRACT_SIZE,
            gamma=self.gamma * CONTRACT_SIZE,
            theta=self.theta * CONTRACT_SIZE,
            vega=self.vega * CONTRACT_SIZE,
            rho=self.rho * CONTRACT_SIZE,
        )


def greeks(
    spot: float, strike: float, years: float, rate: float, vol: float, kind: str
) -> Greeks:
    """Price and sensitivities for one share of exposure.

    An expired or zero-volatility contract has no sensitivities worth
    quoting: delta is a step function, gamma is undefined at the strike, and
    reporting the limit as though it were a measurement invites a trader to
    lean on a number that does not exist. Delta is given as the step, which
    is true, and everything else as zero.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        moneyness = spot > strike if kind == "call" else spot < strike
        sign = 1.0 if kind == "call" else -1.0
        return Greeks(
            price=intrinsic_value(max(spot, 0.0), max(strike, 1e-9), kind),
            delta=sign if moneyness else 0.0,
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
        )

    d1, d2 = _d1_d2(spot, strike, years, rate, vol)
    discount = math.exp(-rate * years)
    sqrt_t = math.sqrt(years)
    pdf_d1 = _normal_pdf(d1)

    price = black_scholes_price(spot, strike, years, rate, vol, kind)
    gamma_value = pdf_d1 / (spot * vol * sqrt_t)
    vega_annual = spot * pdf_d1 * sqrt_t
    decay = -spot * pdf_d1 * vol / (2.0 * sqrt_t)

    if kind == "call":
        delta = _normal_cdf(d1)
        theta_annual = decay - rate * strike * discount * _normal_cdf(d2)
        rho_annual = strike * years * discount * _normal_cdf(d2)
    else:
        delta = _normal_cdf(d1) - 1.0
        theta_annual = decay + rate * strike * discount * _normal_cdf(-d2)
        rho_annual = -strike * years * discount * _normal_cdf(-d2)

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma_value,
        theta=theta_annual / _DAYS_PER_YEAR,
        vega=vega_annual / 100.0,
        rho=rho_annual / 100.0,
    )


def implied_volatility(
    target: float, spot: float, strike: float, years: float, rate: float, kind: str
) -> float | None:
    """The volatility that reproduces a quoted price, or None if none does.

    Newton's method converges in a handful of steps from a sensible start
    and is what every pricing library uses, but it is not safe on its own:
    vega collapses toward zero for deep in- and out-of-the-money contracts,
    and dividing by it throws the iteration somewhere absurd. So each step
    is checked, and anything that leaves the bracket hands over to bisection,
    which cannot diverge because the price is monotonic in volatility.

    A quote below intrinsic or above the underlying has no implied
    volatility at all -- it is stale, crossed, or an artefact of a chain
    printed when the market was shut. Returning None says so; returning a
    floor would quietly invent a number the market never traded.

    None also comes back when the answer exists but cannot be measured. A
    deep in-the-money contract a day from expiry is worth its intrinsic
    value to the cent whether volatility is 8% or 75%, so the quote pins
    nothing down and any figure printed beside it is the solver's starting
    guess wearing a number's clothes. The test is vega: if a full point of
    implied volatility moves the price less than half a cent, there is no
    implied volatility to report.
    """
    if years <= 0 or spot <= 0 or strike <= 0 or target <= 0:
        return None
    floor = max(intrinsic_value(spot, strike, kind) - strike * (1 - math.exp(-rate * years)), 0.0)
    ceiling = spot if kind == "call" else strike
    if target < floor - 1e-9 or target > ceiling + 1e-9:
        return None

    low, high = _MIN_VOL, _MAX_VOL
    vol = 0.25
    solved = False
    for _ in range(60):
        estimate = greeks(spot, strike, years, rate, vol, kind)
        error = estimate.price - target
        if abs(error) < 1e-7:
            solved = True
            break
        # Keep the bracket tight as we go, so the fallback has as little
        # ground to cover as possible.
        if error > 0:
            high = vol
        else:
            low = vol
        vega_per_unit = estimate.vega * 100.0
        step = vol - error / vega_per_unit if vega_per_unit > 1e-8 else None
        vol = step if step is not None and low < step < high else 0.5 * (low + high)

    if not solved or not _MIN_VOL < vol < _MAX_VOL:
        return None
    return vol if greeks(spot, strike, years, rate, vol, kind).vega >= _MIN_VEGA else None
