"""Synthetic option chains carrying a realistic volatility smile.

`options_pricing/refined/data_scraper.synthetic_equity_chain` prices every
strike at one flat volatility. That is the right fixture for checking
Black-Scholes identities, but the wrong one for this pipeline: a flat smile
implies a lognormal terminal distribution whose skew is a fixed function of
volatility alone, so the skew channel of the divergence signal would be
testing nothing.

Here the chain is generated *from* an SVI slice with a negative rho -- the
downside-heavy shape equity index options actually trade at -- and each
strike is then priced with Black-Scholes at its own implied volatility.
Fitting SVI back to the result and extracting the density is a genuine
round trip: `market_view_from_chain` should recover the generating skew.

Quote noise is added in volatility, not price, because that is where it
lives in real chains: a one-tick bid-ask on a far out-of-the-money option
is a large move in price terms and a small one in vol terms.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..quant_model.black_scholes import call_price, put_price
from ..quant_model.svi_model import SVIParams, svi_total_variance

__all__ = ["mock_chain_series", "mock_option_chain", "smile_params"]

_VERTEX_WIDTH = 0.35  # SVI sigma: the curvature scale of the smile vertex
_DEFAULT_WING = 0.11  # SVI b: wing angle, i.e. how fast the smile opens up


def smile_params(atm_vol: float, expiry_years: float, rho: float = -0.55) -> SVIParams:
    """SVI parameters reproducing a given at-the-money vol and skew sign.

    With the vertex at m = 0 the at-the-money total variance is
    w(0) = a + b*sigma, so pinning w(0) = atm_vol^2 * T fixes

        a = atm_vol^2 * T - b * sigma.

    The wing angle b is held at a constant so that the smile's *shape*
    stays fixed as the level moves; only a absorbs the level. b is reduced
    if the no-arbitrage variance floor a + b*sigma*sqrt(1-rho^2) >= 0 would
    otherwise be breached at low volatility -- that floor binds exactly when
    the requested smile is deeper than the requested level can support.

    Raises ValueError for non-positive vol or expiry, or |rho| >= 1.
    """
    if atm_vol <= 0.0:
        raise ValueError(f"atm_vol must be strictly positive, got {atm_vol}")
    if expiry_years <= 0.0:
        raise ValueError(f"expiry_years must be strictly positive, got {expiry_years}")
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must lie strictly in (-1, 1), got {rho}")

    atm_variance = atm_vol**2 * expiry_years
    # min_k w = a + b sigma sqrt(1-rho^2) = atm_variance - b sigma (1 - sqrt(1-rho^2)),
    # so b <= atm_variance / (sigma (1 - sqrt(1-rho^2))) keeps it non-negative.
    shrinkage = _VERTEX_WIDTH * (1.0 - np.sqrt(1.0 - rho**2))
    b_cap = atm_variance / shrinkage if shrinkage > 0.0 else np.inf
    b = float(min(_DEFAULT_WING, 0.9 * b_cap))
    return SVIParams(
        a=float(atm_variance - b * _VERTEX_WIDTH), b=b, rho=float(rho), m=0.0, sigma=_VERTEX_WIDTH
    )


def mock_option_chain(
    spot: float = 100.0,
    expiry_years: float = 21.0 / 252.0,
    rate: float = 0.03,
    atm_vol: float = 0.20,
    rho: float = -0.55,
    n_strikes: int = 31,
    log_moneyness_width: float = 0.30,
    quote_noise: float = 0.0,
    expiry_label: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Price one expiry's chain off an SVI smile.

    Parameters
    ----------
    spot : underlying price.
    expiry_years : time to expiry in years; the default is one month.
    rate : continuously compounded risk-free rate.
    atm_vol : at-the-money-forward implied volatility, as a decimal.
    rho : SVI skew. Negative (the default) lifts the downside wing, giving
        the negative return skew equity indices price.
    n_strikes : strikes on the ladder, >= 5 so SVI can be refitted.
    log_moneyness_width : half-width of the ladder in ln(K/F). 0.30 spans
        roughly 74%-135% of forward, a realistic listed range.
    quote_noise : standard deviation of additive volatility noise, in vol
        points (0.005 = half a vol point). Non-zero makes the refitted
        smile an approximation rather than an exact recovery, which is what
        real quotes force.
    expiry_label : value for the `expiry` column; a synthetic tag by default.
    seed : seed for the quote noise.

    Returns
    -------
    DataFrame in the canonical chain schema: `strike`, `call_price`,
    `put_price`, `iv`, `expiry`, `timestamp`, sorted by strike.
    """
    if n_strikes < 5:
        raise ValueError(f"need at least 5 strikes to refit a 5-parameter smile, got {n_strikes}")
    if log_moneyness_width <= 0.0:
        raise ValueError(f"log_moneyness_width must be positive, got {log_moneyness_width}")
    if quote_noise < 0.0:
        raise ValueError(f"quote_noise must be non-negative, got {quote_noise}")

    params = smile_params(atm_vol, expiry_years, rho)
    forward = spot * np.exp(rate * expiry_years)
    k = np.linspace(-log_moneyness_width, log_moneyness_width, n_strikes)
    strikes = forward * np.exp(k)

    total_variance = np.clip(svi_total_variance(k, params), 1e-10, None)
    iv = np.sqrt(total_variance / expiry_years)
    if quote_noise > 0.0:
        rng = np.random.default_rng(seed)
        # Floor at a tenth of the clean level: noise may perturb a quote,
        # not invert it into a negative volatility.
        iv = np.maximum(iv + rng.standard_normal(n_strikes) * quote_noise, 0.1 * iv)

    return pd.DataFrame(
        {
            "strike": strikes,
            "call_price": call_price(spot, strikes, expiry_years, rate, iv),
            "put_price": put_price(spot, strikes, expiry_years, rate, iv),
            "iv": iv,
            "expiry": expiry_label or f"SYNTH_{expiry_years:.4f}Y",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ).sort_values("strike", ignore_index=True)


def mock_chain_series(
    dates: pd.DatetimeIndex,
    spots: ArrayLike,
    vols: ArrayLike,
    expiry_years: float = 21.0 / 252.0,
    rate: float = 0.03,
    rho: float = -0.55,
    skew_sensitivity: float = -1.2,
    quote_noise: float = 0.0025,
    seed: int | None = None,
    **chain_kwargs: object,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """One chain per date, with the smile responding to the volatility level.

    `rho` is not held constant across days. Real index skew steepens as
    volatility rises, so rho is pushed further negative in proportion to how
    far that day's volatility sits above the sample median:

        rho_t = clip(rho + skew_sensitivity * (vol_t - median(vol)), -0.95, 0)

    Without this the skew channel of the divergence signal would be constant
    through every regime and could never contribute anything. The clip keeps
    rho inside SVI's admissible range.

    Parameters
    ----------
    dates : the dates to quote; keys of the returned mapping.
    spots : underlying price on each date, same length as `dates`.
    vols : at-the-money implied volatility on each date, same length.
    expiry_years, rate : as for `mock_option_chain`.
    rho : baseline skew, applied at the median volatility.
    skew_sensitivity : how much rho moves per unit of volatility deviation.
    quote_noise : per-quote volatility noise; defaults to a quarter of a
        vol point, which is a plausible half-spread on a liquid index.
    seed : base seed; each date draws from a distinct derived stream so
        that quote noise is independent across days but reproducible.
    """
    spot_array = np.asarray(spots, dtype=np.float64).ravel()
    vol_array = np.asarray(vols, dtype=np.float64).ravel()
    if not len(dates) == spot_array.size == vol_array.size:
        raise ValueError(
            f"dates, spots and vols must have equal length; got {len(dates)}, "
            f"{spot_array.size}, {vol_array.size}"
        )
    if spot_array.size == 0:
        raise ValueError("cannot build a chain series over zero dates")
    if np.any(~np.isfinite(vol_array)) or np.any(vol_array <= 0.0):
        raise ValueError("vols must all be finite and strictly positive")

    median_vol = float(np.median(vol_array))
    chains: dict[pd.Timestamp, pd.DataFrame] = {}
    for offset, (date, spot, vol) in enumerate(zip(dates, spot_array, vol_array)):
        day_rho = float(np.clip(rho + skew_sensitivity * (vol - median_vol), -0.95, 0.0))
        chains[pd.Timestamp(date)] = mock_option_chain(
            spot=float(spot),
            expiry_years=expiry_years,
            rate=rate,
            atm_vol=float(vol),
            rho=day_rho,
            quote_noise=quote_noise,
            expiry_label=pd.Timestamp(date).strftime("%Y-%m-%d"),
            seed=None if seed is None else seed + offset,
            **chain_kwargs,  # type: ignore[arg-type]
        )
    return chains
