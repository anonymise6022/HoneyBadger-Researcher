"""Options-chain retrieval for equity and FX underlyings.

Network I/O is confined to `fetch_stock_options_chain` and `_paginate`;
`build_chain`, `synthetic_equity_chain` and `synthetic_fx_chain` are pure
and testable without a socket. Every path returns one schema, so
downstream modules never learn where the quotes came from:

    strike      strike price, in the underlying's quote currency
    call_price  call close/mid (NaN when unquoted)
    put_price   put close/mid (NaN when unquoted)
    iv          implied volatility as a decimal (0.25 = 25%)
    expiry      expiration label, YYYY-MM-DD for live data
    timestamp   UTC ISO-8601 retrieval time

Equity chains come from Polygon.io's options snapshot endpoint (free API
key via POLYGON_API_KEY). FX options have no free public chain: they trade
OTC, quoted bank-to-bank by delta rather than strike, and Polygon/OPRA
covers only listed equity and index options. `synthetic_fx_chain` stands in
with Black-Scholes prices at realistic G10 vol levels; point a paid feed
(Bloomberg BGN, ICE) at `build_chain` to swap in real data.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests
from numpy.typing import ArrayLike

from black_scholes import call_price, put_price

__all__ = [
    "CHAIN_COLUMNS",
    "OptionsDataError",
    "build_chain",
    "fetch_stock_options_chain",
    "synthetic_equity_chain",
    "synthetic_fx_chain",
]

POLYGON_BASE_URL = "https://api.polygon.io"
REQUEST_DELAY_SEC = 0.25  # throttle between pages; free tier allows ~5 req/min
REQUEST_TIMEOUT_SEC = 10.0
MAX_PAGES = 20  # hard stop so a malformed cursor cannot loop forever

CHAIN_COLUMNS = ["strike", "call_price", "put_price", "iv", "expiry", "timestamp"]


class OptionsDataError(RuntimeError):
    """Raised when an options chain cannot be retrieved or parsed."""


def build_chain(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    put_prices: ArrayLike,
    iv: ArrayLike,
    expiry: str,
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Assemble the canonical chain DataFrame from aligned arrays.

    The single construction point for the schema, so live, synthetic and
    future vendor sources emit identical columns. Pure apart from reading
    the clock when `timestamp` is omitted.
    """
    cols = [np.asarray(x, dtype=np.float64).ravel() for x in (strikes, call_prices, put_prices)]
    if cols[0].size == 0:
        raise ValueError("cannot build a chain with zero strikes")
    if len({c.size for c in cols}) != 1:
        raise ValueError(
            "strikes, call_prices and put_prices must have equal length; got "
            f"{cols[0].size}, {cols[1].size}, {cols[2].size}"
        )

    return pd.DataFrame(
        {
            "strike": cols[0],
            "call_price": cols[1],
            "put_price": cols[2],
            "iv": np.broadcast_to(np.asarray(iv, dtype=np.float64), cols[0].shape),
            "expiry": expiry,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
    ).sort_values("strike", ignore_index=True)


def _synthetic_chain(
    spot: float, years: float, rate: float, vol: float, n_strikes: int, width: float, label: str
) -> pd.DataFrame:
    """Price a strike ladder spanning spot*(1 +/- width), vectorized.

    Positivity of spot/years/vol is enforced downstream by
    black_scholes._prepare, which names the offending argument.
    """
    if n_strikes < 3:
        raise ValueError(f"need at least 3 strikes for a usable chain, got {n_strikes}")
    if not 0.0 < width < 1.0:
        raise ValueError(f"width must lie in (0, 1) as a fraction of spot, got {width}")
    strikes = np.linspace(spot * (1.0 - width), spot * (1.0 + width), n_strikes)
    calls = call_price(spot, strikes, years, rate, vol)
    puts = put_price(spot, strikes, years, rate, vol)
    return build_chain(strikes, calls, puts, vol, label)


def synthetic_equity_chain(
    spot: float = 100.0,
    expiry_years: float = 0.5,
    rate: float = 0.03,
    vol: float = 0.25,
    n_strikes: int = 25,
    width: float = 0.4,
) -> pd.DataFrame:
    """Black-Scholes equity chain, arbitrage-free by construction.

    The right fixture for verifying SVI and density extraction: the
    recovered distribution must match the known lognormal with mean
    S*exp(rT) and variance S^2 exp(2rT)(exp(sigma^2 T) - 1).
    """
    return _synthetic_chain(spot, expiry_years, rate, vol, n_strikes, width, f"SYNTH_EQ_{vol:.2f}")


def synthetic_fx_chain(
    pair: str = "EURUSD",
    spot: float = 1.08,
    expiry_years: float = 0.25,
    domestic_rate: float = 0.02,
    vol: float = 0.09,
    n_strikes: int = 25,
    width: float = 0.08,
) -> pd.DataFrame:
    """Black-Scholes FX chain for a currency pair.

    Uses the domestic rate rather than full Garman-Kohlhagen (1983), which
    also discounts the spot leg by exp(-r_f T); the two agree once spot is
    replaced by the forward, and only internal consistency is needed here.
    Defaults follow G10 convention: 9% vol and a +/-8% strike band, versus
    an equity's 25% and +/-40%, because FX distributions are far tighter.
    """
    if not pair:
        raise ValueError("pair must be a non-empty currency-pair label, e.g. 'EURUSD'")
    return _synthetic_chain(
        spot, expiry_years, domestic_rate, vol, n_strikes, width, f"SYNTH_FX_{pair}"
    )


def fetch_stock_options_chain(
    ticker: str,
    expiry: str | None = None,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a live equity chain from Polygon.io, pairing calls and puts.

    ticker : underlying symbol, e.g. "AAPL".
    expiry : expiration as YYYY-MM-DD; None returns all listed expiries.
    api_key : Polygon key, else the POLYGON_API_KEY environment variable.
    session : optional requests.Session for connection reuse.

    Raises OptionsDataError when credentials are missing, a request fails,
    or no usable contracts come back. Callers wanting an offline fallback
    should catch it and call `synthetic_equity_chain`.
    """
    if not ticker:
        raise ValueError("ticker must be a non-empty symbol, e.g. 'AAPL'")
    key = api_key or os.environ.get("POLYGON_API_KEY")
    if not key:
        raise OptionsDataError(
            "no Polygon.io API key: pass api_key= or set POLYGON_API_KEY "
            "(free key at https://polygon.io/)"
        )

    params: dict[str, Any] = {"apiKey": key, "limit": 250}
    if expiry is not None:
        params["expiration_date"] = expiry

    http = session or requests.Session()
    rows: dict[tuple[float, str], dict[str, Any]] = {}
    for payload in _paginate(http, f"{POLYGON_BASE_URL}/v3/snapshot/options/{ticker}", params):
        _accumulate_contracts(payload.get("results", []), rows)

    if not rows:
        raise OptionsDataError(
            f"Polygon returned no contracts for {ticker}{' ' + expiry if expiry else ''}; "
            "check the symbol, the expiration date, and your plan's entitlements"
        )

    ordered = sorted(rows.values(), key=lambda row: row["strike"])
    return build_chain(
        [r["strike"] for r in ordered],
        [r["call_price"] for r in ordered],
        [r["put_price"] for r in ordered],
        [r["iv"] for r in ordered],
        expiry or "MIXED",
    )


def _paginate(
    http: requests.Session, url: str, params: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Yield Polygon pages, following the next_url cursor under a page cap."""
    api_key = params.get("apiKey")
    next_url: str | None = url
    page_params: dict[str, Any] | None = params

    for page in range(MAX_PAGES):
        if next_url is None:
            return
        if page > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            response = http.get(next_url, params=page_params, timeout=REQUEST_TIMEOUT_SEC)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise OptionsDataError(f"Polygon request timed out after {REQUEST_TIMEOUT_SEC}s") from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            hint = " (free tier allows ~5 requests/minute)" if status == 429 else ""
            raise OptionsDataError(f"Polygon returned HTTP {status}{hint}") from exc
        except requests.RequestException as exc:
            raise OptionsDataError(f"Polygon request failed: {exc}") from exc
        except ValueError as exc:
            raise OptionsDataError("Polygon returned a non-JSON response") from exc

        status = payload.get("status")
        if status not in ("OK", "DELAYED"):
            raise OptionsDataError(
                f"Polygon reported status {status!r}: {payload.get('error', 'no detail')}"
            )
        yield payload

        next_url = payload.get("next_url")  # cursor embeds its own query string
        page_params = {"apiKey": api_key} if next_url else None


def _accumulate_contracts(
    contracts: list[dict[str, Any]], rows: dict[tuple[float, str], dict[str, Any]]
) -> None:
    """Fold snapshot contracts into per-(strike, expiry) call/put rows.

    Malformed entries are skipped rather than raised on: one bad contract
    should not discard an otherwise usable chain.
    """
    for contract in contracts:
        details = contract.get("details") or {}
        strike, expiry = details.get("strike_price"), details.get("expiration_date")
        side = details.get("contract_type")
        if strike is None or expiry is None or side not in ("call", "put"):
            continue

        row = rows.setdefault(
            (float(strike), str(expiry)),
            {
                "strike": float(strike),
                "expiry": str(expiry),
                "call_price": np.nan,
                "put_price": np.nan,
                "iv": np.nan,
            },
        )
        row[f"{side}_price"] = (contract.get("day") or {}).get("close", np.nan)
        iv = contract.get("implied_volatility")
        if iv is not None and np.isnan(row["iv"]):
            row["iv"] = float(iv)
