"""
Options-chain scraper for stock options (Polygon.io) and FX options.

Stock options: Polygon.io's options snapshot endpoint
(GET /v3/snapshot/options/{underlyingAsset}) returns per-contract quotes
(strike, last quote/trade, greeks, implied_volatility) for a given
underlying and expiration. Requires a free Polygon.io API key
(POLYGON_API_KEY env var or passed explicitly); falls back to mock data
without one.

FX options: there is no free, keyless public API that publishes FX options
chains (OTC market, quoted bank-to-bank; Polygon/OPRA covers only
equity/index options). No live fetch is implemented for FX — instead
`generate_fx_mock_chain` produces a Black-Scholes-consistent synthetic
chain for a given currency pair, so the rest of the pipeline (SVI fit,
density extraction) can still be exercised. Swap in a paid vol provider
(e.g. Bloomberg BGN, ICE) behind the same DataFrame schema if needed.
"""

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from black_scholes import call_price, put_price

POLYGON_BASE_URL = "https://api.polygon.io"
REQUEST_DELAY_SEC = 0.2  # stay under Polygon's free-tier rate limit (5 req/min-ish)


def _get(endpoint, params, timeout=10):
    try:
        resp = requests.get(f"{POLYGON_BASE_URL}{endpoint}", params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") not in ("OK", "DELAYED"):
            raise ValueError(f"unexpected response status: {payload.get('status')}")
        return payload
    except (requests.RequestException, ValueError) as exc:
        print(f"[data_scraper] request failed for {endpoint}: {exc}")
        return None


def fetch_stock_options_chain(ticker="AAPL", expiry=None, api_key=None):
    """
    Build a DataFrame [strike, call_price, put_price, IV, expiry, timestamp]
    for a stock's options chain via Polygon.io.

    ticker : underlying stock symbol, e.g. "AAPL"
    expiry : expiration date string "YYYY-MM-DD", or None for all listed
    api_key: Polygon.io API key; falls back to POLYGON_API_KEY env var,
             then to mock data if neither is available or the request fails
    """
    api_key = api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("[data_scraper] no Polygon API key set, falling back to mock data")
        return generate_mock_chain()

    params = {"apiKey": api_key, "limit": 250}
    if expiry is not None:
        params["expiration_date"] = expiry

    rows = {}
    next_url = f"/v3/snapshot/options/{ticker}"
    while next_url is not None:
        time.sleep(REQUEST_DELAY_SEC)  # basic rate limiting between pages
        if next_url.startswith("/"):
            payload = _get(next_url, params)
        else:
            try:
                resp = requests.get(next_url, timeout=10)
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                print(f"[data_scraper] pagination request failed: {exc}")
                payload = None
        if payload is None:
            break

        for contract in payload.get("results", []):
            details = contract.get("details", {})
            strike = details.get("strike_price")
            exp = details.get("expiration_date")
            option_type = details.get("contract_type")  # "call" or "put"
            price = (contract.get("day", {}) or {}).get("close", np.nan)
            iv = contract.get("implied_volatility", np.nan)

            key = (strike, exp)
            rows.setdefault(key, {"strike": strike, "expiry": exp,
                                   "call_price": np.nan, "put_price": np.nan,
                                   "IV": iv})
            if option_type == "call":
                rows[key]["call_price"] = price
            elif option_type == "put":
                rows[key]["put_price"] = price

        next_url = payload.get("next_url")  # Polygon cursor-based pagination
        params = {"apiKey": api_key}  # next_url already embeds other params

    if not rows:
        print("[data_scraper] no contracts retrieved, falling back to mock data")
        return generate_mock_chain()

    df = pd.DataFrame(rows.values())
    df["timestamp"] = datetime.now(timezone.utc).isoformat()
    return df.sort_values("strike").reset_index(drop=True)


def generate_mock_chain(S=100.0, T=0.5, r=0.03, sigma=0.25, n_strikes=15):
    """Synthetic equity-style call/put chain via Black-Scholes."""
    strikes = np.linspace(S * 0.6, S * 1.4, n_strikes)
    return _bs_chain(strikes, S, T, r, sigma, expiry_label="MOCK_EQUITY")


def generate_fx_mock_chain(pair="EURUSD", S=1.08, T=0.25, r=0.02, sigma=0.09, n_strikes=15):
    """
    Synthetic FX options chain via Black-Scholes (Garman-Kohlhagen reduces
    to the same formula with r = domestic rate; foreign rate is folded into
    a forward-adjusted spot in a full implementation, omitted here for
    simplicity). Stands in for real FX vol-surface data, which has no free
    public source. sigma=0.09 and narrow strike range reflect typical G10
    FX vol levels versus equities.
    """
    strikes = np.linspace(S * 0.92, S * 1.08, n_strikes)
    return _bs_chain(strikes, S, T, r, sigma, expiry_label=f"MOCK_FX_{pair}")


def _bs_chain(strikes, S, T, r, sigma, expiry_label):
    calls = call_price(S, strikes, T, r, sigma)
    puts = put_price(S, strikes, T, r, sigma)
    return pd.DataFrame({
        "strike": strikes,
        "call_price": calls,
        "put_price": puts,
        "IV": sigma,
        "expiry": expiry_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    print("-- stock options (AAPL, Polygon.io or mock fallback) --")
    stock_chain = fetch_stock_options_chain(ticker="AAPL")
    print(stock_chain.head())
    print(f"rows: {len(stock_chain)}")

    print("\n-- FX options (EURUSD, mock only: no free live source) --")
    fx_chain = generate_fx_mock_chain(pair="EURUSD")
    print(fx_chain.head())
    print(f"rows: {len(fx_chain)}")
