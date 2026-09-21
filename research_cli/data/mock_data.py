"""Synthetic market data, so the tool runs with no API key and no network.

Two jobs. It is the fallback when a live fetch fails -- a beginner typing
their first query should get a working demonstration and a clear label, not
a stack trace -- and it is the fixture layer for tests, which must never
touch the network.

Everything is **deterministic in the symbol**: `mock_price_history("AAPL")`
returns the same series on every machine, every run, forever, because the
seed is derived from a hash of the symbol. That is what lets a test assert
on an actual number instead of just a shape.

The series are built to be *plausible*, not accurate. Prices follow a
geometric random walk whose volatility clusters (a slow autoregressive
log-vol, so quiet weeks and wild weeks both happen), with fat-tailed
shocks. Index symbols get lower volatility than single stocks, and known
high-beta names get more -- enough structure that outlier detection and hit
rates have something real to chew on. Nothing here is fitted to history and
no output computed from it means anything about a real company.
"""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pandas as pd

__all__ = [
    "MOCK_PROFILES",
    "MOCK_SECTORS",
    "mock_fundamentals",
    "mock_headlines",
    "mock_macro_panel",
    "mock_price_history",
    "symbol_seed",
]

#: Per-symbol (annualized vol, annual drift) used to shape the walk. Symbols
#: absent from the table fall back to a mid-cap-ish default derived from the
#: symbol hash, so an unknown ticker still produces something sane.
MOCK_PROFILES: dict[str, tuple[float, float]] = {
    "SPY": (0.16, 0.09), "QQQ": (0.21, 0.12), "DIA": (0.15, 0.07),
    "IWM": (0.22, 0.06), "^VIX": (0.85, 0.00),
    "AAPL": (0.26, 0.14), "MSFT": (0.25, 0.15), "GOOGL": (0.28, 0.12),
    "AMZN": (0.32, 0.13), "META": (0.36, 0.14), "NVDA": (0.45, 0.30),
    "TSLA": (0.55, 0.15), "NFLX": (0.38, 0.11),
    "JNJ": (0.15, 0.05), "KO": (0.16, 0.05), "PG": (0.15, 0.06),
    "XOM": (0.28, 0.08), "JPM": (0.24, 0.09), "T": (0.22, 0.02),
    "F": (0.34, 0.03), "GME": (0.85, 0.00), "PLTR": (0.55, 0.20),
    "BRK-B": (0.17, 0.10), "GLD": (0.14, 0.04), "USO": (0.35, 0.02),
    "TLT": (0.14, 0.01), "BTC-USD": (0.60, 0.25),
    "EURUSD=X": (0.08, 0.00), "GBPUSD=X": (0.09, 0.00),
    "USDJPY=X": (0.09, 0.00), "GBPJPY=X": (0.11, 0.00),
}

#: Real sectors for symbols a user is likely to type. Without this the
#: mock labels NVDA "Consumer Defensive", which reads as a broken tool even
#: though the data is openly synthetic.
MOCK_SECTORS: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "T": "Communication Services",
    "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical",
    "F": "Consumer Cyclical", "GME": "Consumer Cyclical",
    "JNJ": "Healthcare", "KO": "Consumer Defensive", "PG": "Consumer Defensive",
    "XOM": "Energy", "JPM": "Financial Services", "BRK-B": "Financial Services",
    "PLTR": "Technology",
}

_TRADING_DAYS = 252


def symbol_seed(symbol: str) -> int:
    """A stable 32-bit seed derived from the symbol.

    `hash()` is salted per process in Python 3, so it cannot be used here --
    the same symbol would give different data on every run and no test could
    assert a value. BLAKE2b is stable across processes and platforms.
    """
    digest = hashlib.blake2b(symbol.upper().encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _profile(symbol: str) -> tuple[float, float]:
    """Volatility and drift for a symbol, invented stably if unknown."""
    upper = symbol.upper()
    if upper in MOCK_PROFILES:
        return MOCK_PROFILES[upper]
    rng = np.random.default_rng(symbol_seed(upper))
    return float(rng.uniform(0.20, 0.45)), float(rng.uniform(-0.05, 0.18))


def mock_price_history(
    symbol: str,
    start: date | str,
    end: date | str,
    initial_price: float | None = None,
) -> pd.DataFrame:
    """Generate a plausible daily OHLCV history for a symbol.

    Parameters
    ----------
    symbol : ticker, used to seed the generator and pick a volatility.
    start, end : inclusive date bounds. Only business days are emitted;
        exchange holidays are not modelled, which is a known simplification
        (it means the mock has ~9 more sessions a year than reality).
    initial_price : starting level. Defaults to a stable per-symbol level
        between 20 and 500.

    Returns
    -------
    DataFrame indexed by date with columns Open, High, Low, Close, Volume --
    the same schema `price_ingest` gets from the live vendor, so no
    downstream code can tell which it received.
    """
    sessions = pd.bdate_range(start=start, end=end, name="Date")
    if len(sessions) == 0:
        raise ValueError(
            f"no business days between {start} and {end}; widen the range"
        )

    annual_vol, annual_drift = _profile(symbol)
    rng = np.random.default_rng(symbol_seed(symbol))
    n = len(sessions)

    # Volatility clusters: an AR(1) in log-vol means calm and wild stretches
    # both occur, which a constant-sigma walk never produces and which
    # outlier detection needs in order to be tested on anything realistic.
    log_vol = np.empty(n)
    log_vol[0] = np.log(annual_vol)
    shocks = rng.standard_normal(n) * 0.10
    for t in range(1, n):
        log_vol[t] = 0.97 * log_vol[t - 1] + 0.03 * np.log(annual_vol) + shocks[t]
    daily_vol = np.exp(log_vol) / np.sqrt(_TRADING_DAYS)

    # Student-t innovations (df=4), rescaled to unit variance, so the series
    # has the fat tails real returns have and 2-sigma days actually occur.
    df = 4.0
    raw = rng.standard_t(df, size=n) * np.sqrt((df - 2.0) / df)
    # The +sigma^2/2 term compensates for volatility drag. Without it the
    # *compounded* growth of the path is annual_drift - sigma^2/2, which for
    # a clustered-volatility series can be deeply negative: a 9%-drift "SPY"
    # lost money over eleven simulated years, which is not a plausible
    # fixture for a backtest demo. With it, annual_drift means the geometric
    # return the path actually delivers.
    returns = (annual_drift / _TRADING_DAYS + 0.5 * daily_vol**2) + daily_vol * raw

    if initial_price is None:
        initial_price = float(np.random.default_rng(symbol_seed(symbol) + 1).uniform(20.0, 500.0))
    close = initial_price * np.cumprod(1.0 + returns)

    # Intraday range scaled to that day's volatility, with the open gapping
    # a fraction of the way from the prior close.
    previous_close = np.concatenate([[initial_price], close[:-1]])
    gap = rng.normal(0.0, 0.3, n) * daily_vol
    open_ = previous_close * (1.0 + gap)
    span = np.abs(rng.normal(0.0, 1.0, n)) * daily_vol * close
    high = np.maximum(open_, close) + span * 0.5
    low = np.minimum(open_, close) - span * 0.5
    volume = (rng.lognormal(15.5, 0.45, n) * (1.0 + 3.0 * np.abs(returns))).round()

    return pd.DataFrame(
        {
            "Open": open_, "High": high, "Low": np.maximum(low, 0.01),
            "Close": close, "Volume": volume,
        },
        index=sessions,
    )


def mock_fundamentals(symbol: str) -> dict[str, object]:
    """Invent a plausible fundamentals dictionary for a symbol.

    Shaped like the subset of yfinance's `.info` that `evidence/snapshot`
    consumes, including the awkward cases that break naive code: a
    loss-making company has a *negative* trailing P/E, a non-payer has no
    dividend yield at all rather than zero, and some fields are simply
    absent. Those are the edge cases worth having in a fixture.
    """
    upper = symbol.upper()
    rng = np.random.default_rng(symbol_seed(upper) + 7)
    annual_vol, _ = _profile(upper)

    # High-volatility names are given growth-company financials: fast
    # revenue growth, thin or negative margins.
    growthy = annual_vol > 0.35
    profitable = bool(rng.random() > (0.35 if growthy else 0.05))

    # A loss-maker gets a strictly negative margin: a positive margin
    # beside a negative P/E is internally inconsistent, and inconsistent
    # fixtures teach downstream code to tolerate impossible states.
    margin = float(rng.uniform(0.05, 0.35) if profitable else rng.uniform(-0.25, -0.01))
    revenue_growth = float(rng.uniform(0.15, 0.60) if growthy else rng.uniform(-0.05, 0.18))
    market_cap = float(rng.uniform(2e9, 3.2e12))

    info: dict[str, object] = {
        "symbol": upper,
        "shortName": f"{upper} (synthetic)",
        "quoteType": "EQUITY",
        "sector": MOCK_SECTORS.get(upper, str(rng.choice([
            "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
            "Consumer Defensive", "Energy", "Industrials", "Communication Services",
        ]))),
        "marketCap": market_cap,
        "profitMargins": margin,
        "revenueGrowth": revenue_growth,
        # PERCENT, as yfinance reports it (78.4 means 0.78x).
        "debtToEquity": float(rng.uniform(5.0, 220.0)),
        "returnOnEquity": float(rng.uniform(-0.15, 0.45)),
        "beta": float(rng.uniform(0.5, 2.2)),
    }
    if profitable:
        info["trailingPE"] = float(rng.uniform(9.0, 55.0))
        info["forwardPE"] = float(info["trailingPE"]) * float(rng.uniform(0.75, 1.1))
    else:
        # A loss-maker's trailing P/E is negative or simply not published.
        if rng.random() > 0.5:
            info["trailingPE"] = float(rng.uniform(-60.0, -5.0))
    if rng.random() > 0.45:  # a bit over half of names pay nothing
        # PERCENT, matching yfinance's convention for this field. The mock
        # must speak the vendor's units, not ours, or the normalization in
        # `fundamentals.normalize_info` would be exercised only in
        # production -- which is where a units bug is most expensive.
        info["dividendYield"] = float(rng.uniform(0.4, 6.0))
    return info


def mock_headlines(symbol: str, limit: int = 6) -> list:
    """Deterministic, obviously-synthetic headlines for a symbol.

    Written so nobody could mistake them for real news: they name no events,
    make no claims, and say "synthetic" in the publisher field.
    """
    from .fundamentals import Headline

    rng = np.random.default_rng(symbol_seed(symbol) + 11)
    templates = [
        "{s} shares in focus as quarterly reporting season approaches",
        "Analysts update coverage of {s}",
        "{s} trading volume above recent averages",
        "What {s} investors are watching this week",
        "{s} added to a widely followed index screen",
        "Sector peers of {s} move in step",
    ]
    base = pd.Timestamp("2026-09-21")
    chosen = rng.permutation(len(templates))[:limit]
    return [
        Headline(
            title=templates[int(i)].format(s=symbol.upper()),
            publisher="synthetic newswire (not real)",
            published=(base - pd.Timedelta(days=int(offset))).to_pydatetime(),
            url=None,
        )
        for offset, i in enumerate(chosen)
    ]


# --- macro ----------------------------------------------------------------

def _ornstein_uhlenbeck(
    n: int, mean: float, reversion: float, vol: float, rng: np.random.Generator,
    x0: float | None = None,
) -> np.ndarray:
    """Discretized OU path: x_t = x_{t-1} + kappa(mu - x_{t-1}) + sigma eps_t."""
    path = np.empty(n, dtype=np.float64)
    path[0] = mean if x0 is None else x0
    shocks = rng.standard_normal(n) * vol
    for t in range(1, n):
        path[t] = path[t - 1] + reversion * (mean - path[t - 1]) + shocks[t]
    return path


def mock_macro_panel(
    start: date | str, end: date | str, seed: int = 20260921
) -> dict[str, pd.Series]:
    """Synthetic versions of the five FRED series, at their real frequencies.

    Ported from `quant_pipeline/mock_data/macro.py` and trimmed: the
    research tool needs plausible levels and plausible *surprises*, not the
    full Taylor-rule machinery the forecasting pipeline used.

    The series are not independent. The short rate responds to inflation and
    slack, the curve flattens when policy tightens, and credit spreads widen
    with unemployment -- so a report that says "the 2-year rose and credit
    spreads widened on the same day" describes something the generator
    actually linked, rather than two unrelated random walks.

    Returns a mapping from FRED series ID to a Series dated by *reference
    period*, exactly as FRED dates them. Applying the publication lag is
    `macro_ingest.align_to_daily`'s job.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(start=start, end=end, freq="MS")
    bdays = pd.bdate_range(start=start, end=end, name="date")
    if len(months) < 3 or len(bdays) < 30:
        raise ValueError(
            f"need at least 3 months and 30 business days between {start} and {end}; "
            f"got {len(months)} and {len(bdays)}"
        )

    n = len(months)
    persistence, target = 0.72, 0.0021  # ~2.5% annualized inflation
    inflation = np.empty(n)
    inflation[0] = target
    innovations = rng.standard_normal(n) * 0.0011
    for t in range(1, n):
        inflation[t] = target + persistence * (inflation[t - 1] - target) + innovations[t]
    cpi = pd.Series(300.0 * np.exp(np.cumsum(inflation)), index=months, name="CPIAUCSL")

    unemployment = pd.Series(
        np.round(np.clip(_ornstein_uhlenbeck(n, 4.2, 0.06, 0.13, rng, x0=4.1), 3.2, 11.0), 1),
        index=months, name="UNRATE",
    )

    yoy = ((cpi / cpi.shift(12) - 1.0) * 100.0)
    yoy_daily = yoy.reindex(bdays.union(months)).ffill().reindex(bdays).bfill()
    unrate_daily = unemployment.reindex(bdays.union(months)).ffill().reindex(bdays).bfill()

    policy = np.clip(
        2.0 + 1.4 * (yoy_daily.to_numpy() - 2.0) + 0.5 * (4.2 - unrate_daily.to_numpy()), 0.0, 9.0
    )
    smoothed = pd.Series(policy, index=bdays).ewm(span=90, adjust=False).mean().to_numpy()
    dgs2 = np.clip(smoothed + _ornstein_uhlenbeck(len(bdays), 0.0, 0.03, 0.045, rng), 0.02, None)
    slope = _ornstein_uhlenbeck(len(bdays), 1.05, 0.01, 0.035, rng) - 0.55 * np.clip(
        smoothed - 2.0, 0.0, None
    )
    dgs10 = np.clip(dgs2 + slope, 0.05, None)
    credit = np.exp(
        _ornstein_uhlenbeck(len(bdays), np.log(3.6), 0.012, 0.028, rng)
        + 0.30 * (unrate_daily.to_numpy() - 4.2)
    )

    return {
        "CPIAUCSL": cpi,
        "UNRATE": unemployment,
        "DGS2": pd.Series(np.round(dgs2, 2), index=bdays, name="DGS2"),
        "DGS10": pd.Series(np.round(dgs10, 2), index=bdays, name="DGS10"),
        "BAMLH0A0HYM2": pd.Series(np.round(credit, 2), index=bdays, name="BAMLH0A0HYM2"),
    }
