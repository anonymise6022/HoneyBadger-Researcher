"""Find a ticker from a company name, so people can type "apple" not "AAPL".

Yahoo's search endpoint does the work. Two things wrap it.

**A local table is tried first.** The common cases -- the mega-caps, the
index ETFs, the major currency pairs -- resolve instantly and work offline,
which matters because this sits behind a search box where a round trip per
keystroke would feel broken.

**Results are filtered and ranked.** A raw search for "apple" returns stock
futures, a Canadian depositary receipt and a 2x leveraged ETF above the
actual company. Futures and warrants are dropped, ordinary equities and
funds are preferred over derivatives, and exact ticker matches are promoted
to the top -- so typing "F" finds Ford rather than a fund whose description
happens to contain the letter.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

__all__ = ["SymbolMatch", "search_symbols"]

_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
_TIMEOUT = 8.0
_HEADERS = {"User-Agent": "Mozilla/5.0"}

#: Types worth offering. Futures, warrants and options chains are noise for
#: a tool that charts and backtests a single underlying.
_USEFUL_TYPES = {"EQUITY", "ETF", "INDEX", "CURRENCY", "MUTUALFUND", "CRYPTOCURRENCY"}

_TYPE_RANK = {
    "EQUITY": 0, "ETF": 1, "INDEX": 2, "CURRENCY": 2,
    "CRYPTOCURRENCY": 3, "MUTUALFUND": 4,
}

#: Resolved without a network call. Covers most of what gets typed.
_LOCAL: dict[str, tuple[str, str, str]] = {
    "apple": ("AAPL", "Apple Inc.", "EQUITY"),
    "microsoft": ("MSFT", "Microsoft Corporation", "EQUITY"),
    "nvidia": ("NVDA", "NVIDIA Corporation", "EQUITY"),
    "amazon": ("AMZN", "Amazon.com, Inc.", "EQUITY"),
    "google": ("GOOGL", "Alphabet Inc.", "EQUITY"),
    "alphabet": ("GOOGL", "Alphabet Inc.", "EQUITY"),
    "meta": ("META", "Meta Platforms, Inc.", "EQUITY"),
    "facebook": ("META", "Meta Platforms, Inc.", "EQUITY"),
    "tesla": ("TSLA", "Tesla, Inc.", "EQUITY"),
    "netflix": ("NFLX", "Netflix, Inc.", "EQUITY"),
    "coca cola": ("KO", "The Coca-Cola Company", "EQUITY"),
    "coke": ("KO", "The Coca-Cola Company", "EQUITY"),
    "ford": ("F", "Ford Motor Company", "EQUITY"),
    "gamestop": ("GME", "GameStop Corp.", "EQUITY"),
    "palantir": ("PLTR", "Palantir Technologies Inc.", "EQUITY"),
    "berkshire": ("BRK-B", "Berkshire Hathaway Inc.", "EQUITY"),
    "johnson": ("JNJ", "Johnson & Johnson", "EQUITY"),
    "exxon": ("XOM", "Exxon Mobil Corporation", "EQUITY"),
    "jpmorgan": ("JPM", "JPMorgan Chase & Co.", "EQUITY"),
    "s&p 500": ("SPY", "SPDR S&P 500 ETF Trust", "ETF"),
    "sp500": ("SPY", "SPDR S&P 500 ETF Trust", "ETF"),
    "the market": ("SPY", "SPDR S&P 500 ETF Trust", "ETF"),
    "nasdaq": ("QQQ", "Invesco QQQ Trust", "ETF"),
    "dow": ("DIA", "SPDR Dow Jones Industrial Average ETF", "ETF"),
    "russell": ("IWM", "iShares Russell 2000 ETF", "ETF"),
    "gold": ("GLD", "SPDR Gold Shares", "ETF"),
    "oil": ("USO", "United States Oil Fund", "ETF"),
    "bitcoin": ("BTC-USD", "Bitcoin USD", "CRYPTOCURRENCY"),
    "ethereum": ("ETH-USD", "Ethereum USD", "CRYPTOCURRENCY"),
    "vix": ("^VIX", "CBOE Volatility Index", "INDEX"),
    "euro": ("EURUSD=X", "EUR/USD", "CURRENCY"),
    "eurusd": ("EURUSD=X", "EUR/USD", "CURRENCY"),
    "pound": ("GBPUSD=X", "GBP/USD", "CURRENCY"),
    "yen": ("USDJPY=X", "USD/JPY", "CURRENCY"),
}


@dataclass(frozen=True)
class SymbolMatch:
    """One search result."""

    symbol: str
    name: str
    kind: str
    exchange: str = ""

    @property
    def label(self) -> str:
        return f"{self.symbol} — {self.name}" if self.name else self.symbol


def _local_matches(query: str) -> list[SymbolMatch]:
    lowered = query.strip().lower()
    if not lowered:
        return []
    hits = [
        SymbolMatch(symbol, name, kind)
        for key, (symbol, name, kind) in _LOCAL.items()
        if key.startswith(lowered) or lowered == symbol.lower()
    ]
    # Deduplicate by symbol, keeping the first match for each.
    seen: dict[str, SymbolMatch] = {}
    for hit in hits:
        seen.setdefault(hit.symbol, hit)
    # An exact ticker always outranks a name that merely starts with the same
    # letters: typing "F" means Ford, not Facebook.
    return sorted(seen.values(), key=lambda m: 0 if m.symbol.lower() == lowered else 1)


def search_symbols(query: str, limit: int = 8, offline: bool = False) -> list[SymbolMatch]:
    """Find symbols matching a name or ticker, best first.

    Never raises: a search box that throws on a network hiccup is worse than
    one that quietly returns the local matches it already has.
    """
    text = (query or "").strip()
    if not text:
        return []
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

    results = _local_matches(text)
    if offline:
        return results[:limit]

    try:
        response = requests.get(
            _SEARCH_URL,
            params={"q": text, "quotesCount": limit * 3, "newsCount": 0},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
    except (requests.RequestException, ValueError):
        return results[:limit]

    known = {match.symbol for match in results}
    remote: list[SymbolMatch] = []
    for quote in quotes:
        symbol = quote.get("symbol")
        kind = (quote.get("quoteType") or "").upper()
        if not symbol or kind not in _USEFUL_TYPES or symbol in known:
            continue
        known.add(symbol)
        remote.append(
            SymbolMatch(
                symbol=symbol,
                name=quote.get("shortname") or quote.get("longname") or "",
                kind=kind,
                exchange=quote.get("exchDisp") or quote.get("exchange") or "",
            )
        )

    upper = text.upper()
    remote.sort(
        key=lambda m: (
            # An exact ticker match is almost always what was meant.
            0 if m.symbol == upper else 1,
            _TYPE_RANK.get(m.kind, 9),
            # Prefer plain tickers over class shares and foreign listings.
            len(m.symbol),
        )
    )
    return (results + remote)[:limit]
