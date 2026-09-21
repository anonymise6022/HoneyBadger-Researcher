"""Company fundamentals and recent headlines, normalized at the boundary.

**The units in yfinance's `.info` are not consistent with each other**, and
getting that wrong is the kind of bug that destroys trust in a tool aimed at
beginners. Measured against live data:

    dividendYield   PERCENT     KO returns 2.4, meaning 2.4%
    revenueGrowth   FRACTION    AAPL returns 0.164, meaning 16.4%
    profitMargins   FRACTION    AAPL returns 0.276, meaning 27.6%
    debtToEquity    PERCENT     AAPL returns 78.4, meaning 0.78x

Formatting a dividend yield of 2.4 with a percent formatter prints "240%".
So every field is converted to a **fraction** here, once, at the edge, and
everything downstream can format uniformly. Implausible values survive the
conversion as `suspicious_fields` rather than being silently clamped -- if
the vendor changes convention again, the report says the number looks wrong
instead of quietly printing a lie.

**ETFs have almost no fundamentals.** A stock returns 17 of the 18 fields
this module looks for; SPY returns 5, with no sector, no margins and no
revenue growth, because an index fund has no income statement. That is not
an error, and the snapshot must say "this is a fund, these questions do not
apply to it" rather than printing a column of "n/a".

Headlines come from `yfinance`'s `.news`, which returns Yahoo-syndicated
items. Titles and timestamps only -- no summaries are passed downstream,
because a summary is prose this tool has not verified and the report's whole
claim is that everything in it traces to a checked field.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .mock_data import mock_fundamentals, mock_headlines

__all__ = [
    "FUNDAMENTAL_FIELDS",
    "Headline",
    "RawFundamentals",
    "fetch_fundamentals",
    "fetch_headlines",
    "normalize_info",
]

#: yfinance `.info` keys -> (our name, whether the vendor reports a percent).
FUNDAMENTAL_FIELDS: dict[str, tuple[str, bool]] = {
    "trailingPE": ("trailing_pe", False),
    "forwardPE": ("forward_pe", False),
    "revenueGrowth": ("revenue_growth", False),
    "profitMargins": ("profit_margin", False),
    "returnOnEquity": ("return_on_equity", False),
    "debtToEquity": ("debt_to_equity", True),
    "dividendYield": ("dividend_yield", True),
    "beta": ("beta", False),
    "marketCap": ("market_cap", False),
    "priceToBook": ("price_to_book", False),
}

#: Ceilings past which a value is almost certainly a units error rather than
#: a remarkable company. A 60% dividend yield or a 5000% margin means the
#: vendor changed convention.
_PLAUSIBLE_CEILING: dict[str, float] = {
    "dividend_yield": 0.60,
    "profit_margin": 5.0,
    "revenue_growth": 50.0,
    "return_on_equity": 50.0,
}

_MAX_HEADLINES = 6


@dataclass(frozen=True)
class Headline:
    """One recent news item. Title and source only -- never a summary."""

    title: str
    publisher: str
    published: datetime | None
    url: str | None = None


@dataclass
class RawFundamentals:
    """Normalized fundamentals plus an honest account of what is missing.

    Attributes
    ----------
    values : field name -> value, all as fractions where proportional.
    company_name, sector, industry : descriptive fields, or None.
    quote_type : "EQUITY", "ETF", "CURRENCY"... Used to decide whether the
        absence of fundamentals is a data gap or simply the wrong question.
    missing_fields : requested fields the vendor did not supply.
    suspicious_fields : fields whose value fails a plausibility check, with
        the reason. Reported, never silently corrected.
    source : "yfinance" or "mock".
    """

    values: dict[str, float] = field(default_factory=dict)
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    quote_type: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    suspicious_fields: dict[str, str] = field(default_factory=dict)
    source: str = "yfinance"

    @property
    def is_fund(self) -> bool:
        """Whether company fundamentals are simply the wrong question here."""
        return (self.quote_type or "").upper() in {"ETF", "MUTUALFUND", "INDEX", "CURRENCY"}

    def get(self, name: str) -> float | None:
        return self.values.get(name)


def normalize_info(info: dict[str, Any], symbol: str, source: str = "yfinance") -> RawFundamentals:
    """Convert a vendor `.info` mapping into normalized fractions.

    Pure: no network, so the unit handling can be tested against a recorded
    payload rather than against whatever the vendor returns today.
    """
    result = RawFundamentals(source=source)
    result.company_name = info.get("longName") or info.get("shortName")
    result.sector = info.get("sector")
    result.industry = info.get("industry")
    result.quote_type = info.get("quoteType")

    for vendor_key, (name, is_percent) in FUNDAMENTAL_FIELDS.items():
        raw = info.get(vendor_key)
        if raw is None:
            result.missing_fields.append(name)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            result.missing_fields.append(name)
            continue
        if math.isnan(value):
            result.missing_fields.append(name)
            continue

        result.values[name] = value / 100.0 if is_percent else value

        ceiling = _PLAUSIBLE_CEILING.get(name)
        if ceiling is not None and abs(result.values[name]) > ceiling:
            result.suspicious_fields[name] = (
                f"{result.values[name]:.2f} is outside the plausible range "
                f"(|value| <= {ceiling:g}); the data source may have changed units"
            )
    return result


def fetch_fundamentals(symbol: str, source: str = "auto") -> RawFundamentals:
    """Fetch and normalize fundamentals for one symbol.

    `source="mock"` never touches the network. `"auto"` falls back to mock
    with the source recorded, so a report can say the numbers are synthetic.
    `"live"` raises on failure.
    """
    if source not in ("auto", "live", "mock"):
        raise ValueError(f"source must be 'auto', 'live' or 'mock', got {source!r}")
    if source == "mock":
        return normalize_info(mock_fundamentals(symbol), symbol, source="mock")

    try:
        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = yf.Ticker(symbol).info
        if not info or len(info) < 5:
            raise ValueError(f"vendor returned {len(info or {})} fields for {symbol}")
    except Exception as exc:
        if source == "live":
            raise
        result = normalize_info(mock_fundamentals(symbol), symbol, source="mock")
        result.suspicious_fields["_fetch"] = f"live fundamentals unavailable ({exc})"
        return result
    return normalize_info(info, symbol, source="yfinance")


def fetch_headlines(symbol: str, source: str = "auto", limit: int = _MAX_HEADLINES) -> list[Headline]:
    """Recent headlines for a symbol, titles and sources only.

    Returns an empty list on any failure. News is genuinely optional context
    -- losing it costs a section, not the report -- and headline feeds are the
    least reliable thing this tool touches.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if source == "mock":
        return mock_headlines(symbol, limit)

    try:
        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            items = yf.Ticker(symbol).news or []
    except Exception:  # noqa: BLE001 - news is optional context
        return mock_headlines(symbol, limit) if source == "auto" else []

    headlines: list[Headline] = []
    for item in items[:limit]:
        content = item.get("content") or item
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider") or {}
        published = None
        raw_date = content.get("pubDate") or content.get("displayTime")
        if isinstance(raw_date, str):
            try:
                published = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except ValueError:
                published = None
        url = (content.get("canonicalUrl") or {}).get("url") if isinstance(
            content.get("canonicalUrl"), dict
        ) else None
        headlines.append(
            Headline(
                title=str(title),
                publisher=str(provider.get("displayName") or "unknown"),
                published=published,
                url=url,
            )
        )
    return headlines
