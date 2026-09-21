"""Assemble the evidence for "is X good to invest" -- without answering it.

The question cannot be answered by a tool. Whether something suits you
depends on your time horizon, what else you own, your tax position and how
you would behave if it halved -- none of which this program knows. What it
can do is lay out what the company earns, what it costs relative to its
peers, how bumpy it has been, and what the figures do *not* cover.

So the bundle is built to make a reader think rather than to reach a
verdict. Three consequences:

**Every metric is paired with its peer group.** A P/E of 39 means nothing
alone. Against a sector median of 26 it means the market expects faster
growth from this company than from its peers -- which is a fact about
expectations, not a judgement about value.

**Absence is reported, not filled.** A loss-making company has no
meaningful P/E; an ETF has no margins at all. Those come back as missing
with a reason, never as zero.

**Counterevidence is mandatory.** Whatever the figures look like, the
bundle carries the things that cut the other way: a high multiple means a
lot is already priced in, a high yield can signal a market that doubts the
dividend, one year of strong returns is not a forecast.

Peer groups are a static, hand-maintained map of large liquid names per
sector. That is a real limitation -- a small-cap industrial compared against
mega-cap industrials will look cheap on every measure for reasons that have
nothing to do with value -- and it is recorded in the bundle's limitations
rather than hidden.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..data.fundamentals import RawFundamentals, fetch_fundamentals, fetch_headlines
from ..data.price_ingest import PriceDataError, Source, fetch_history
from ..query_parser import ParsedQuery
from .schema import (
    Counterevidence,
    EvidenceBundle,
    NewsItem,
    PeerComparison,
    PriceTrend,
    SnapshotFundamentals,
)

__all__ = ["METRIC_LABELS", "SECTOR_PEERS", "build_snapshot_bundle"]

#: Representative large, liquid names per sector. Static and hand-maintained:
#: a real peer screen would match on size, geography and business mix, which
#: needs a fundamentals database this tool does not have. The consequence --
#: a small-cap looks cheap against mega-cap peers for reasons unrelated to
#: value -- is recorded in every bundle's limitations.
SECTOR_PEERS: dict[str, tuple[str, ...]] = {
    "Technology": ("AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM"),
    "Communication Services": ("GOOGL", "META", "NFLX", "DIS", "T", "VZ"),
    "Consumer Cyclical": ("AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"),
    "Consumer Defensive": ("PG", "KO", "PEP", "COST", "WMT", "CL"),
    "Healthcare": ("JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK"),
    "Financial Services": ("JPM", "BAC", "WFC", "GS", "MS", "BRK-B"),
    "Energy": ("XOM", "CVX", "COP", "SLB", "EOG", "PSX"),
    "Industrials": ("CAT", "HON", "UNP", "GE", "BA", "LMT"),
    "Utilities": ("NEE", "DUK", "SO", "D", "AEP", "EXC"),
    "Real Estate": ("PLD", "AMT", "EQIX", "SPG", "O", "PSA"),
    "Basic Materials": ("LIN", "SHW", "APD", "ECL", "NEM", "FCX"),
}

#: Metrics compared against peers, with how to read them.
METRIC_LABELS: dict[str, tuple[str, str, bool | None]] = {
    "trailing_pe": (
        "Price-to-earnings ratio",
        "what you pay per $1 of last year's profit",
        None,
    ),
    "profit_margin": (
        "Profit margin",
        "the share of revenue left as profit",
        True,
    ),
    "revenue_growth": (
        "Revenue growth",
        "how fast sales grew over the past year",
        True,
    ),
    "dividend_yield": (
        "Dividend yield",
        "cash paid out per year as a share of the price",
        None,
    ),
    "debt_to_equity": (
        "Debt relative to equity",
        "how much of the business is funded by borrowing",
        False,
    ),
    "return_on_equity": (
        "Return on equity",
        "profit generated per $1 shareholders have invested",
        True,
    ),
}

_TRADING_DAYS = 252
_MIN_TREND_SESSIONS = 30


def _price_trend(frame: pd.DataFrame, as_of: date, ticker: str) -> PriceTrend | None:
    """Medium-term price behaviour: returns, volatility, drawdown."""
    usable = frame[frame.index <= pd.Timestamp(as_of)]
    if len(usable) < _MIN_TREND_SESSIONS:
        return None

    closes = usable["Close"].astype(float)
    current = float(closes.iloc[-1])

    def horizon_return(sessions: int) -> float | None:
        if len(closes) <= sessions:
            return None
        return float((current / float(closes.iloc[-1 - sessions]) - 1.0) * 100.0)

    returns = closes.pct_change().dropna()
    window = returns.tail(_TRADING_DAYS)
    annualized_vol = float(window.std(ddof=1) * np.sqrt(_TRADING_DAYS) * 100.0)

    year = closes.tail(_TRADING_DAYS)
    high, low = float(year.max()), float(year.min())
    drawdown = float((current / high - 1.0) * 100.0) if high > 0 else None

    return_1y = horizon_return(_TRADING_DAYS)
    parts = [f"{ticker} trades at {current:,.2f}"]
    if return_1y is not None:
        direction = "up" if return_1y >= 0 else "down"
        parts.append(f"{direction} {abs(return_1y):.1f}% over the past year")
    parts.append(
        f"with day-to-day swings averaging {annualized_vol:.1f}% a year"
    )
    if drawdown is not None and drawdown < -1.0:
        parts.append(f"and sitting {abs(drawdown):.1f}% below its 12-month high")

    return PriceTrend(
        current_price=current,
        return_1m_pct=horizon_return(21),
        return_3m_pct=horizon_return(63),
        return_1y_pct=return_1y,
        annualized_vol_pct=annualized_vol,
        high_52w=high,
        low_52w=low,
        drawdown_from_high_pct=drawdown,
        sessions_available=len(usable),
        summary=", ".join(parts) + ".",
    )


def _to_snapshot_fundamentals(symbol: str, raw: RawFundamentals) -> SnapshotFundamentals:
    return SnapshotFundamentals(
        symbol=symbol,
        company_name=raw.company_name,
        sector=raw.sector,
        market_cap=raw.get("market_cap"),
        trailing_pe=raw.get("trailing_pe"),
        forward_pe=raw.get("forward_pe"),
        revenue_growth=raw.get("revenue_growth"),
        profit_margin=raw.get("profit_margin"),
        debt_to_equity=raw.get("debt_to_equity"),
        return_on_equity=raw.get("return_on_equity"),
        dividend_yield=raw.get("dividend_yield"),
        beta=raw.get("beta"),
        missing_fields=list(raw.missing_fields),
    )


def _peer_comparisons(
    symbol: str, subject: RawFundamentals, sector: str | None, source: Source
) -> tuple[list[PeerComparison], list[str], list[str]]:
    """Compare the subject's metrics against its sector peers' medians."""
    if not sector or sector not in SECTOR_PEERS:
        return [], [], (
            [f"No peer group is defined for sector {sector!r}, so no comparison is shown."]
            if sector
            else ["No sector was reported for this symbol, so no peer comparison is shown."]
        )

    peers = [p for p in SECTOR_PEERS[sector] if p.upper() != symbol.upper()]
    fetched: dict[str, RawFundamentals] = {}
    for peer in peers:
        try:
            fetched[peer] = fetch_fundamentals(peer, source=source)
        except Exception:  # noqa: BLE001, S112 - one bad peer must not cost the section
            continue

    if len(fetched) < 3:
        return [], [], [
            (
                f"Only {len(fetched)} peers could be fetched for {sector}; too few for "
                "a meaningful comparison, so none is shown."
            )
        ]

    comparisons: list[PeerComparison] = []
    for metric, (label, meaning, higher_better) in METRIC_LABELS.items():
        subject_value = subject.get(metric)
        peer_values = {
            name: value
            for name, data in fetched.items()
            if (value := data.get(metric)) is not None and np.isfinite(value)
        }
        if len(peer_values) < 3:
            continue
        comparisons.append(
            PeerComparison(
                symbol=symbol,
                label=label,
                metric=metric,
                value=subject_value,
                subject_value=subject_value,
                peer_median=float(np.median(list(peer_values.values()))),
                peer_values=peer_values,
                higher_is_better=higher_better,
                note=meaning,
            )
        )

    notes = [
        (
            f"Peers are a fixed list of large {sector} companies "
            f"({', '.join(sorted(fetched))}), not a screen matched on size or business "
            "mix. A company much smaller than these will look different for reasons "
            "unrelated to how good a business it is."
        )
    ]
    return comparisons, notes, []


def _counterevidence(
    symbol: str,
    raw: RawFundamentals,
    trend: PriceTrend | None,
    comparisons: list[PeerComparison],
) -> list[Counterevidence]:
    """The things that cut against whatever the headline figures suggest."""
    items: list[Counterevidence] = []

    if raw.is_fund:
        # Without this a fund produces an empty counterevidence section,
        # which reads as "nothing to worry about" -- the opposite of what an
        # absence of company data should convey.
        items.append(
            Counterevidence(
                label="There is no company here to analyse",
                detail=(
                    f"{symbol} is a fund holding many companies at once, so the usual "
                    "checks -- is it profitable, is it growing, is it cheap -- have no "
                    "single answer. What you are buying is the average of whatever it "
                    "holds, and its ups and downs will track that market rather than any "
                    "one business."
                ),
            )
        )
        items.append(
            Counterevidence(
                label="Fund costs are not shown here",
                detail=(
                    "Every fund charges an annual fee that comes out of your return, and "
                    "this tool does not retrieve it. Look up the expense ratio before "
                    "comparing one fund with another; over decades a small difference "
                    "compounds into a large one."
                ),
            )
        )

    margin = raw.get("profit_margin")
    if margin is not None and margin <= 0:
        items.append(
            Counterevidence(
                label="This company is not currently profitable",
                detail=(
                    f"Its profit margin is {margin:.1%}, meaning it spends more than it "
                    "earns. Loss-making companies depend on future growth or on raising "
                    "money, and a price-to-earnings ratio cannot be read in the usual way."
                ),
            )
        )

    pe = raw.get("trailing_pe")
    pe_comparison = next((c for c in comparisons if c.metric == "trailing_pe"), None)
    if (
        pe is not None
        and pe > 0
        and pe_comparison
        and pe_comparison.vs_median_pct is not None
        and pe_comparison.vs_median_pct > 25
    ):
        items.append(
            Counterevidence(
                label="A lot of future growth is already priced in",
                detail=(
                    f"At {pe:.1f} times earnings against a peer median of "
                    f"{pe_comparison.peer_median:.1f}, buyers are paying "
                    f"{pe_comparison.vs_median_pct:.0f}% more per dollar of profit "
                    "than for the average peer. That only works out if the company "
                    "grows into it; if growth disappoints, the multiple can fall even "
                    "when profits do not."
                ),
            )
        )

    yield_ = raw.get("dividend_yield")
    if yield_ is None:
        items.append(
            Counterevidence(
                label="No dividend is paid",
                detail=(
                    f"{symbol} does not currently pay a dividend, so any return has to "
                    "come from the share price rising. If you are looking for income, "
                    "this is not a source of it."
                ),
            )
        )
    elif yield_ > 0.06:
        items.append(
            Counterevidence(
                label="An unusually high dividend yield can be a warning",
                detail=(
                    f"The yield is {yield_:.1%}. A yield that high often means the share "
                    "price has fallen because the market doubts the payout will continue, "
                    "rather than that the company is unusually generous."
                ),
            )
        )

    debt = raw.get("debt_to_equity")
    if debt is not None and debt > 2.0:
        items.append(
            Counterevidence(
                label="The company carries a lot of debt",
                detail=(
                    f"Borrowings are about {debt:.1f} times shareholders' equity. Debt "
                    "magnifies both good and bad years, and makes a business more "
                    "sensitive to interest rates."
                ),
            )
        )

    if trend is not None:
        if trend.return_1y_pct is not None and trend.return_1y_pct > 50:
            items.append(
                Counterevidence(
                    label="Recent performance is not a forecast",
                    detail=(
                        f"The price is up {trend.return_1y_pct:.0f}% over the past year. "
                        "Strong past returns are the most common reason people buy and "
                        "one of the weakest reasons to: the return you get depends on "
                        "what happens next, not what already has."
                    ),
                )
            )
        if trend.is_more_volatile_than_market:
            items.append(
                Counterevidence(
                    label="This moves more than the overall market",
                    detail=(
                        f"Day-to-day swings average {trend.annualized_vol_pct:.0f}% a year, "
                        "against roughly the mid-teens for a broad index fund. Expect "
                        "larger falls as well as larger rises."
                    ),
                )
            )
    return items


def build_snapshot_bundle(
    parsed: ParsedQuery,
    source: Source = "auto",
    include_news: bool = True,
) -> EvidenceBundle:
    """Build the evidence bundle for an "is X worth investing in" question.

    Raises `PriceDataError` only when the symbol's price history cannot be
    obtained at all. Fundamentals, peers and news are each optional; losing
    one costs a section and a recorded limitation, not the report.
    """
    symbol = parsed.yf_symbol
    as_of = parsed.end
    warnings = list(parsed.warnings)
    limitations: list[str] = []
    data_sources: list[str] = []

    frame, price_source, price_warnings = fetch_history(
        symbol, as_of - timedelta(days=500), as_of, source=source
    )
    warnings.extend(price_warnings)
    data_sources.append(
        f"Prices: {'Yahoo Finance' if price_source == 'yfinance' else 'synthetic'}"
    )
    trend = _price_trend(frame, as_of, parsed.ticker)
    if trend is None:
        raise PriceDataError(
            f"not enough price history for {parsed.ticker} to describe a trend",
            suggestion="This may be a very newly listed symbol. Try a more established one.",
        )

    raw = fetch_fundamentals(symbol, source=source)
    data_sources.append(
        f"Fundamentals: {'Yahoo Finance' if raw.source == 'yfinance' else 'synthetic'}"
    )
    if raw.source == "mock" and source != "mock":
        warnings.append(
            f"live fundamentals were unavailable for {parsed.ticker}; the figures below "
            "are synthetic and NOT real"
        )
    for name, reason in raw.suspicious_fields.items():
        warnings.append(f"{name} looks wrong and was not used: {reason}")

    if raw.is_fund:
        limitations.append(
            f"{parsed.ticker} is a fund, not a company. It has no revenue, margins or "
            "debt of its own -- it holds many companies at once -- so most of the "
            "company measures in this report do not apply to it."
        )

    comparisons, peer_notes, peer_problems = _peer_comparisons(
        parsed.ticker, raw, raw.sector, source
    )
    limitations.extend(peer_notes)
    limitations.extend(peer_problems)

    news: list[NewsItem] = []
    if include_news:
        for headline in fetch_headlines(symbol, source=source):
            news.append(
                NewsItem(
                    title=headline.title,
                    publisher=headline.publisher,
                    published=headline.published.date() if headline.published else None,
                    url=headline.url,
                )
            )
        if news:
            data_sources.append("Headlines: Yahoo Finance")
            limitations.append(
                "Headlines are listed as titles only. This tool has not read the "
                "articles and cannot tell you whether any of them matters. The feed is "
                f"also only loosely tied to {parsed.ticker} -- it often carries general "
                "market stories that mention other companies -- so check that a headline "
                "is actually about this one before reading anything into it."
            )

    if raw.missing_fields:
        limitations.append(
            f"{len(raw.missing_fields)} measure(s) were not published for "
            f"{parsed.ticker}: {', '.join(sorted(raw.missing_fields))}. They are shown "
            "as unavailable rather than filled in."
        )
    limitations.append(
        "Everything above describes the past and the present. None of it forecasts "
        "returns, and none of it accounts for your own situation."
    )

    return EvidenceBundle(
        query=parsed.raw_query,
        question_type="snapshot",
        ticker=parsed.ticker,
        period_description=f"as of {trend.sessions_available} sessions to {as_of.isoformat()}",
        fundamentals=_to_snapshot_fundamentals(parsed.ticker, raw),
        peers=comparisons,
        trend=trend,
        news=news,
        counterevidence=_counterevidence(parsed.ticker, raw, trend, comparisons),
        data_sources=data_sources,
        limitations=limitations,
        warnings=warnings,
        is_synthetic=price_source == "mock" or raw.source == "mock",
    )
