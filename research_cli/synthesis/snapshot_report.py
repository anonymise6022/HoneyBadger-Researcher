"""Render the "is X worth investing in" bundle, without answering the question.

The hard part of this report is not the numbers, it is the ending. A note
that lays out a company's figures and then closes with a generic "this is
not financial advice" has, in practice, given advice -- the reader takes the
tone of the numbers as the verdict and reads the disclaimer as legal
throat-clearing.

So the closing section here is **generated from the specific company**. It
names the actual tension in the actual figures ("whether 39 times earnings
is sensible depends on whether you think Apple grows into it, which these
numbers cannot tell you"), states what this tool does not know about the
reader, and ends without a recommendation. `validator` enforces that no
verdict vocabulary appears anywhere; this module's job is to make the
absence of a verdict feel like an answer rather than an evasion.

The section order mirrors how a careful person would actually work through
it: what is this thing, what does it earn, is that normal for its industry,
what has the price done, what is in the news, and -- last and longest --
what none of that covers.
"""

from __future__ import annotations

from ..evidence.schema import EvidenceBundle, PeerComparison

__all__ = ["SNAPSHOT_SECTIONS", "render_snapshot_report"]

SNAPSHOT_SECTIONS = (
    "QUESTION",
    "WHAT YOU ARE LOOKING AT",
    "THE NUMBERS",
    "COMPARED WITH SIMILAR COMPANIES",
    "PRICE HISTORY",
    "WHAT THIS DOES NOT TELL YOU",
    "RAW DATA",
)

_WIDTH = 78


def _rule(char: str = "=") -> str:
    return char * _WIDTH


def _heading(title: str) -> str:
    return f"\n{title}\n{_rule('-')}"


def _wrap(text: str, indent: str = "  ") -> str:
    # A bullet marker belongs on the first line only. Reusing the full
    # indent for continuations prints "- " down the left margin of every
    # wrapped line, which reads as a list of fragments rather than one item.
    continuation = " " * len(indent)
    lines: list[str] = []
    current = indent
    for word in text.split():
        if current.strip() and len(current) + len(word) > _WIDTH:
            lines.append(current.rstrip())
            current = f"{continuation}{word} "
        else:
            current = f"{current}{word} "
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def _money(value: float | None) -> str:
    """Format a market capitalisation the way a person would say it."""
    if value is None:
        return "not available"
    for threshold, suffix in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.1f} {suffix}"
    return f"${value:,.0f}"


def _percent(value: float | None, decimals: int = 1) -> str:
    return "not available" if value is None else f"{value * 100:.{decimals}f}%"


def _ratio(value: float | None, decimals: int = 1) -> str:
    return "not available" if value is None else f"{value:.{decimals}f}"


def _render_question(bundle: EvidenceBundle) -> str:
    lines = [_heading("QUESTION"), f'  "{bundle.query}"']
    lines.append(
        _wrap(
            "This note lays out what can be measured about "
            f"{bundle.ticker}. It does not answer whether you should own it, "
            "and the last section explains why that is not a dodge."
        )
    )
    for warning in bundle.warnings:
        lines.append(_wrap(f"Note: {warning}"))
    return "\n".join(lines)


def _render_identity(bundle: EvidenceBundle) -> str:
    fundamentals = bundle.fundamentals
    lines = [_heading("WHAT YOU ARE LOOKING AT")]
    if fundamentals is None:
        lines.append("  No company information was available.")
        return "\n".join(lines)

    name = fundamentals.company_name or bundle.ticker
    if fundamentals.sector:
        lines.append(
            _wrap(
                f"{name} ({bundle.ticker}) is a {fundamentals.sector.lower()} company "
                f"with a market value of {_money(fundamentals.market_cap)}. Market value "
                "is what the whole company is worth at today's share price."
            )
        )
    else:
        lines.append(
            _wrap(
                f"{name} ({bundle.ticker}). No sector was reported for this symbol, which "
                "usually means it is a fund rather than a single company."
            )
        )
    if fundamentals.is_profitable is False:
        lines.append("")
        lines.append(
            _wrap(
                "This company currently loses money. That is not automatically bad -- many "
                "growing companies do -- but it changes how every other number below "
                "should be read."
            )
        )
    return "\n".join(lines)


def _render_numbers(bundle: EvidenceBundle) -> str:
    fundamentals = bundle.fundamentals
    lines = [_heading("THE NUMBERS")]
    if fundamentals is None:
        lines.append("  No fundamentals were available.")
        return "\n".join(lines)

    rows = [
        (
            "Price-to-earnings",
            _ratio(fundamentals.trailing_pe),
            "what you pay for each $1 of last year's profit",
        ),
        (
            "Revenue growth",
            _percent(fundamentals.revenue_growth),
            "how much sales grew over the past year",
        ),
        (
            "Profit margin",
            _percent(fundamentals.profit_margin),
            "how much of each sales dollar is left as profit",
        ),
        (
            "Return on equity",
            _percent(fundamentals.return_on_equity),
            "profit made per $1 that shareholders have put in",
        ),
        (
            "Debt vs equity",
            _ratio(fundamentals.debt_to_equity, 2),
            (
                "borrowings compared with shareholders' money; above 1 means more "
                "debt than equity"
            ),
        ),
        (
            "Dividend yield",
            _percent(fundamentals.dividend_yield, 2),
            "cash paid to shareholders each year, as a share of the price",
        ),
    ]
    for label, value, meaning in rows:
        lines.append(f"  {label:<20} {value}")
        lines.append(_wrap(meaning, indent="      "))
    if fundamentals.missing_fields:
        lines.append("")
        lines.append(
            _wrap(
                "Shown as not available -- the data source did not publish them: "
                f"{', '.join(sorted(fundamentals.missing_fields))}."
            )
        )
    return "\n".join(lines)


def _peer_line(comparison: PeerComparison) -> str:
    """One peer row, with the comparison read out in words."""
    if comparison.subject_value is None:
        return _wrap(
            f"{comparison.label}: not published for {comparison.symbol}, so no "
            "comparison is possible.",
            indent="  - ",
        )

    is_rate = comparison.metric in {
        "profit_margin", "revenue_growth", "dividend_yield", "return_on_equity",
    }
    subject = _percent(comparison.subject_value) if is_rate else _ratio(comparison.subject_value, 2)
    median = _percent(comparison.peer_median) if is_rate else _ratio(comparison.peer_median, 2)

    text = f"{comparison.label}: {subject} for {comparison.symbol}, against {median} "
    text += f"for the typical peer ({comparison.note})."
    if comparison.vs_median_pct is not None:
        direction = "higher" if comparison.vs_median_pct > 0 else "lower"
        text += f" That is {abs(comparison.vs_median_pct):.0f}% {direction} than the peer median."
    return _wrap(text, indent="  - ")


def _render_peers(bundle: EvidenceBundle) -> str:
    lines = [_heading("COMPARED WITH SIMILAR COMPANIES")]
    if not bundle.peers:
        lines.append(
            _wrap(
                "No peer comparison is available for this symbol. A number on its own is "
                "hard to judge, so treat the figures above with extra caution."
            )
        )
        return "\n".join(lines)

    lines.append(
        _wrap(
            "A number means little by itself. These compare the same measures against "
            "other large companies in the same industry."
        )
    )
    lines.append("")
    for comparison in bundle.peers:
        lines.append(_peer_line(comparison))
    return "\n".join(lines)


def _render_price(bundle: EvidenceBundle) -> str:
    trend = bundle.trend
    lines = [_heading("PRICE HISTORY")]
    if trend is None:
        lines.append("  No price history was available.")
        return "\n".join(lines)

    lines.append(_wrap(trend.summary))
    lines.append("")
    for label, value in (
        ("Past month", trend.return_1m_pct),
        ("Past three months", trend.return_3m_pct),
        ("Past year", trend.return_1y_pct),
    ):
        if value is not None:
            lines.append(f"  {label:<20} {value:+.1f}%")
    if trend.high_52w is not None and trend.low_52w is not None:
        lines.append(f"  {'12-month range':<20} {trend.low_52w:,.2f} to {trend.high_52w:,.2f}")
    lines.append("")
    lines.append(
        _wrap(
            "Past returns describe what already happened. They are not a forecast, and "
            "buying after a strong run is the most common way people end up paying a "
            "high price."
        )
    )

    if bundle.news:
        lines.append("")
        lines.append("  RECENT HEADLINES")
        for item in bundle.news:
            date_text = f" ({item.published.isoformat()})" if item.published else ""
            lines.append(_wrap(f"{item.title} -- {item.publisher}{date_text}", indent="     - "))
        lines.append(
            _wrap(
                "Titles only. This tool has not read these articles and cannot tell you "
                "whether any of them matters.",
                indent="     ",
            )
        )
    return "\n".join(lines)


def _closing_statement(bundle: EvidenceBundle) -> str:
    """A closing written from this company's actual figures.

    Deliberately not boilerplate. A generic disclaimer after a page of
    numbers reads as legal throat-clearing and the reader takes the numbers
    as the verdict anyway. Naming the specific unresolved tension is what
    makes "this does not answer your question" land as a real answer.
    """
    fundamentals = bundle.fundamentals
    trend = bundle.trend
    tensions: list[str] = []

    pe_comparison = next((c for c in bundle.peers if c.metric == "trailing_pe"), None)
    if (
        fundamentals is not None
        and fundamentals.trailing_pe is not None
        and fundamentals.trailing_pe > 0
        and pe_comparison is not None
        and pe_comparison.vs_median_pct is not None
    ):
        if pe_comparison.vs_median_pct > 15:
            tensions.append(
                f"whether {bundle.ticker} deserves to cost "
                f"{fundamentals.trailing_pe:.0f} times its earnings when the typical peer "
                f"costs {pe_comparison.peer_median:.0f} times -- that is a judgement about "
                "the future, and nothing above measures the future"
            )
        elif pe_comparison.vs_median_pct < -15:
            tensions.append(
                f"whether {bundle.ticker} is cheap at {fundamentals.trailing_pe:.0f} times "
                f"earnings against a peer median of {pe_comparison.peer_median:.0f}, or "
                "whether the market is pricing in a problem these figures do not show"
            )

    if fundamentals is not None and fundamentals.is_profitable is False:
        tensions.append(
            "whether this company reaches profitability before it needs to raise more "
            "money -- which depends on decisions that have not been made yet"
        )

    if trend is not None and trend.is_more_volatile_than_market:
        tensions.append(
            f"whether you would hold on through a fall, given swings averaging "
            f"{trend.annualized_vol_pct:.0f}% a year -- which is a question about you, "
            "not about the company"
        )

    lines = [_heading("WHAT THIS DOES NOT TELL YOU")]
    if tensions:
        lines.append(_wrap("The figures above leave the real questions open:"))
        lines.append("")
        for tension in tensions:
            lines.append(_wrap(tension, indent="  - "))
        lines.append("")

    if bundle.counterevidence:
        lines.append(_wrap("Things that cut against the headline figures:"))
        lines.append("")
        for item in bundle.counterevidence:
            lines.append(f"  - {item.label}")
            lines.append(_wrap(item.detail, indent="      "))
            lines.append("")

    if bundle.limitations:
        lines.append(_wrap("Limits of this data:"))
        for limitation in bundle.limitations:
            lines.append(_wrap(limitation, indent="  - "))
        lines.append("")

    lines.append(
        _wrap(
            f"This note has not said whether to own {bundle.ticker}, and that is not "
            "caution for its own sake. The same figures point different ways for "
            "different people: how long you plan to hold, what else you already own, "
            "what you would do if it fell by half, and what you need the money for all "
            "change the answer, and none of them appear anywhere above. What is here is "
            "evidence. The decision is a separate thing, and it is yours."
        )
    )
    return "\n".join(lines)


def _render_raw(bundle: EvidenceBundle) -> str:
    lines = [_heading("RAW DATA")]
    fundamentals, trend = bundle.fundamentals, bundle.trend
    if fundamentals is not None:
        for label, value in (
            ("market_cap", fundamentals.market_cap),
            ("trailing_pe", fundamentals.trailing_pe),
            ("forward_pe", fundamentals.forward_pe),
            ("revenue_growth", fundamentals.revenue_growth),
            ("profit_margin", fundamentals.profit_margin),
            ("return_on_equity", fundamentals.return_on_equity),
            ("debt_to_equity", fundamentals.debt_to_equity),
            ("dividend_yield", fundamentals.dividend_yield),
            ("beta", fundamentals.beta),
        ):
            lines.append(f"  {label:<22}{'not available' if value is None else f'{value:,.4f}'}")
    if trend is not None:
        lines.append(f"  {'current_price':<22}{trend.current_price:,.4f}")
        lines.append(f"  {'annualized_vol_pct':<22}{trend.annualized_vol_pct:,.2f}")
        lines.append(f"  {'sessions_available':<22}{trend.sessions_available}")
    lines.append("")
    lines.append(f"  Data sources: {'; '.join(bundle.data_sources) or 'none recorded'}")
    if bundle.is_synthetic:
        lines.append("  WARNING: this report uses SYNTHETIC data. The numbers are not real.")
    return "\n".join(lines)


def render_snapshot_report(bundle: EvidenceBundle) -> str:
    """Render a full snapshot report as plain text.

    Raises ValueError for an attribution bundle, which has its own renderer.
    """
    if bundle.question_type != "snapshot":
        raise ValueError(
            f"render_snapshot_report handles snapshot bundles; got "
            f"{bundle.question_type!r}. Use synthesis.template_report for price moves."
        )
    return (
        "\n".join(
            [
                _rule(),
                f"  COMPANY SNAPSHOT -- {bundle.ticker}",
                _rule(),
                _render_question(bundle),
                _render_identity(bundle),
                _render_numbers(bundle),
                _render_peers(bundle),
                _render_price(bundle),
                _closing_statement(bundle),
                _render_raw(bundle),
                "",
                _rule(),
            ]
        )
        + "\n"
    )
