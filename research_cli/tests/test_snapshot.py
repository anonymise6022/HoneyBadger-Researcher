"""Tests for the fundamentals snapshot: units, edge cases, and no verdicts.

The unit tests here matter more than they look. yfinance's `.info` mixes
percent and fraction conventions *within the same payload*, and getting that
wrong prints a 2.4% dividend yield as "240%".
"""

from __future__ import annotations

from datetime import date

import pytest

from research_cli.data.fundamentals import normalize_info
from research_cli.evidence.snapshot import SECTOR_PEERS, build_snapshot_bundle
from research_cli.query_parser import parse_query
from research_cli.synthesis.snapshot_report import (
    SNAPSHOT_SECTIONS,
    render_snapshot_report,
)
from research_cli.synthesis.validator import validate_report

TODAY = date(2026, 9, 21)


def snapshot(query: str):
    return build_snapshot_bundle(parse_query(query, today=TODAY), source="mock")


# --- unit normalization ----------------------------------------------------

def test_dividend_yield_is_converted_from_percent_to_fraction() -> None:
    """yfinance reports KO's 2.4% yield as 2.4, not 0.024.

    Treating it as a fraction would print "240%".
    """
    result = normalize_info({"dividendYield": 2.4, "quoteType": "EQUITY"}, "KO")
    assert result.get("dividend_yield") == pytest.approx(0.024)


def test_margins_and_growth_are_already_fractions() -> None:
    result = normalize_info(
        {"profitMargins": 0.276, "revenueGrowth": 0.164, "quoteType": "EQUITY"}, "AAPL"
    )
    assert result.get("profit_margin") == pytest.approx(0.276)
    assert result.get("revenue_growth") == pytest.approx(0.164)


def test_debt_to_equity_is_converted_from_percent() -> None:
    """78.4 means 0.78x, not 78x."""
    result = normalize_info({"debtToEquity": 78.4, "quoteType": "EQUITY"}, "AAPL")
    assert result.get("debt_to_equity") == pytest.approx(0.784)


def test_an_implausible_value_is_flagged_not_clamped() -> None:
    """If the vendor changes convention, say so rather than print a lie."""
    result = normalize_info({"dividendYield": 250.0, "quoteType": "EQUITY"}, "X")
    assert "dividend_yield" in result.suspicious_fields
    assert "units" in result.suspicious_fields["dividend_yield"]


def test_missing_fields_are_listed_not_zero_filled() -> None:
    result = normalize_info({"trailingPE": 20.0, "quoteType": "EQUITY"}, "X")
    assert result.get("profit_margin") is None
    assert "profit_margin" in result.missing_fields


def test_nan_is_treated_as_missing() -> None:
    result = normalize_info({"trailingPE": float("nan"), "quoteType": "EQUITY"}, "X")
    assert result.get("trailing_pe") is None
    assert "trailing_pe" in result.missing_fields


@pytest.mark.parametrize("quote_type", ["ETF", "MUTUALFUND", "INDEX", "CURRENCY"])
def test_funds_are_recognized(quote_type: str) -> None:
    assert normalize_info({"quoteType": quote_type}, "SPY").is_fund


def test_equities_are_not_funds() -> None:
    assert not normalize_info({"quoteType": "EQUITY"}, "AAPL").is_fund


# --- the bundle across company types --------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "is AAPL good to invest",     # mega-cap tech
        "should i buy TSLA",          # growth, no dividend
        "is KO a good investment",    # dividend / value
        "tell me about GME",          # small-cap
        "is JNJ worth buying",        # defensive dividend payer
        "tell me about PLTR",         # high-growth
    ],
)
def test_snapshot_builds_and_validates_for_every_company_type(query: str) -> None:
    bundle = snapshot(query)
    report = render_snapshot_report(bundle)
    result = validate_report(report, bundle, required_sections=SNAPSHOT_SECTIONS)
    assert result.ok, "\n".join(str(i) for i in result.issues)


def test_every_snapshot_carries_limitations() -> None:
    assert snapshot("is AAPL good to invest").limitations


def test_loss_making_company_is_called_out() -> None:
    """A negative margin changes how every other figure reads."""

    bundle = snapshot("tell me about LOSSCO")
    if bundle.fundamentals.is_profitable is False:
        labels = " ".join(c.label for c in bundle.counterevidence)
        assert "not currently profitable" in labels


def test_a_company_with_no_dividend_says_so() -> None:
    for query in ("tell me about TSLA", "tell me about PLTR", "tell me about NVDA"):
        bundle = snapshot(query)
        if bundle.fundamentals.dividend_yield is None:
            labels = " ".join(c.label for c in bundle.counterevidence)
            assert "No dividend" in labels
            return
    pytest.skip("every mock company in this sample pays a dividend")


def test_high_multiple_produces_a_priced_in_warning() -> None:
    for query in ("tell me about NVDA", "tell me about PLTR", "tell me about TSLA"):
        bundle = snapshot(query)
        pe = next((p for p in bundle.peers if p.metric == "trailing_pe"), None)
        if pe and pe.vs_median_pct and pe.vs_median_pct > 25:
            assert any("priced in" in c.label for c in bundle.counterevidence)
            return
    pytest.skip("no richly valued company in this sample")


def test_peer_group_excludes_the_subject_itself() -> None:
    bundle = snapshot("is AAPL good to invest")
    for peer in bundle.peers:
        assert "AAPL" not in peer.peer_values


def test_sector_peer_lists_are_well_formed() -> None:
    for sector, peers in SECTOR_PEERS.items():
        assert len(peers) >= 4, f"{sector} needs enough peers for a stable median"
        assert len(set(peers)) == len(peers), f"{sector} has a duplicate"


# --- no verdicts -----------------------------------------------------------

@pytest.mark.parametrize(
    "query", ["is AAPL good to invest", "should i buy TSLA", "is KO a buy"]
)
def test_no_report_ever_gives_a_verdict(query: str) -> None:
    bundle = snapshot(query)
    report = render_snapshot_report(bundle)
    result = validate_report(report, bundle, required_sections=SNAPSHOT_SECTIONS)
    assert not any(i.kind == "verdict_language" for i in result.errors)


def test_echoing_a_bluntly_worded_question_is_not_a_verdict() -> None:
    """'is KO a buy' quoted back is the user's phrase, not a recommendation."""
    bundle = snapshot("is KO a buy")
    report = render_snapshot_report(bundle)
    assert "is KO a buy" in report
    assert validate_report(report, bundle, required_sections=SNAPSHOT_SECTIONS).ok


def test_closing_statement_is_specific_not_boilerplate() -> None:
    """A generic disclaimer reads as throat-clearing; naming the real tension does not."""
    bundle = snapshot("is AAPL good to invest")
    report = render_snapshot_report(bundle)
    # Normalized: the renderer word-wraps, so a phrase can span a line break.
    closing = " ".join(report.split("WHAT THIS DOES NOT TELL YOU")[1].split())
    assert bundle.ticker in closing
    assert "decision is a separate thing, and it is yours" in closing
    assert "fell by half" in closing
    assert "what you need the money for" in closing


def test_attribution_bundle_is_refused_by_the_snapshot_renderer() -> None:
    bundle = snapshot("is AAPL good to invest").model_copy(
        update={"question_type": "attribution"}
    )
    with pytest.raises(ValueError, match="snapshot"):
        render_snapshot_report(bundle)


def test_fund_gets_fund_specific_counterevidence() -> None:
    """Without it an ETF produces an empty section reading as 'nothing to worry about'."""
    from research_cli.data.fundamentals import normalize_info
    from research_cli.evidence.snapshot import _counterevidence

    raw = normalize_info({"quoteType": "ETF", "trailingPE": 25.0}, "SPY")
    items = _counterevidence("SPY", raw, None, [])
    labels = " ".join(i.label for i in items)
    assert "no company here" in labels
    assert "costs" in labels
