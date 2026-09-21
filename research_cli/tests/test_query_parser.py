"""Tests for the deterministic query parser.

Every test fixes `today` explicitly. A test that only passes on the day it
was written is worse than no test, and relative-date parsing is exactly
where that trap lives.
"""

from __future__ import annotations

from datetime import date

import pytest

from research_cli.query_parser import (
    ParsedQuery,
    QueryParseError,
    parse_query,
    resolve_symbol,
)

TODAY = date(2026, 9, 21)  # a Monday


def parsed(query: str) -> ParsedQuery:
    return parse_query(query, today=TODAY)


# --- question type ---------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "why did SPY fall today",
        "why is AAPL down",
        "what happened to TSLA yesterday",
        "what caused MSFT to surge",
        "explain SPY's decline this week",
        "how come NVDA dropped",
        "what drove the QQQ rally",
        "SPY jumped today why",
    ],
)
def test_attribution_phrasings(query: str) -> None:
    assert parsed(query).question_type == "attribution"


@pytest.mark.parametrize(
    "query",
    [
        "is AAPL good to invest",
        "is TSLA a good investment",
        "should i buy NVDA",
        "is MSFT worth buying",
        "tell me about AMZN",
        "what do you think about GOOGL",
        "AAPL fundamentals",
        "research META",
    ],
)
def test_snapshot_phrasings(query: str) -> None:
    assert parsed(query).question_type == "snapshot"


def test_attribution_wins_when_it_is_asked_first() -> None:
    """'why did X fall, should i buy' is a why-question with a coda."""
    assert parsed("why did AAPL fall today, should i buy").question_type == "attribution"


def test_snapshot_wins_when_it_is_asked_first() -> None:
    assert parsed("should i buy AAPL after what happened").question_type == "snapshot"


# --- direction -------------------------------------------------------------

@pytest.mark.parametrize(
    "verb",
    ["fall", "fell", "drop", "dropped", "decline", "declined", "plunge",
     "tank", "tanked", "slide", "slid", "tumble", "sank", "dipped"],
)
def test_down_verbs(verb: str) -> None:
    assert parsed(f"why did SPY {verb} today").direction == "down"


@pytest.mark.parametrize(
    "verb",
    ["rally", "rallied", "jump", "jumped", "surge", "surged", "soar",
     "soared", "spike", "spiked", "climb", "climbed", "rose", "popped"],
)
def test_up_verbs(verb: str) -> None:
    assert parsed(f"why did SPY {verb} today").direction == "up"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("sell off", "down"),
        ("sold off", "down"),
        ("selling off", "down"),
        ("go down", "down"),
        ("went down", "down"),
        ("went up", "up"),
        ("take off", "up"),
        ("bounce back", "up"),
    ],
)
def test_multi_word_direction_phrases(phrase: str, expected: str) -> None:
    assert parsed(f"why did SPY {phrase} today").direction == expected


def test_direction_is_unspecified_when_absent() -> None:
    assert parsed("what happened to TSLA yesterday").direction == "unspecified"


def test_first_direction_word_wins() -> None:
    """The move being asked about comes first; later ones are context."""
    assert parsed("why did SPY fall after the rally").direction == "down"


# --- symbol resolution -----------------------------------------------------

def test_plain_uppercase_ticker() -> None:
    assert parsed("why did SPY fall today").ticker == "SPY"


def test_dollar_prefix_beats_everything() -> None:
    """'$F' must win over the uppercase scan and the stopword list."""
    assert parsed("why did $F tank last 5 days").ticker == "F"


def test_lowercase_ticker_is_read_and_announced() -> None:
    result = parsed("why did aapl fall today")
    assert result.ticker == "AAPL"
    assert any("aapl" in warning for warning in result.warnings)


def test_class_share_is_translated_to_vendor_spelling() -> None:
    """Users write BRK.B; the price vendor wants BRK-B."""
    assert parsed("why did brk.b slide on 9/15").ticker == "BRK-B"
    assert parsed("why did BRK.B slide today").ticker == "BRK-B"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("why did apple fall today", "AAPL"),
        ("tell me about tesla", "TSLA"),
        ("why did the nasdaq drop", "QQQ"),
        ("what happened to the dow", "DIA"),
        ("why did bitcoin crash", "BTC-USD"),
        ("is gold good to invest", "GLD"),
    ],
)
def test_company_and_index_aliases(query: str, expected: str) -> None:
    assert parsed(query).ticker == expected


@pytest.mark.parametrize(
    "query",
    ["why did the stock market fall today", "why are stocks down", "why did the market drop"],
)
def test_broad_market_phrases_resolve_and_always_warn(query: str) -> None:
    """Substituting SPY for 'the market' is an interpretation, so it is said."""
    result = parsed(query)
    assert result.ticker == "SPY"
    assert any("stand-in" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("query", "ticker"),
    [
        ("why did EURUSD rally today", "EURUSD"),
        ("why did EUR/USD rally today", "EURUSD"),
        ("why did eur/usd fall today", "EURUSD"),
        ("what happened to GBPJPY", "GBPJPY"),
    ],
)
def test_fx_pairs(query: str, ticker: str) -> None:
    result = parsed(query)
    assert result.ticker == ticker
    assert result.is_fx
    assert result.yf_symbol == f"{ticker}=X"


def test_equity_is_not_flagged_as_fx() -> None:
    result = parsed("why did AAPL fall today")
    assert not result.is_fx
    assert result.yf_symbol == "AAPL"


def test_several_symbols_answers_the_first_and_warns() -> None:
    result = parsed("why did AAPL and MSFT fall today")
    assert result.ticker == "AAPL"
    assert any("several" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("token", "expected"),
    [("$aapl", ("AAPL", False)), ("EUR/USD", ("EURUSD", True)),
     ("brk.b", ("BRK-B", False)), ("spy", ("SPY", False))],
)
def test_resolve_symbol_directly(token: str, expected: tuple[str, bool]) -> None:
    assert resolve_symbol(token) == expected


# --- dates -----------------------------------------------------------------

def test_today() -> None:
    result = parsed("why did SPY fall today")
    assert result.start == result.end == TODAY
    assert result.is_single_day


def test_yesterday() -> None:
    result = parsed("what happened to TSLA yesterday")
    assert result.start == result.end == date(2026, 9, 20)


def test_iso_date() -> None:
    assert parsed("why did NVDA drop on 2026-09-15").start == date(2026, 9, 15)


def test_us_slashed_date_with_year() -> None:
    assert parsed("why did NVDA drop on 3/14/2026").start == date(2026, 3, 14)


def test_slashed_date_without_year_never_lands_in_the_future() -> None:
    """'12/25' asked in September means last December, not this one."""
    assert parsed("why did SPY fall on 12/25").start == date(2025, 12, 25)


def test_written_month_date() -> None:
    assert parsed("why did SPY fall on September 15").start == date(2026, 9, 15)
    assert parsed("why did SPY fall on 15 March 2025").start == date(2025, 3, 15)


@pytest.mark.parametrize(
    ("phrase", "days"),
    [("this week", 7), ("last week", 7), ("this month", 30), ("last year", 365)],
)
def test_relative_ranges(phrase: str, days: int) -> None:
    result = parsed(f"why did SPY decline {phrase}")
    assert result.end == TODAY
    assert (result.end - result.start).days == days


def test_last_n_days() -> None:
    result = parsed("why did SPY fall last 5 days")
    assert (result.end - result.start).days == 5
    assert not result.is_single_day


def test_missing_date_defaults_to_today_and_warns() -> None:
    result = parsed("why is AAPL down")
    assert result.start == TODAY
    assert any("no date given" in warning for warning in result.warnings)


def test_future_date_is_refused() -> None:
    with pytest.raises(QueryParseError, match="future"):
        parsed("why did SPY fall on 2026-12-25")


def test_impossible_date_is_refused() -> None:
    with pytest.raises(QueryParseError, match="not a real date"):
        parsed("why did SPY fall on 2026-02-30")


# --- errors ----------------------------------------------------------------

@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_refused_with_an_example(query: str) -> None:
    with pytest.raises(QueryParseError) as excinfo:
        parsed(query)
    assert "research" in excinfo.value.suggestion


def test_query_with_no_symbol_at_all_is_refused_with_an_example() -> None:
    with pytest.raises(QueryParseError) as excinfo:
        parsed("why did it happen today")
    assert excinfo.value.suggestion


# --- rendering helpers -----------------------------------------------------

def test_describe_period_for_a_single_day() -> None:
    assert parsed("why did SPY fall today").describe_period() == "Monday 21 September 2026"


def test_describe_period_for_a_range() -> None:
    assert "to" in parsed("why did SPY fall this week").describe_period()


def test_parsing_is_deterministic() -> None:
    """The same string must parse identically every time -- no model, no clock."""
    first, second = parsed("why did SPY fall today"), parsed("why did SPY fall today")
    assert first == second
