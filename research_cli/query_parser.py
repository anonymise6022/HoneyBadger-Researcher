"""Natural-language query -> structured request, deterministically.

No model, no embeddings, no API call: regular expressions and keyword
tables only. A research tool that cannot explain its own numbers is not
worth much, and that starts with the parse. If this module decides your
question was about AAPL over the last week, it can point at the exact token
that made it think so, and it will make the same decision tomorrow.

Two question types are recognized:

    attribution   "why did SPY fall today"       -> what coincided with a move
    snapshot      "is AAPL good to invest"       -> fundamentals and context

Everything the parser is unsure about becomes a `warning` on the result
rather than a silent default, so the CLI can tell the user what it assumed.

**Known limits, by design.** Only one ticker per query -- "compare AAPL and
MSFT" parses the first and warns. Relative dates resolve against the
*calendar*, not the exchange calendar, so "today" on a Sunday resolves to
Sunday and the price layer is responsible for stepping back to the last
session. And an unrecognized word will be read as a ticker; the data
layer's "no such symbol" error is the backstop, which is the right place
for it, since the parser has no way to know what trades. Symbol detection
runs strongest-signal-first -- "$AAPL", then a slashed currency pair, then
the alias table, then uppercase tokens, then a lowercase word that is not
ordinary English -- and every guess below the uppercase tier is announced
as a warning rather than made silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

__all__ = [
    "DIRECTION_WORDS",
    "ParsedQuery",
    "QueryParseError",
    "QuestionType",
    "parse_query",
    "resolve_symbol",
]

QuestionType = Literal["attribution", "snapshot"]
Direction = Literal["down", "up", "unspecified"]


class QueryParseError(ValueError):
    """Raised when a query cannot be resolved into a request.

    Carries a `suggestion` so the CLI can print something more useful than
    "parse error" to somebody who has just typed their first query.
    """

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.suggestion = suggestion


# --- vocabulary ------------------------------------------------------------

#: Direction verbs, mapped to the sign of the move they describe. Grouped
#: rather than stemmed: stemming "rose" to "rise" needs a stemmer, and a
#: literal table is both shorter and auditable.
DIRECTION_WORDS: dict[str, Direction] = {
    # down
    "fall": "down", "fell": "down", "falling": "down", "falls": "down",
    "drop": "down", "dropped": "down", "dropping": "down", "drops": "down",
    "decline": "down", "declined": "down", "declining": "down", "declines": "down",
    "down": "down", "lower": "down", "loss": "down", "losses": "down",
    "plunge": "down", "plunged": "down", "tank": "down", "tanked": "down",
    "slide": "down", "slid": "down", "sink": "down", "sank": "down", "sunk": "down",
    "tumble": "down", "tumbled": "down", "slump": "down", "slumped": "down",
    "crash": "down", "crashed": "down", "sell-off": "down", "selloff": "down",
    "retreat": "down", "retreated": "down", "dip": "down", "dipped": "down",
    "weaken": "down", "weakened": "down", "red": "down",
    # up
    "rally": "up", "rallied": "up", "rallying": "up", "rallies": "up",
    "jump": "up", "jumped": "up", "jumping": "up", "jumps": "up",
    "rise": "up", "rose": "up", "rising": "up", "rises": "up",
    "surge": "up", "surged": "up", "soar": "up", "soared": "up",
    "spike": "up", "spiked": "up", "climb": "up", "climbed": "up",
    "gain": "up", "gained": "up", "gains": "up", "up": "up", "higher": "up",
    "advance": "up", "advanced": "up", "pop": "up", "popped": "up",
    "surging": "up", "green": "up", "boom": "up", "boomed": "up",
}

#: Phrases that mark a query as asking "why did it move".
_ATTRIBUTION_MARKERS = (
    "why", "what happened", "what caused", "what drove", "what's behind",
    "whats behind", "what is behind", "explain", "reason for", "cause of",
    "how come", "what made",
)

#: Phrases that mark a query as asking for a fundamentals snapshot. "should
#: i buy" is routed here deliberately: the tool answers it with evidence and
#: an explicit refusal to give a verdict, which is more useful to a beginner
#: than declining to parse the question they actually asked.
_SNAPSHOT_MARKERS = (
    "good to invest", "good investment", "worth buying", "worth investing",
    "should i buy", "should i invest", "should i own", "should i get",
    "is it a buy", "a good buy", "good stock", "tell me about",
    "what do you think", "look into", "research", "analyze", "analyse",
    "fundamentals", "overview of", "snapshot of", "invest in", "investing in",
)

#: Words that look like tickers but are not. Kept deliberately tight: the
#: cost of missing a real ticker is a confusing error, while the cost of an
#: extra entry here is only that "$X" must be used for that symbol.
_TICKER_STOPWORDS = frozenset({
    "A", "I", "IS", "IT", "DID", "DO", "THE", "WHY", "WHAT", "HOW", "SO", "ON",
    "IN", "AT", "TO", "OF", "AND", "OR", "BUY", "SELL", "GOOD", "BAD", "ME",
    "MY", "US", "PE", "EPS", "ETF", "CEO", "IPO", "AI", "Q1", "Q2", "Q3", "Q4",
    "YTD", "USD", "EUR", "GBP", "JPY", "OK", "VS", "FOR", "BE", "AM", "PM",
    "TODAY", "WEEK", "MONTH", "YEAR", "DAY", "LAST", "THIS", "PAST", "NOW",
    "UP", "DOWN", "ALL", "ANY", "NEW", "OLD", "BIG", "TOP", "LOW", "HIGH",
    "GET", "HAS", "HAD", "WAS", "WERE", "ARE", "CAN", "WILL", "WOULD", "SHOULD",
    "THINK", "ABOUT", "INTO", "OVER", "FROM", "WITH", "THAT", "THAN", "THEN",
})

#: Ordinary English that must never be mistaken for a lowercase ticker.
#: Only consulted by the last-resort lowercase scan; the uppercase scan has
#: its own, tighter list. Direction verbs and marker phrases are folded in
#: programmatically below rather than repeated here.
_COMMON_WORDS = frozenset(["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "doing", "done", "for", "from", "get", "got", "had", "has", "have", "he", "her", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "like", "make", "me", "my", "no", "not", "now", "of", "off", "on", "once", "one", "only", "or", "other", "our", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "too", "under", "until", "up", "us", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with", "would", "you", "your", "yours", "about", "after", "again", "against", "all", "also", "am", "any", "because", "before", "being", "below", "between", "both", "during", "each", "few", "further", "here", "more", "most", "much", "must", "need", "no", "nor", "own", "than", "through", "very", "what", "whom", "does", "isn't", "wasn't", "weren't", "don't", "doesn't", "didn't", "stock", "stocks", "share", "shares", "price", "prices", "market", "markets", "today", "yesterday", "week", "weeks", "month", "months", "year", "years", "day", "days", "ago", "last", "past", "this", "next", "since", "happened", "happen", "cause", "caused", "causes", "reason", "reasons", "explain", "look", "looking", "think", "thought", "tell", "told", "worth", "good", "bad", "better", "best", "buy", "buying", "bought", "sell", "selling", "sold", "invest", "investing", "investment", "hold", "holding", "own", "owning", "company", "companies", "news", "earnings", "report"])


#: Everyday names -> the symbol a retail user almost certainly means. An
#: index is mapped to its most liquid ETF because that is what actually has
#: an OHLCV history a beginner can buy and compare against.
_ALIASES: dict[str, str] = {
    "S&P": "SPY", "S&P500": "SPY", "SP500": "SPY", "SPX": "SPY",
    "THE S&P": "SPY", "S AND P": "SPY", "SANDP": "SPY",
    "NASDAQ": "QQQ", "THE NASDAQ": "QQQ", "NDX": "QQQ",
    "DOW": "DIA", "THE DOW": "DIA", "DJIA": "DIA",
    "RUSSELL": "IWM", "RUSSELL2000": "IWM", "SMALL CAPS": "IWM",
    "BITCOIN": "BTC-USD", "BTC": "BTC-USD", "ETHEREUM": "ETH-USD", "ETH": "ETH-USD",
    "GOLD": "GLD", "OIL": "USO", "CRUDE": "USO",
    "VIX": "^VIX", "THE VIX": "^VIX",
    "APPLE": "AAPL", "TESLA": "TSLA", "MICROSOFT": "MSFT", "AMAZON": "AMZN",
    "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "NVIDIA": "NVDA", "META": "META",
    "FACEBOOK": "META", "NETFLIX": "NFLX",
}

#: Vague phrases a beginner uses for "the market". These resolve to a
#: liquid proxy *and always warn*, because substituting an index ETF for
#: "the market" is an interpretation, not a lookup -- unlike "apple" ->
#: AAPL, which is simply the name of the thing.
_BROAD_ALIASES: dict[str, str] = {
    "THE STOCK MARKET": "SPY", "STOCK MARKET": "SPY", "THE MARKET": "SPY",
    "THE MARKETS": "SPY", "MARKETS": "SPY", "STOCKS": "SPY", "EQUITIES": "SPY",
    "MY PORTFOLIO": "SPY", "TECH STOCKS": "QQQ", "TECH": "QQQ",
}

_BROAD_DESCRIPTIONS = {
    "SPY": "SPY, an S&P 500 index fund, as a stand-in for the US stock market",
    "QQQ": "QQQ, a Nasdaq-100 index fund, as a stand-in for tech stocks",
}

#: Six-letter currency pairs the FX branch recognizes, plus their slashed
#: spellings. yfinance quotes these as "<PAIR>=X".
_FX_CURRENCIES = frozenset({
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "MXN", "CNY",
})

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


@dataclass(frozen=True)
class ParsedQuery:
    """A query resolved into everything the data layer needs.

    Attributes
    ----------
    question_type : "attribution" or "snapshot".
    ticker : the symbol as the user would recognize it, e.g. "SPY",
        "EURUSD". Use `yf_symbol` for the data-vendor spelling.
    start, end : inclusive date range the question covers. For "today" both
        are the same date. The price layer steps back to the most recent
        session if these land on a weekend or holiday.
    direction : the move direction the user asserted, if any. Used only to
        check the user against the data -- if somebody asks why SPY fell on
        a day it rose, that is worth telling them, and `direction_conflict`
        on the bundle carries it.
    raw_query : the original string, kept for the report header.
    is_fx : whether the symbol is a currency pair, which changes which
        candidate factors make sense (no sector breadth for EURUSD).
    warnings : assumptions the parser made, to be surfaced to the user.
    """

    question_type: QuestionType
    ticker: str
    start: date
    end: date
    direction: Direction
    raw_query: str
    is_fx: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def yf_symbol(self) -> str:
        """The symbol spelled the way the price vendor expects it."""
        return f"{self.ticker}=X" if self.is_fx else self.ticker

    @property
    def is_single_day(self) -> bool:
        return self.start == self.end

    def describe_period(self) -> str:
        """A human phrase for the date range, for report headers."""
        if self.is_single_day:
            return self.start.strftime("%A %d %B %Y")
        return (
            f"{self.start.strftime('%d %b %Y')} to {self.end.strftime('%d %b %Y')} "
            f"({(self.end - self.start).days + 1} calendar days)"
        )


# --- symbol resolution -----------------------------------------------------

def resolve_symbol(token: str) -> tuple[str, bool]:
    """Normalize one token into (ticker, is_fx).

    Handles "$AAPL", "EUR/USD", "eurusd", "brk.b" and the alias table.
    Returns the token uppercased and stripped if nothing special applies --
    validation that the symbol actually trades belongs to the data layer.
    """
    cleaned = token.strip().upper().lstrip("$")
    if cleaned in _ALIASES:
        resolved = _ALIASES[cleaned]
        return resolved, resolved.endswith("=X")

    slashless = cleaned.replace("/", "")
    if (
        len(slashless) == 6
        and slashless[:3] in _FX_CURRENCIES
        and slashless[3:] in _FX_CURRENCIES
    ):
        return slashless, True

    # Class shares are written BRK.B by users and BRK-B by the vendor.
    if re.fullmatch(r"[A-Z]{1,5}\.[A-Z]", cleaned):
        return cleaned.replace(".", "-"), False
    return cleaned, False


def _extract_ticker(query: str) -> tuple[str, bool, list[str]]:
    """Find the symbol in the query, preferring the most explicit signal.

    Precedence, strongest first: an explicit "$TICKER"; a slashed currency
    pair; a multi-word alias ("the s&p"); a single-word alias; an uppercase
    token that is not a stopword. The order matters -- "why did $F fall"
    must find F, which the stopword-filtered scan would also find, but
    "should i buy GOLD" must resolve the alias rather than treat GOLD as a
    literal ticker.
    """
    warnings: list[str] = []

    dollar = re.findall(r"\$([A-Za-z][A-Za-z.\-]{0,6})", query)
    if dollar:
        if len(dollar) > 1:
            warnings.append(
                f"found several symbols ({', '.join('$' + d for d in dollar)}); "
                f"answering about ${dollar[0]} only"
            )
        ticker, is_fx = resolve_symbol(dollar[0])
        return ticker, is_fx, warnings

    slashed = re.findall(r"\b([A-Za-z]{3})\s*/\s*([A-Za-z]{3})\b", query)
    for base, quote in slashed:
        if base.upper() in _FX_CURRENCIES and quote.upper() in _FX_CURRENCIES:
            return f"{base.upper()}{quote.upper()}", True, warnings

    upper_query = query.upper()
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", upper_query):
            resolved = _ALIASES[alias]
            return resolved, resolved.endswith("=X"), warnings

    for alias in sorted(_BROAD_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", upper_query):
            resolved = _BROAD_ALIASES[alias]
            warnings.append(
                f"{alias.lower()!r} is not a ticker, so this uses "
                f"{_BROAD_DESCRIPTIONS.get(resolved, resolved)}"
            )
            return resolved, False, warnings

    candidates = [
        token
        for token in re.findall(r"\b[A-Z][A-Z.\-]{0,5}\b", query)
        if token not in _TICKER_STOPWORDS
    ]
    if candidates:
        if len(set(candidates)) > 1:
            warnings.append(
                f"found several possible symbols ({', '.join(dict.fromkeys(candidates))}); "
                f"answering about {candidates[0]} only"
            )
        ticker, is_fx = resolve_symbol(candidates[0])
        return ticker, is_fx, warnings

    # A dot inside a short lowercase token is a class-share ticker ("brk.b")
    # essentially always -- English does not put a full stop mid-word -- so
    # this needs no warning.
    dotted = re.findall(r"\b([a-z]{1,5}\.[a-z])\b", query)
    if dotted:
        ticker, is_fx = resolve_symbol(dotted[0])
        return ticker, is_fx, warnings

    # Last resort: a lowercase word that is not ordinary English. Beginners
    # type "why did aapl drop" far more often than they type capitals, so
    # refusing here would reject a large share of real questions. The guess
    # is always announced, and an actual non-symbol is caught downstream by
    # the data layer, which is the only layer that knows what trades.
    lowercase_stop = (
        _COMMON_WORDS
        | set(DIRECTION_WORDS)
        | {word.lower() for word in _TICKER_STOPWORDS}
    )
    loose = [
        token
        for token in re.findall(r"\b[a-z][a-z\-]{0,5}\b", query)
        if token not in lowercase_stop
    ]
    if loose:
        ticker, is_fx = resolve_symbol(loose[0])
        warnings.append(f"read {loose[0]!r} as the ticker {ticker}")
        return ticker, is_fx, warnings

    # A long run of capitals is a symbol the user believes in; saying "no
    # ticker found" would be misleading, because one was found and rejected.
    overlong = re.findall(r"\b[A-Z]{7,}\b", query)
    if overlong:
        raise QueryParseError(
            f"{overlong[0]!r} does not look like a ticker symbol -- real tickers are "
            "1 to 5 letters",
            suggestion=(
                "Check the symbol on your broker or on finance.yahoo.com, then try:\n"
                '  research "why did AAPL fall today"'
            ),
        )

    raise QueryParseError(
        "could not find a ticker symbol in that question",
        suggestion=(
            "Write the symbol in capitals or with a dollar sign, for example:\n"
            '  research why did SPY fall today\n'
            '  research "is $AAPL good to invest"'
        ),
    )


# --- date resolution -------------------------------------------------------

def _explicit_date(query: str, today: date) -> date | None:
    """Parse an explicit date, if the query holds one.

    Accepts ISO (2026-09-15), US slashed (9/15/2026 and 9/15), and written
    months ("September 15", "15 Sept 2026"). A slashed date without a year
    is assumed to be the most recent such date not in the future, which is
    what somebody typing "why did SPY drop on 3/14" means.
    """
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", query)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError as exc:
            raise QueryParseError(
                f"{iso.group(0)!r} is not a real date", suggestion="Use YYYY-MM-DD."
            ) from exc

    slashed = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", query)
    if slashed:
        month, day = int(slashed.group(1)), int(slashed.group(2))
        if slashed.group(3):
            year = int(slashed.group(3))
            year += 2000 if year < 100 else 0
        else:
            year = today.year
        try:
            parsed = date(year, month, day)
        except ValueError as exc:
            raise QueryParseError(
                f"{slashed.group(0)!r} is not a real date",
                suggestion="Use MM/DD/YYYY or YYYY-MM-DD.",
            ) from exc
        if not slashed.group(3) and parsed > today:
            parsed = date(year - 1, month, day)
        return parsed

    written = re.search(
        r"\b(?:on\s+)?([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", query
    )
    if written and written.group(1).lower() in _MONTHS:
        month = _MONTHS[written.group(1).lower()]
        day, year = int(written.group(2)), int(written.group(3) or today.year)
        try:
            parsed = date(year, month, day)
        except ValueError as exc:
            raise QueryParseError(
                f"{written.group(0)!r} is not a real date", suggestion="Use YYYY-MM-DD."
            ) from exc
        if not written.group(3) and parsed > today:
            parsed = date(year - 1, month, day)
        return parsed

    reversed_written = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?(?:,?\s+(\d{4}))?\b", query
    )
    if reversed_written and reversed_written.group(2).lower() in _MONTHS:
        month = _MONTHS[reversed_written.group(2).lower()]
        day, year = int(reversed_written.group(1)), int(reversed_written.group(3) or today.year)
        try:
            parsed = date(year, month, day)
        except ValueError as exc:
            raise QueryParseError(
                f"{reversed_written.group(0)!r} is not a real date",
                suggestion="Use YYYY-MM-DD.",
            ) from exc
        if not reversed_written.group(3) and parsed > today:
            parsed = date(year - 1, month, day)
        return parsed
    return None


def _relative_range(query: str, today: date) -> tuple[date, date, list[str]]:
    """Resolve a relative period phrase, defaulting to today.

    Relative phrases resolve against the calendar, not the exchange
    calendar: "today" on a Sunday is that Sunday. Stepping back to the last
    traded session is the price layer's job, because only it knows which
    days the symbol actually traded.
    """
    warnings: list[str] = []
    lowered = query.lower()

    if re.search(r"\byesterday\b", lowered):
        return today - timedelta(days=1), today - timedelta(days=1), warnings
    if re.search(r"\b(today|so far today|right now|currently)\b", lowered):
        return today, today, warnings

    span = re.search(r"\b(?:last|past|previous)\s+(\d{1,3})\s+(day|week|month)s?\b", lowered)
    if span:
        count, unit = int(span.group(1)), span.group(2)
        days = count * {"day": 1, "week": 7, "month": 30}[unit]
        return today - timedelta(days=days), today, warnings

    for pattern, days in (
        (r"\b(this|the past|past|last)\s+week\b", 7),
        (r"\b(this|the past|past|last)\s+month\b", 30),
        (r"\b(this|the past|past|last)\s+quarter\b", 91),
        (r"\b(this|the past|past|last)\s+year\b", 365),
    ):
        if re.search(pattern, lowered):
            return today - timedelta(days=days), today, warnings

    warnings.append("no date given, so this answers about the most recent session")
    return today, today, warnings


#: Multi-word direction phrases, collapsed before the token scan so they do
#: not have to be re-spelled in `DIRECTION_WORDS` for every tense.
_DIRECTION_PHRASES = (
    (r"\bsold\s+off\b", "selloff"),
    (r"\bsell(?:ing|s)?\s+off\b", "selloff"),
    (r"\bgave?\s+back\b", "decline"),
    (r"\bbounced?\s+back\b", "rally"),
    (r"\bwent\s+up\b", "rise"),
    (r"\bwent\s+down\b", "fall"),
    (r"\b(?:take|takes|taking|took)\s+off\b", "surge"),
)


def _extract_direction(query: str) -> Direction:
    """Read the move direction the user asserted, if they asserted one.

    Multi-word phrases ("sold off", "went up") are collapsed to a single
    token first, then the first direction word wins. First rather than most
    frequent: in "why did SPY fall after the rally", the fall is the thing
    being asked about and the rally is context.
    """
    lowered = query.lower()
    for pattern, replacement in _DIRECTION_PHRASES:
        lowered = re.sub(pattern, replacement, lowered)
    for token in re.findall(r"[a-z][a-z\-']*", lowered):
        if token in DIRECTION_WORDS:
            return DIRECTION_WORDS[token]
    return "unspecified"


def _classify(query: str) -> QuestionType:
    """Decide which question is being asked.

    Snapshot markers are checked first because "what do you think about
    AAPL" contains neither a direction word nor "why", while "why is AAPL a
    good investment" contains both -- and in that case the user wants the
    fundamentals, not a one-day attribution. An explicit attribution marker
    plus a direction word beats a snapshot marker, which is what makes
    "should i buy AAPL after it fell today" parse as a snapshot but "why
    did AAPL fall today, should i buy" parse as attribution.
    """
    lowered = query.lower()
    has_snapshot = any(marker in lowered for marker in _SNAPSHOT_MARKERS)
    has_attribution = any(marker in lowered for marker in _ATTRIBUTION_MARKERS)

    if has_attribution and not has_snapshot:
        return "attribution"
    if has_snapshot and not has_attribution:
        return "snapshot"
    if has_attribution and has_snapshot:
        # Both present: whichever marker appears first is the real question.
        first_attribution = min(
            (lowered.index(m) for m in _ATTRIBUTION_MARKERS if m in lowered), default=len(lowered)
        )
        first_snapshot = min(
            (lowered.index(m) for m in _SNAPSHOT_MARKERS if m in lowered), default=len(lowered)
        )
        return "attribution" if first_attribution <= first_snapshot else "snapshot"
    # Neither marker: a bare direction word still reads as "why did it move".
    return "attribution" if _extract_direction(query) != "unspecified" else "snapshot"


def parse_query(query: str, today: date | None = None) -> ParsedQuery:
    """Parse a natural-language research query into a structured request.

    Parameters
    ----------
    query : the user's question, e.g. "why did SPY fall today".
    today : the date "today" resolves to. Injectable so tests are not
        dated -- a test that passes only on the day it was written is worse
        than no test.

    Raises
    ------
    QueryParseError : when no symbol can be found, a date is impossible, or
        the range runs backwards. The error carries a `suggestion` with
        worked examples.
    """
    if not query or not query.strip():
        raise QueryParseError(
            "the query is empty",
            suggestion='Try: research why did SPY fall today',
        )

    cleaned = query.strip()
    question_type = _classify(cleaned)
    ticker, is_fx, warnings = _extract_ticker(cleaned)
    # Local civil date: market sessions are local-calendar events, so a
    # UTC 'today' would answer about the wrong session near midnight.
    today = today or date.today()  # noqa: DTZ011

    explicit = _explicit_date(cleaned, today)
    if explicit is not None:
        start = end = explicit
        if explicit > today:
            raise QueryParseError(
                f"{explicit.isoformat()} is in the future",
                suggestion="Ask about a date that has already happened.",
            )
    else:
        start, end, range_warnings = _relative_range(cleaned, today)
        warnings.extend(range_warnings)

    if start > end:
        raise QueryParseError(
            f"the range runs backwards ({start} to {end})",
            suggestion="Give the earlier date first.",
        )

    return ParsedQuery(
        question_type=question_type,
        ticker=ticker,
        start=start,
        end=end,
        direction=_extract_direction(cleaned),
        raw_query=cleaned,
        is_fx=is_fx,
        warnings=tuple(warnings),
    )
