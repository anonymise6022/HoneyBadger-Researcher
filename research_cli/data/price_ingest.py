"""Daily price history, plus the context needed to judge whether a move was unusual.

A beginner asking "why did SPY fall today" needs one number before any
explanation is worth reading: **was this actually unusual?** A 0.4% drop in
a market that moves 0.9% on a typical day is noise, and the honest answer
is "nothing in particular happened". Presenting a list of candidate causes
for a non-event is the single most misleading thing a tool like this can
do, so the size of the move relative to its own recent volatility is
computed here and carried on every observation.

Three things worth knowing about how it is computed:

**The volatility baseline excludes the day being judged.** Realized
volatility comes from the `lookback` sessions *before* the target session.
Including the target would let a large move inflate its own denominator and
shrink its own sigma multiple -- a 3-sigma day would measure as 2.4-sigma
purely because it is in the sample.

**Non-trading days step backwards, and say so.** "Today" on a Saturday, or
a market holiday, resolves to the most recent session at or before the
requested date, with a warning. Silently returning Friday's data for a
Saturday question is how a user ends up confused about which day they are
reading.

**A failed fetch degrades to synthetic data rather than crashing**, clearly
labelled on `source` and in a warning, so the tool still demonstrates what
it does when Yahoo is down or the user is offline. `source="live"` forces a
hard failure instead, which is what tests and any real analysis should use.
"""

from __future__ import annotations

import logging
import warnings as _warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .mock_data import mock_price_history

__all__ = [
    "OHLCV_COLUMNS",
    "PriceDataError",
    "PriceObservation",
    "Source",
    "UnknownSymbolError",
    "build_observation",
    "fetch_history",
    "fetch_many",
]

Source = Literal["auto", "live", "mock"]

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

#: Sessions of prior history used as the volatility baseline. Twenty is about
#: a trading month -- long enough for a stable standard deviation, short
#: enough to reflect the current regime rather than last year's.
DEFAULT_LOOKBACK = 20

#: Extra calendar days fetched before the requested start so the lookback
#: window can be filled. 2.2x covers weekends and holidays with room to spare.
_CALENDAR_PAD_FACTOR = 2.2
_MIN_VOL_SESSIONS = 5
_CACHE_TTL_SECONDS = 6 * 3600

_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"


class UnknownSymbolError(RuntimeError):
    """The vendor answered, and has never heard of this symbol.

    Kept distinct from `PriceDataError` because the two need opposite
    handling. A network failure should fall back to synthetic data so the
    tool still demonstrates itself; an unknown symbol must not, or a user
    who types "AAPI" instead of "AAPL" receives a complete, confident-looking
    report about a company that does not exist. A visible "synthetic data"
    warning is not sufficient mitigation for that -- the fix is to refuse.
    """

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.suggestion = suggestion


class PriceDataError(RuntimeError):
    """Raised when price data cannot be obtained and no fallback is allowed.

    Carries a `suggestion` so the CLI can print guidance rather than a
    traceback to somebody who mistyped a ticker.
    """

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.suggestion = suggestion


@dataclass(frozen=True)
class PriceObservation:
    """One symbol's move over a requested period, with volatility context.

    Attributes
    ----------
    symbol : the vendor spelling actually queried, e.g. "EURUSD=X".
    display_symbol : the spelling to show a user, e.g. "EURUSD".
    session_date : the session this describes. May be earlier than the date
        requested, if that date was not a trading day.
    requested_date : what the user asked for, kept so the report can say
        "you asked about Saturday; here is Friday".
    open, high, low, close, volume : that session's bar.
    prior_close : the previous session's close, the base for `return_pct`.
    return_pct : close-to-close return for `session_date`, in percent.
    period_return_pct : cumulative return over the whole requested range.
        Equal to `return_pct` for a single-day question.
    daily_vol_pct : standard deviation of daily returns over the lookback
        window *ending before* `session_date`, in percent.
    annualized_vol_pct : `daily_vol_pct` scaled by sqrt(252).
    sigma_multiple : `return_pct / daily_vol_pct`. The headline number.
    outlier_label : "typical", "1-sigma", "2-sigma" or "3-sigma+".
    volume_ratio : session volume over mean lookback volume. Above ~1.5
        suggests the move came with real participation.
    context : the lookback window, for modules that need the raw series.
    history : everything fetched, including the target session.
    source : "yfinance" or "mock".
    warnings : anything the user should know about this observation.
    """

    symbol: str
    display_symbol: str
    session_date: date
    requested_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    prior_close: float
    return_pct: float
    period_return_pct: float
    daily_vol_pct: float
    annualized_vol_pct: float
    sigma_multiple: float
    outlier_label: str
    volume_ratio: float
    context: pd.DataFrame = field(repr=False)
    history: pd.DataFrame = field(repr=False)
    source: str = "yfinance"
    warnings: tuple[str, ...] = ()

    @property
    def is_outlier(self) -> bool:
        """True when the move is at least a 2-sigma event."""
        return abs(self.sigma_multiple) >= 2.0

    @property
    def direction(self) -> str:
        """"up", "down" or "flat" for the period return."""
        if abs(self.period_return_pct) < 0.005:
            return "flat"
        return "up" if self.period_return_pct > 0 else "down"

    def describe_move(self) -> str:
        """A plain-language sentence a beginner can read without a glossary."""
        size = abs(self.period_return_pct)
        if self.outlier_label == "typical":
            judgement = (
                "that is within its normal daily range, so it does not need a "
                "special explanation"
            )
        elif self.outlier_label == "1-sigma":
            judgement = "a bit larger than a typical day, but not remarkable"
        elif self.outlier_label == "2-sigma":
            # No percentile figure here: quoting "bigger than 95% of days" would
            # assume normally distributed returns, and returns have fat tails,
            # so the real percentile is lower and varies by symbol.
            judgement = "a genuinely large move, well outside its recent daily range"
        else:
            judgement = "an extreme move, far outside its recent range"
        return (
            f"{self.display_symbol} {'rose' if self.direction == 'up' else 'fell' if self.direction == 'down' else 'was flat'} "
            f"{size:.2f}% versus a typical daily swing of {self.daily_vol_pct:.2f}% "
            f"({self.sigma_multiple:+.1f} standard deviations) -- {judgement}."
        )


def _classify_outlier(sigma_multiple: float) -> str:
    """Bucket a sigma multiple into a label a non-specialist can read."""
    magnitude = abs(sigma_multiple)
    if magnitude >= 3.0:
        return "3-sigma+"
    if magnitude >= 2.0:
        return "2-sigma"
    if magnitude >= 1.0:
        return "1-sigma"
    return "typical"


def _normalize_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce a vendor frame into the canonical schema, tz-naive and sorted.

    yfinance returns a timezone-aware index (and, for some calls, a
    MultiIndex on columns). Both leak into every downstream comparison
    against a plain `datetime.date` and produce confusing failures far from
    here, so they are flattened once, at the boundary.
    """
    if frame is None or len(frame) == 0:
        # The vendor responded; it simply has no such symbol. Falling back to
        # synthetic data here would fabricate a company.
        raise UnknownSymbolError(
            f"{symbol!r} does not appear to be a real, currently listed symbol",
            suggestion=(
                "Check the spelling on your broker or at finance.yahoo.com. "
                "For a fund or index, use its ETF ticker, for example SPY for the "
                "S&P 500 or QQQ for the Nasdaq 100."
            ),
        )
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    missing = [c for c in OHLCV_COLUMNS if c not in result.columns]
    if missing:
        raise PriceDataError(
            f"price data for {symbol} is missing column(s) {missing}",
            suggestion="This usually means the vendor returned a partial response; retry.",
        )

    result = result[list(OHLCV_COLUMNS)]
    index = pd.DatetimeIndex(result.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    result.index.name = "Date"
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.dropna(subset=["Close"])


def _cache_path(symbol: str, start: date, end: date) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in symbol.upper())
    return _CACHE_DIR / f"{safe}_{start.isoformat()}_{end.isoformat()}.csv"


def _read_cache(path: Path) -> pd.DataFrame | None:
    """Return cached rows if they exist and are fresh enough.

    A stale or corrupt cache file is treated as a miss, never as an error:
    a caching layer that can break the tool is worse than no caching layer.
    """
    try:
        if not path.is_file():
            return None
        # Naive local clock, compared against a naive mtime; both sides
        # use the same reference so the difference is correct.
        age = datetime.now().timestamp() - path.stat().st_mtime  # noqa: DTZ005
        if age > _CACHE_TTL_SECONDS:
            return None
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return frame if len(frame) else None
    except Exception:  # noqa: BLE001 - a bad cache must never be fatal
        return None


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
    except Exception:  # noqa: BLE001, S110 - caching is best-effort and must never fail a query
        pass


def _fetch_live(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch from yfinance, normalized. Raises PriceDataError on any failure."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise PriceDataError(
            "yfinance is not installed",
            suggestion="pip install -r research_cli/requirements.txt",
        ) from exc

    # yfinance logs vendor-side failures straight to the console
    # ("$XLK: Data doesn't exist for startDate = 505285200"). Those are
    # internal details in a Unix timestamp, shown to somebody who asked why
    # their stock moved. Failures are already surfaced as PriceDataError, so
    # the logger is silenced here rather than left to leak.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            # yfinance treats `end` as exclusive, so ask for one day more.
            raw = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=True,
            )
    except Exception as exc:
        raise PriceDataError(
            f"could not download {symbol}: {type(exc).__name__}: {exc}",
            suggestion="Check your internet connection, or run with --mock.",
        ) from exc
    return _normalize_frame(raw, symbol)


def fetch_history(
    symbol: str,
    start: date,
    end: date,
    source: Source = "auto",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    """Fetch daily OHLCV for one symbol.

    Parameters
    ----------
    symbol : vendor spelling, e.g. "AAPL" or "EURUSD=X".
    start, end : inclusive date bounds.
    source : "live" fails if the vendor cannot be reached; "mock" never
        touches the network; "auto" tries live and falls back to mock with a
        warning.
    use_cache : read and write the on-disk cache. Repeated queries about the
        same symbol are the normal case for this tool, and each live fetch
        is a round trip the user waits on.

    Returns
    -------
    (frame, source_used, warnings)

    Raises
    ------
    PriceDataError : on an invalid range, or when source="live" and the
        fetch fails.
    """
    if start > end:
        raise PriceDataError(
            f"the date range runs backwards ({start} to {end})",
            suggestion="Give the earlier date first.",
        )
    if source not in ("auto", "live", "mock"):
        raise ValueError(f"source must be 'auto', 'live' or 'mock', got {source!r}")

    if source == "mock":
        return mock_price_history(symbol, start, end), "mock", ()

    if use_cache:
        cached = _read_cache(_cache_path(symbol, start, end))
        if cached is not None:
            return _normalize_frame(cached, symbol), "yfinance", ()

    try:
        frame = _fetch_live(symbol, start, end)
    except UnknownSymbolError:
        # Never degrade to synthetic data for a symbol that does not exist.
        raise
    except PriceDataError as exc:
        if source == "live":
            raise
        message = (
            f"could not fetch live prices for {symbol} ({exc}); showing synthetic data "
            "so you can still see how this works -- the numbers below are NOT real"
        )
        return mock_price_history(symbol, start, end), "mock", (message,)

    if use_cache:
        _write_cache(_cache_path(symbol, start, end), frame)
    return frame, "yfinance", ()


def fetch_many(
    symbols: list[str], start: date, end: date, source: Source = "auto"
) -> dict[str, pd.DataFrame]:
    """Fetch several symbols, skipping any that fail.

    Candidate-factor symbols (VIX, sector ETFs, bond proxies) are
    individually optional -- losing one should cost that one factor, not the
    whole report -- so failures are dropped rather than raised. The caller
    sees which symbols are missing by their absence from the mapping.
    """
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame, _, _ = fetch_history(symbol, start, end, source)
            if len(frame):
                result[symbol] = frame
        except PriceDataError:
            continue
    return result


def build_observation(
    symbol: str,
    start: date,
    end: date,
    display_symbol: str | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    source: Source = "auto",
) -> PriceObservation:
    """Fetch a symbol and measure its move against its own recent volatility.

    Parameters
    ----------
    symbol : vendor spelling.
    start, end : the period the user asked about. Equal for a single day.
    display_symbol : the spelling to show the user; defaults to `symbol`.
    lookback : sessions of prior history used as the volatility baseline.
    source : see `fetch_history`.

    Raises
    ------
    PriceDataError : when no session exists at or before `end`, or when
        there is too little history to form a volatility baseline.
    """
    if lookback < _MIN_VOL_SESSIONS:
        raise ValueError(
            f"lookback must be at least {_MIN_VOL_SESSIONS} sessions to give a usable "
            f"standard deviation, got {lookback}"
        )

    pad_days = int(lookback * _CALENDAR_PAD_FACTOR) + 10
    fetch_start = start - timedelta(days=pad_days)
    frame, source_used, warnings = fetch_history(symbol, fetch_start, end, source)
    warnings = list(warnings)

    target_ts = pd.Timestamp(end)
    at_or_before = frame.index[frame.index <= target_ts]
    if len(at_or_before) == 0:
        raise PriceDataError(
            f"no trading session for {symbol} on or before {end.isoformat()}",
            suggestion=(
                "The symbol may not have existed then, or the date may predate its "
                "listing. Try a more recent date."
            ),
        )

    session_ts = at_or_before[-1]
    session_date = session_ts.date()
    if session_date != end:
        warnings.append(
            f"{end.isoformat()} was not a trading day for {display_symbol or symbol}; "
            f"this shows the most recent session, {session_date.isoformat()}"
        )

    position = frame.index.get_loc(session_ts)
    if position == 0:
        raise PriceDataError(
            f"only one session of history available for {symbol}; cannot compute a return",
            suggestion="Try a wider date range.",
        )

    # The volatility baseline stops at the session *before* the target, so a
    # large move cannot inflate its own denominator.
    context = frame.iloc[max(0, position - lookback) : position]
    if len(context) < _MIN_VOL_SESSIONS:
        raise PriceDataError(
            f"only {len(context)} sessions of prior history for {symbol}; need at least "
            f"{_MIN_VOL_SESSIONS} to judge whether a move is unusual",
            suggestion="This symbol may be newly listed. Try a more established one.",
        )

    context_returns = context["Close"].pct_change().dropna()
    daily_vol_pct = float(context_returns.std(ddof=1) * 100.0)

    row = frame.loc[session_ts]
    prior_close = float(frame["Close"].iloc[position - 1])
    close = float(row["Close"])
    return_pct = (close / prior_close - 1.0) * 100.0

    # Period return: from the close before the window opened, to the target
    # close. Using the first in-window close as the base would silently drop
    # the first day's move from a multi-day question.
    #
    # The window opens at min(start, session), not at `start`. When the
    # requested date is not a trading day the session steps *backwards* --
    # ask about a Sunday and you get Friday -- and measuring from the last
    # session before Sunday would make Friday its own baseline and report a
    # 0.00% move. That was a real bug: every weekend and holiday query
    # returned exactly zero.
    period_start = min(pd.Timestamp(start), session_ts)
    start_positions = frame.index[frame.index < period_start]
    if len(start_positions):
        period_base = float(frame.loc[start_positions[-1], "Close"])
    else:
        period_base = prior_close
    period_return_pct = (close / period_base - 1.0) * 100.0

    if daily_vol_pct <= 1e-9:
        sigma_multiple = 0.0
        warnings.append(
            f"{display_symbol or symbol} did not move at all over the prior "
            f"{len(context)} sessions, so 'unusual' cannot be measured"
        )
    else:
        sigma_multiple = period_return_pct / daily_vol_pct

    mean_volume = float(context["Volume"].mean())
    volume_ratio = float(row["Volume"] / mean_volume) if mean_volume > 0 else float("nan")

    if source_used == "mock" and not any("synthetic" in w for w in warnings):
        warnings.append("using synthetic data -- these numbers are NOT real market data")

    return PriceObservation(
        symbol=symbol,
        display_symbol=display_symbol or symbol,
        session_date=session_date,
        requested_date=end,
        open=float(row["Open"]),
        high=float(row["High"]),
        low=float(row["Low"]),
        close=close,
        volume=float(row["Volume"]),
        prior_close=prior_close,
        return_pct=return_pct,
        period_return_pct=period_return_pct,
        daily_vol_pct=daily_vol_pct,
        annualized_vol_pct=daily_vol_pct * float(np.sqrt(252.0)),
        sigma_multiple=sigma_multiple,
        outlier_label=_classify_outlier(sigma_multiple),
        volume_ratio=volume_ratio,
        context=context,
        history=frame,
        source=source_used,
        warnings=tuple(warnings),
    )
