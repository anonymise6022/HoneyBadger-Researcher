"""Option chains: real quotes when the market can be reached, modelled when not.

**What is real.** Strikes, bid, ask, last, volume, open interest and the
vendor's implied volatility come from Yahoo through `yfinance`, the same
source the rest of the tool uses for prices. They are delayed -- roughly a
quarter of an hour on equity options, and stale overnight and at weekends --
which is stated on screen rather than left for the user to discover by
comparing against a broker.

**What is modelled.** With no network, or for a symbol with no listed
options, a chain is *constructed*: strikes on the real listing increments
around the last price, implied volatility from the underlying's own realized
volatility plus a skew, prices from Black-Scholes, and a bid-ask spread that
widens away from the money the way a real one does. It is labelled
`source="mock"` everywhere it surfaces. It exists so the desk can be used
offline, not so anyone can pretend to have traded.

**Three things a live chain does that a naive reading gets wrong.**

A quote of 0.00 x 0.05 is not a price; it is an empty book. Rows with no bid
are kept -- they are real listed contracts, and their open interest is
information -- but nothing that needs a price uses the midpoint of a book
with one side missing.

The vendor's implied volatility is not always usable. Yahoo publishes a
figure for every row including ones where the price cannot pin it down, and
it is occasionally a placeholder like 1e-5 or 3.0 on an untraded strike. It
is used when it is plausible and re-solved from the midpoint when it is not.

Time to expiry is measured to the close on expiry day, 4pm in New York, not
to midnight. On the last day that is the difference between a contract with
six hours of life in it and one with eighteen, which is most of what its
remaining time value is worth.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from .greeks import Greeks, black_scholes_price, greeks, implied_volatility

__all__ = [
    "Chain",
    "ChainError",
    "ContractQuote",
    "RISK_FREE_RATE",
    "fetch_chain",
    "list_expiries",
]

#: Discount rate for pricing. A constant rather than a live curve: at the
#: tenors a retail desk trades, rho is the smallest of the greeks by an order
#: of magnitude, and a wrong rate moves a one-month option by less than a
#: cent. It is the one input here not worth a network call.
RISK_FREE_RATE = 0.04

#: Implied volatility outside this band is a vendor placeholder, not a quote.
_IV_FLOOR, _IV_CEILING = 0.01, 4.0

_NEW_YORK_CLOSE = time(16, 0)
_MIN_YEARS = 1.0 / (365.0 * 24.0)  # one hour, so a expiring contract still prices


class ChainError(RuntimeError):
    """No chain could be produced, with something the user can act on."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.suggestion = suggestion


def _expiry_moment(expiry: date) -> datetime:
    """When the contract stops trading, as an instant.

    Options cease trading at the New York close on expiry day. Using
    midnight instead -- the default if you subtract two dates -- overstates
    the life of every contract by the last eight hours of its existence,
    which on expiry day is most of what is left.
    """
    try:
        from zoneinfo import ZoneInfo

        return datetime.combine(expiry, _NEW_YORK_CLOSE, tzinfo=ZoneInfo("America/New_York"))
    except Exception:  # pragma: no cover - only without a tz database
        # 20:00 UTC is the New York close on daylight time, and an hour out
        # for the winter months. An hour at these tenors is immaterial.
        return datetime.combine(expiry, time(20, 0), tzinfo=timezone.utc)


@dataclass(frozen=True)
class ContractQuote:
    """One listed contract, as the market last showed it."""

    kind: str            # "call" or "put"
    strike: float
    bid: float | None
    ask: float | None
    last: float | None
    vendor_iv: float | None
    volume: int
    open_interest: int

    @property
    def mid(self) -> float | None:
        """The midpoint, only when both sides of the book are there.

        A one-sided book has no midpoint. Treating a missing bid as zero
        halves the apparent price of every illiquid contract on the screen,
        which is exactly the kind of quiet error that makes a paper account
        profitable and a real one not.
        """
        if self.bid is None or self.ask is None or self.ask <= 0:
            return None
        if self.bid <= 0:
            return None
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= 0:
            return None
        return max(self.ask - self.bid, 0.0)

    @property
    def spread_pct(self) -> float | None:
        """The spread as a fraction of the midpoint -- the cost of a round trip."""
        mid, spread = self.mid, self.spread
        if mid is None or spread is None or mid <= 0:
            return None
        return spread / mid

    @property
    def tradeable(self) -> bool:
        """Whether an order could be filled against this book at all."""
        return self.bid is not None and self.ask is not None and self.ask > 0

    def fill_price(self, side: str) -> float | None:
        """What a market order actually pays: the ask to buy, the bid to sell.

        Not the midpoint. Filling paper trades at the midpoint is the single
        most flattering assumption a simulator can make -- on a contract
        quoted 1.05 x 1.35 it hands the trader 15 cents a share, $15 a
        contract, that the real book was never going to give them.
        """
        if side == "buy":
            return self.ask if self.ask and self.ask > 0 else None
        if side == "sell":
            return self.bid if self.bid and self.bid > 0 else None
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def implied_vol(self, spot: float, years: float, rate: float = RISK_FREE_RATE) -> tuple[float | None, str]:
        """Implied volatility and where it came from.

        The vendor's figure is preferred when it is inside a plausible band,
        because it is derived from quotes this code never sees. Outside that
        band it is a placeholder, and the midpoint is re-solved instead. When
        neither works the answer is None, and the greeks that depend on it
        are simply not shown.
        """
        if self.vendor_iv is not None and _IV_FLOOR <= self.vendor_iv <= _IV_CEILING:
            return self.vendor_iv, "vendor"
        mid = self.mid
        if mid is not None:
            solved = implied_volatility(mid, spot, self.strike, years, rate, self.kind)
            if solved is not None:
                return solved, "solved from the midpoint"
        return None, "unavailable"

    def greeks(self, spot: float, years: float, rate: float = RISK_FREE_RATE) -> Greeks | None:
        vol, _ = self.implied_vol(spot, years, rate)
        if vol is None:
            return None
        return greeks(spot, self.strike, years, rate, vol, self.kind)


@dataclass(frozen=True)
class Chain:
    """Every listed contract at one expiry, with the underlying beside it."""

    symbol: str
    expiry: date
    spot: float
    quoted_at: datetime
    source: str                       # "yfinance" or "mock"
    rows: tuple[ContractQuote, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def synthetic(self) -> bool:
        return self.source == "mock"

    def years_to_expiry(self, now: datetime | None = None) -> float:
        moment = now or datetime.now(timezone.utc)
        seconds = (_expiry_moment(self.expiry) - moment).total_seconds()
        return max(seconds / (365.0 * 24.0 * 3600.0), _MIN_YEARS)

    def days_to_expiry(self, now: datetime | None = None) -> int:
        moment = (now or datetime.now(timezone.utc)).date()
        return max((self.expiry - moment).days, 0)

    def find(self, kind: str, strike: float) -> ContractQuote | None:
        """The contract at this strike, matched with a tolerance.

        Strikes arrive as floats from two different places -- a vendor's
        JSON and the interface's own round trip through JavaScript -- and
        112.5 does not always survive that identically. Matching to a tenth
        of a cent avoids a "no such contract" on a contract plainly on screen.
        """
        for row in self.rows:
            if row.kind == kind and abs(row.strike - strike) < 1e-3:
                return row
        return None


# ----------------------------------------------------------------- live

def _normalise_expiry(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _clean(value: object, *, cast: type = float) -> object | None:
    """Vendor cells arrive as numpy scalars, NaN, None or empty strings."""
    if value is None:
        return None
    try:
        result = cast(value)
    except (TypeError, ValueError):
        return None
    if isinstance(result, float) and not math.isfinite(result):
        return None
    return result


def _live_expiries(symbol: str) -> list[date]:
    import yfinance  # noqa: PLC0415 - optional at import time, required here

    ticker = yfinance.Ticker(symbol)
    raw = getattr(ticker, "options", ()) or ()
    return [_normalise_expiry(value) for value in raw]


def _live_spot(symbol: str) -> float | None:
    """The most recent trade, falling back to the last daily close."""
    import yfinance  # noqa: PLC0415

    ticker = yfinance.Ticker(symbol)
    try:
        price = _clean(ticker.fast_info.get("last_price"))
        if price and price > 0:
            return float(price)
    except Exception:  # noqa: BLE001 - fast_info is best-effort by design
        pass
    try:
        history = ticker.history(period="5d", auto_adjust=False)
        if len(history):
            return float(history["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def _live_chain(symbol: str, expiry: date) -> Chain:
    import yfinance  # noqa: PLC0415

    ticker = yfinance.Ticker(symbol)
    try:
        table = ticker.option_chain(expiry.isoformat())
    except Exception as exc:  # noqa: BLE001 - vendor raises bare exceptions
        raise ChainError(
            f"No option chain came back for {symbol} at {expiry.isoformat()}.",
            "The expiry may have been delisted; pick another date.",
        ) from exc

    rows: list[ContractQuote] = []
    for frame, kind in ((table.calls, "call"), (table.puts, "put")):
        for record in frame.to_dict("records"):
            strike = _clean(record.get("strike"))
            if not strike or strike <= 0:
                continue
            rows.append(
                ContractQuote(
                    kind=kind,
                    strike=float(strike),
                    bid=_clean(record.get("bid")),
                    ask=_clean(record.get("ask")),
                    last=_clean(record.get("lastPrice")),
                    vendor_iv=_clean(record.get("impliedVolatility")),
                    volume=int(_clean(record.get("volume"), cast=float) or 0),
                    open_interest=int(_clean(record.get("openInterest"), cast=float) or 0),
                )
            )
    if not rows:
        raise ChainError(
            f"{symbol} has no listed contracts at {expiry.isoformat()}.",
            "Try a different expiry.",
        )

    spot = _live_spot(symbol)
    if not spot:
        raise ChainError(
            f"Could not price the underlying for {symbol}.",
            "Check the symbol, or try again in a moment.",
        )
    return Chain(
        symbol=symbol,
        expiry=expiry,
        spot=float(spot),
        quoted_at=datetime.now(timezone.utc),
        source="yfinance",
        rows=tuple(sorted(rows, key=lambda r: (r.kind, r.strike))),
        warnings=(
            "Quotes are delayed by roughly 15 minutes, and are last night's "
            "book while the market is shut.",
        ),
    )


# ---------------------------------------------------------------- modelled

def _seed(symbol: str, expiry: date) -> int:
    """A stable seed. Python's own hash() is salted per process, so using it
    would redraw the whole synthetic chain on every launch."""
    digest = hashlib.sha256(f"{symbol}:{expiry.isoformat()}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _strike_step(spot: float) -> float:
    """Listing increments, roughly as the exchanges set them."""
    if spot < 25:
        return 0.5
    if spot < 75:
        return 1.0
    if spot < 200:
        return 2.5
    if spot < 500:
        return 5.0
    return 10.0


def _mock_spot_and_vol(symbol: str) -> tuple[float, float]:
    """Last price and realized volatility from the same synthetic history the
    rest of the app draws, so the desk and the price chart agree."""
    from ..data.mock_data import mock_price_history  # noqa: PLC0415

    end = date.today()  # noqa: DTZ011
    frame = mock_price_history(symbol, end - timedelta(days=200), end)
    closes = [float(value) for value in frame["Close"].to_numpy()]
    spot = closes[-1]
    returns = [math.log(b / a) for a, b in zip(closes[-61:], closes[-60:]) if a > 0 and b > 0]
    if len(returns) > 2:
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        realized = math.sqrt(variance * 252.0)
    else:
        realized = 0.25
    # Options trade above realized volatility nearly all of the time; 1.1 is
    # about the long-run gap on index options and the direction is not in
    # doubt even if the size is.
    return spot, max(min(realized * 1.1, 1.5), 0.08)


def _smile(base_vol: float, spot: float, strike: float, years: float) -> float:
    """A downward skew: puts below the money bid up, calls above it cheaper.

    Equity index options have traded with this shape continuously since
    1987, and a flat volatility surface would make every out-of-the-money
    put on the screen look like a bargain.
    """
    moneyness = math.log(strike / spot)
    skew = -0.35 * moneyness + 0.6 * moneyness * moneyness
    # Short tenors carry more of the smile than long ones.
    intensity = min(1.0, math.sqrt(0.25 / max(years, 0.02)))
    return max(base_vol * (1.0 + skew * intensity), 0.05)


def _tick(price: float) -> float:
    """Options quote in pennies below $3 and nickels above it."""
    return 0.01 if price < 3.0 else 0.05


def _round_to(value: float, tick: float) -> float:
    return round(round(value / tick) * tick, 2)


def _mock_chain(symbol: str, expiry: date) -> Chain:
    spot, base_vol = _mock_spot_and_vol(symbol)
    chain = Chain(
        symbol=symbol, expiry=expiry, spot=spot,
        quoted_at=datetime.now(timezone.utc), source="mock", rows=(),
    )
    years = chain.years_to_expiry()
    rng = random.Random(_seed(symbol, expiry))

    step = _strike_step(spot)
    centre = round(spot / step) * step
    strikes = [centre + step * i for i in range(-12, 13) if centre + step * i > 0]

    rows: list[ContractQuote] = []
    for strike in strikes:
        vol = _smile(base_vol, spot, strike, years)
        distance = abs(math.log(strike / spot))
        for kind in ("call", "put"):
            theoretical = black_scholes_price(spot, strike, years, RISK_FREE_RATE, vol, kind)
            tick = _tick(theoretical)
            # Spreads widen away from the money and never go below one tick.
            half = max(tick, theoretical * (0.015 + 1.1 * distance))
            half = min(half, max(theoretical * 0.45, tick))
            bid = _round_to(max(theoretical - half, 0.0), tick)
            ask = _round_to(theoretical + half, tick)
            if ask <= 0:
                ask = tick
            # Interest concentrates at the money and on round strikes.
            weight = math.exp(-(distance / 0.12) ** 2)
            open_interest = int(weight * rng.uniform(400, 4000) + rng.uniform(0, 60))
            volume = int(open_interest * rng.uniform(0.02, 0.35))
            rows.append(
                ContractQuote(
                    kind=kind, strike=float(strike),
                    bid=bid if bid > 0 else None, ask=ask,
                    last=_round_to(theoretical, tick) if theoretical > 0 else None,
                    vendor_iv=vol, volume=volume, open_interest=open_interest,
                )
            )

    return Chain(
        symbol=symbol, expiry=expiry, spot=spot,
        quoted_at=chain.quoted_at, source="mock",
        rows=tuple(sorted(rows, key=lambda r: (r.kind, r.strike))),
        warnings=(
            "These contracts do not exist. Strikes, prices and spreads are "
            "generated from the underlying's synthetic history.",
        ),
    )


def _mock_expiries(today: date | None = None) -> list[date]:
    """Weekly expiries for two months, then the monthly third Friday.

    Which is close to what a liquid name actually lists, and keeps the
    offline expiry picker from looking like a toy.
    """
    start = today or date.today()  # noqa: DTZ011
    dates: list[date] = []
    friday = start + timedelta(days=(4 - start.weekday()) % 7)
    for week in range(9):
        dates.append(friday + timedelta(weeks=week))
    month = friday.replace(day=1)
    for _ in range(6):
        month = (month + timedelta(days=32)).replace(day=1)
        third_friday = month + timedelta(days=(4 - month.weekday()) % 7 + 14)
        if third_friday not in dates:
            dates.append(third_friday)
    return sorted(dates)


# ------------------------------------------------------------------ public

def list_expiries(symbol: str, source: str = "auto") -> tuple[list[date], str, tuple[str, ...]]:
    """Listed expiries, the source they came from, and any warnings.

    Follows the same contract as `price_ingest.fetch_history`: "live" fails
    rather than substituting, "mock" never touches the network, and "auto"
    tries live and falls back with a warning attached.
    """
    symbol = symbol.upper().strip()
    if not symbol:
        raise ChainError("No symbol given.", "Try SPY, AAPL or NVDA.")
    if source == "mock":
        return _mock_expiries(), "mock", ()

    try:
        expiries = _live_expiries(symbol)
    except Exception as exc:  # noqa: BLE001 - network, parsing, vendor, all the same here
        if source == "live":
            raise ChainError(
                f"Could not reach the options market for {symbol}.",
                "Check your internet connection.",
            ) from exc
        return _mock_expiries(), "mock", (
            f"Could not reach the options market for {symbol}; showing a synthetic chain.",
        )

    if not expiries:
        if source == "live":
            raise ChainError(
                f"{symbol} has no listed options.",
                "Index funds and large caps have the deepest chains -- try SPY or AAPL.",
            )
        return _mock_expiries(), "mock", (
            f"{symbol} has no listed options; showing a synthetic chain.",
        )
    return expiries, "yfinance", ()


def fetch_chain(symbol: str, expiry: date | str, source: str = "auto") -> Chain:
    """The full chain at one expiry."""
    symbol = symbol.upper().strip()
    expiry = _normalise_expiry(expiry)
    if source == "mock":
        return _mock_chain(symbol, expiry)

    try:
        return _live_chain(symbol, expiry)
    except ChainError:
        if source == "live":
            raise
        return _mock_chain(symbol, expiry)
    except Exception as exc:  # noqa: BLE001
        if source == "live":
            raise ChainError(
                f"Could not reach the options market for {symbol}.",
                "Check your internet connection.",
            ) from exc
        return _mock_chain(symbol, expiry)
