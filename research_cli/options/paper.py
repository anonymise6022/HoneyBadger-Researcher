"""A paper options account, kept as close to a real one as simulation allows.

The point of paper trading is to find out whether a plan survives contact
with the mechanics. A simulator that fills at the midpoint, ignores
commission, lets a short position be opened with no collateral and forgets
about expiry teaches the opposite of the lesson, because every one of those
shortcuts pays the trader money the market would not have.

So the rules here are the broker's, not the modeller's convenience:

* **Fills cross the spread.** Buying pays the ask, selling receives the bid.
* **Commission is charged both ways**, per contract, at a rate a retail
  broker actually charges. Two dollars a round trip on a single contract is
  a fifth of the profit on a ten-cent winner.
* **Short positions are collateralised before they are allowed.** A short
  put ties up the cash to buy the shares at the strike; a short call ties up
  a Regulation T margin requirement. Neither can be opened on an account
  that could not cover it, which is the whole risk of being short an option.
* **Positions are marked at what it would cost to get out** -- the bid for a
  long, the ask for a short -- not at the midpoint. The account's value is
  what it could be liquidated for, not what it looks like.
* **Expiry happens whether or not anyone is watching.** Contracts past their
  expiry settle at intrinsic value against the underlying's price the next
  time the account is opened, with assignment on anything short and in the
  money.

Two things a real account has that this one does not, both stated on screen
rather than hidden: **early assignment**, which a short American option can
suffer at any moment and which is modelled only at expiry; and **shares**,
so an exercised call settles in cash rather than delivering stock.

Everything here is standard library, and the account is a plain JSON file,
so it can be read, backed up or deleted by hand.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .greeks import CONTRACT_SIZE, intrinsic_value

__all__ = [
    "Account",
    "OrderError",
    "OrderTicket",
    "Position",
    "STARTING_CASH",
    "account_path",
    "load_account",
    "save_account",
]

#: What a mainstream retail broker charges per contract, each way.
COMMISSION_PER_CONTRACT = 0.65

#: The opening balance, which is what the mainstream paper platforms hand
#: out. It is not generosity: a cash-secured put on a $300 stock ties up
#: $30,000, so an account much smaller than this cannot open the positions
#: the desk exists to teach, and every short would be refused for reasons
#: that look like a bug rather than a lesson.
STARTING_CASH = 100_000.0

#: Regulation T's requirement for an uncovered short call: the greater of
#: 20% of the underlying less the amount out of the money, and 10% of the
#: underlying, plus the premium received.
_REG_T_PRIMARY = 0.20
_REG_T_FLOOR = 0.10


class OrderError(ValueError):
    """An order that a broker would have rejected, with the reason."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.suggestion = suggestion


def account_path() -> Path:
    """Beside the credentials, so everything this tool remembers is in one place."""
    from ..settings import CONFIG_DIR  # noqa: PLC0415 - avoids a cycle at import

    return CONFIG_DIR / "paper_options.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Position:
    """One open line in the account. Negative contracts are short."""

    id: str
    symbol: str
    expiry: str          # ISO date
    strike: float
    kind: str            # "call" or "put"
    contracts: int       # signed
    average_price: float  # per share, the fill price
    opened_at: str       # ISO timestamp
    collateral: float    # cash held against a short, zero for a long

    @property
    def is_short(self) -> bool:
        return self.contracts < 0

    @property
    def shares(self) -> int:
        return abs(self.contracts) * CONTRACT_SIZE

    @property
    def expiry_date(self) -> date:
        return date.fromisoformat(self.expiry)

    @property
    def cost_basis(self) -> float:
        """Signed: what was paid for a long, what was received for a short."""
        return self.average_price * self.contracts * CONTRACT_SIZE

    def label(self) -> str:
        """`AAPL 20 Jun 190 C`, the way a blotter prints it."""
        pretty = self.expiry_date.strftime("%d %b %y")
        strike = f"{self.strike:g}"
        return f"{self.symbol} {pretty} {strike} {self.kind[0].upper()}"

    def value_at(self, close_price: float) -> float:
        """What the line is worth if closed at this price per share."""
        return close_price * self.contracts * CONTRACT_SIZE

    def unrealized(self, close_price: float) -> float:
        return self.value_at(close_price) - self.cost_basis


@dataclass(frozen=True)
class OrderTicket:
    """What an order will cost before it is placed.

    Shown on the ticket rather than computed after the fact, because the
    cash effect of selling an option -- a credit, plus collateral that
    disappears from buying power -- is the part of options trading that
    surprises people.
    """

    side: str            # "buy" or "sell"
    contracts: int
    fill_price: float    # per share
    premium: float       # contracts x 100 x price, unsigned
    commission: float
    collateral: float    # held against a short, zero for a long
    cash_effect: float   # signed change in cash
    buying_power_effect: float
    note: str = ""


@dataclass
class Account:
    """The whole account: cash, open lines and everything that has happened."""

    cash: float = STARTING_CASH
    starting_cash: float = STARTING_CASH
    realized: float = 0.0
    positions: list[Position] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    sequence: int = 0
    created_at: str = field(default_factory=lambda: _now().isoformat())

    # --- money ----------------------------------------------------------

    @property
    def collateral_held(self) -> float:
        return sum(position.collateral for position in self.positions)

    @property
    def buying_power(self) -> float:
        """Cash that is not already promised to a short position."""
        return self.cash - self.collateral_held

    def equity(self, marks: dict[str, float]) -> float:
        """Cash plus what every open line could be closed for.

        `marks` is position id to price per share, from `mark_price`.
        """
        return self.cash + sum(
            position.value_at(marks[position.id])
            for position in self.positions
            if position.id in marks
        )

    # --- orders ---------------------------------------------------------

    def quote_order(
        self, side: str, contracts: int, fill_price: float, strike: float, kind: str, spot: float
    ) -> OrderTicket:
        """Price an order without placing it."""
        if contracts <= 0:
            raise OrderError("Order size must be at least one contract.")
        if fill_price is None or fill_price <= 0:
            raise OrderError(
                "There is no price on the other side of this book.",
                "Nothing is bid or offered here; pick a strike nearer the money.",
            )

        premium = fill_price * contracts * CONTRACT_SIZE
        commission = COMMISSION_PER_CONTRACT * contracts
        opening_short = side == "sell"
        collateral = (
            short_requirement(kind, strike, contracts, spot, fill_price) if opening_short else 0.0
        )
        cash_effect = (premium if opening_short else -premium) - commission
        return OrderTicket(
            side=side,
            contracts=contracts,
            fill_price=fill_price,
            premium=premium,
            commission=commission,
            collateral=collateral,
            cash_effect=cash_effect,
            buying_power_effect=cash_effect - collateral,
            note=(
                "Cash secured at the strike."
                if opening_short and kind == "put"
                else "Regulation T margin on an uncovered call."
                if opening_short
                else ""
            ),
        )

    def open_position(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        kind: str,
        side: str,
        contracts: int,
        fill_price: float,
        spot: float,
    ) -> tuple[Position, OrderTicket]:
        """Place an opening order, or refuse it the way a broker would."""
        ticket = self.quote_order(side, contracts, fill_price, strike, kind, spot)
        needed = -ticket.buying_power_effect
        if needed > self.buying_power + 1e-9:
            raise OrderError(
                f"This order needs ${needed:,.2f} of buying power and the account has "
                f"${self.buying_power:,.2f}.",
                "Reduce the size, or close something first.",
            )

        self.sequence += 1
        position = Position(
            id=f"p{self.sequence}",
            symbol=symbol.upper(),
            expiry=expiry,
            strike=float(strike),
            kind=kind,
            contracts=contracts if side == "buy" else -contracts,
            average_price=fill_price,
            opened_at=_now().isoformat(),
            collateral=ticket.collateral,
        )
        self.cash += ticket.cash_effect
        # Commission is a realized cost the moment it is charged. Leaving it
        # out of `realized` until the position closes would break the one
        # identity the blotter has to satisfy: realized plus unrealized
        # equals equity less what the account started with.
        self.realized -= ticket.commission
        self.positions.append(position)
        self._record(
            "open", position, contracts, fill_price, ticket.commission, ticket.cash_effect,
            f"{'Bought' if side == 'buy' else 'Sold'} {contracts} at {fill_price:.2f}"
            + (" (paid the ask)" if side == "buy" else " (hit the bid)"),
        )
        return position, ticket

    def close_position(
        self, position_id: str, fill_price: float, contracts: int | None = None
    ) -> dict[str, Any]:
        """Close all or part of a line at the price the book offers."""
        position = self.find(position_id)
        if position is None:
            raise OrderError("That position is not open any more.")
        size = abs(position.contracts) if contracts is None else int(contracts)
        if size <= 0 or size > abs(position.contracts):
            raise OrderError(
                f"This position is {abs(position.contracts)} contracts; "
                f"cannot close {size}."
            )
        if fill_price is None or fill_price <= 0:
            raise OrderError(
                "Nothing is quoted on the side you would have to trade against.",
                "An illiquid contract can be impossible to get out of -- which is the "
                "point of the exercise.",
            )

        share_count = size * CONTRACT_SIZE
        commission = COMMISSION_PER_CONTRACT * size
        # Closing a long sells (credit); closing a short buys back (debit).
        proceeds = fill_price * share_count * (1 if position.contracts > 0 else -1)
        released = position.collateral * (size / abs(position.contracts))
        entry_value = position.average_price * share_count * (1 if position.contracts > 0 else -1)
        realized = proceeds - entry_value - commission

        self.cash += proceeds - commission
        self.realized += realized
        self._replace_or_drop(position, size, released)
        self._record(
            "close", position, size, fill_price, commission, proceeds - commission,
            f"Closed {size} at {fill_price:.2f}", realized=realized,
        )
        return {"realized": realized, "commission": commission, "contracts": size}

    # --- expiry -----------------------------------------------------------

    def settle_expired(
        self, price_on: "Callable[[str, date], float | None]", today: date | None = None
    ) -> list[dict[str, Any]]:
        """Settle everything past its expiry against the underlying's price.

        `price_on` is asked for the underlying's close on each contract's own
        expiry day, not for today's price. A position that expired three
        weeks ago settled against the market of three weeks ago, and marking
        it against today would invent a gain or a loss that never happened.

        Cash settled, including on a long call that a real account would
        exercise into shares. The difference matters to anyone planning to
        hold the stock afterwards, so it is said on screen.
        """
        moment = today or _now().date()
        events: list[dict[str, Any]] = []
        for position in list(self.positions):
            if position.expiry_date >= moment:
                continue
            spot = price_on(position.symbol, position.expiry_date)
            if spot is None:
                continue  # cannot settle what cannot be priced; try again later
            value = intrinsic_value(spot, position.strike, position.kind)
            share_count = position.shares
            proceeds = value * share_count * (1 if position.contracts > 0 else -1)
            realized = proceeds - position.cost_basis

            # Collateral is a hold on cash rather than a separate pot, so
            # removing the position below is what releases it.
            self.cash += proceeds
            self.realized += realized
            self._replace_or_drop(position, abs(position.contracts), position.collateral)

            if value <= 0:
                outcome = "expired worthless"
            elif position.contracts > 0:
                outcome = f"expired in the money and was exercised for {value:.2f} a share"
            else:
                outcome = f"was assigned at {value:.2f} a share"
            self._record(
                "expiry", position, abs(position.contracts), value, 0.0, proceeds,
                f"{position.label()} {outcome}", realized=realized,
            )
            events.append({
                "position": position.label(),
                "outcome": outcome,
                "realized": realized,
                "settlement_price": value,
                "underlying": spot,
            })
        return events

    # --- housekeeping -----------------------------------------------------

    def find(self, position_id: str) -> Position | None:
        return next((p for p in self.positions if p.id == position_id), None)

    def symbols(self) -> list[str]:
        return sorted({position.symbol for position in self.positions})

    def reset(self) -> None:
        self.cash = STARTING_CASH
        self.starting_cash = STARTING_CASH
        self.realized = 0.0
        self.positions = []
        self.ledger = []
        self.sequence = 0
        self.created_at = _now().isoformat()

    def _replace_or_drop(self, position: Position, closed: int, released_collateral: float) -> None:
        remaining = abs(position.contracts) - closed
        self.positions = [p for p in self.positions if p.id != position.id]
        if remaining > 0:
            sign = 1 if position.contracts > 0 else -1
            self.positions.append(
                replace(
                    position,
                    contracts=sign * remaining,
                    collateral=max(position.collateral - released_collateral, 0.0),
                )
            )

    def _record(
        self, kind: str, position: Position, contracts: int, price: float,
        commission: float, cash_effect: float, note: str, realized: float | None = None,
    ) -> None:
        self.ledger.append({
            "at": _now().isoformat(),
            "kind": kind,
            "contract": position.label(),
            "symbol": position.symbol,
            "contracts": contracts,
            "price": round(price, 4),
            "commission": round(commission, 2),
            "cash_effect": round(cash_effect, 2),
            "realized": None if realized is None else round(realized, 2),
            "note": note,
        })
        # A blotter nobody can scroll is no use; keep the recent history.
        del self.ledger[:-200]

    # --- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "cash": self.cash,
            "starting_cash": self.starting_cash,
            "realized": self.realized,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "positions": [asdict(position) for position in self.positions],
            "ledger": self.ledger,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Account":
        return cls(
            cash=float(payload.get("cash", STARTING_CASH)),
            starting_cash=float(payload.get("starting_cash", STARTING_CASH)),
            realized=float(payload.get("realized", 0.0)),
            sequence=int(payload.get("sequence", 0)),
            created_at=str(payload.get("created_at", _now().isoformat())),
            positions=[Position(**row) for row in payload.get("positions", [])],
            ledger=list(payload.get("ledger", [])),
        )


def short_requirement(
    kind: str, strike: float, contracts: int, spot: float, premium_per_share: float
) -> float:
    """Collateral a broker holds against one short option line.

    A short put is cash secured: the account must be able to buy the shares
    at the strike, which is the worst case and is exact. A short call has no
    worst case -- the stock can go anywhere -- so Regulation T asks for 20%
    of the underlying less however far out of the money the strike sits,
    floored at 10%, plus the premium taken in.
    """
    shares = contracts * CONTRACT_SIZE
    if kind == "put":
        return strike * shares
    out_of_the_money = max(strike - spot, 0.0)
    primary = (_REG_T_PRIMARY * spot - out_of_the_money + premium_per_share) * shares
    floor = (_REG_T_FLOOR * spot + premium_per_share) * shares
    return max(primary, floor, 0.0)


def mark_price(
    position: Position, bid: float | None, ask: float | None, intrinsic: float = 0.0
) -> float:
    """What the line marks at: the price it could be closed at, not the mid.

    A long is worth what someone will pay for it, which is the bid. A short
    costs what someone will take to hand it back, which is the ask. Marking
    both at the midpoint makes every position look a half-spread better than
    it is, and on a wide book that is most of the position's profit.

    Intrinsic value is the floor either way. An in-the-money contract with an
    empty book is not worthless, and a short one is not free to close; with
    no quote on the closing side, what it is certainly worth is what it would
    settle for today.
    """
    if position.contracts > 0:
        return max(bid if bid and bid > 0 else 0.0, intrinsic)
    return max(ask if ask and ask > 0 else 0.0, intrinsic)


def load_account(path: Path | None = None) -> Account:
    """Read the saved account. A corrupt file starts a fresh one, never crashes."""
    target = path or account_path()
    try:
        if not target.is_file():
            return Account()
        return Account.from_dict(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return Account()


def save_account(account: Account, path: Path | None = None) -> Path:
    """Write the account atomically, so an interrupted save cannot lose it."""
    target = path or account_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(account.to_dict(), stream, indent=2)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target
