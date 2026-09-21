"""Event-driven backtest loop where lookahead is structurally impossible.

The options work in this repository produced a leakage bug that made a model
look excellent until the discipline was enforced by index arithmetic rather
than by care. The lesson generalizes: *a convention that future data must
not be touched will eventually be violated, because nothing stops it.* So
this engine does not rely on a convention.

**The mechanism.** A strategy never receives the price history. It receives
a `SealedWindow`, which holds a frame that has already been truncated at the
current bar. The future rows are not hidden behind a flag or a private
attribute the strategy could reach past -- they are not in the object. There
is no code a strategy can write, deliberately or by accident, that returns
tomorrow's price, because tomorrow's price was never handed over.

**The second trap: execution timing.** A signal computed from bar *t*'s
close cannot be filled at bar *t*'s close -- by the time that price printed,
the session was over. Every order here is filled at bar *t+1*'s **open**.
This one detail is usually the difference between a moving-average crossover
that looks profitable and one that does not.

**The third trap: warm-up.** An indicator needing 200 bars has no value on
bar 5. Rather than letting strategies return a silent zero, the engine skips
the warm-up entirely and the equity curve starts where the strategy's first
real decision does, so no statistic is computed over a period the strategy
was not actually running.

Positions are expressed as a **target exposure** in [-1, 1]: 1.0 is fully
long, 0.0 is flat, -0.5 is half short. Working in exposure rather than share
counts keeps the strategy independent of account size and makes costs
proportional to the size of the change, which is how they actually work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .risk_metrics import PerformanceStats, compute_stats

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "LookaheadError",
    "SealedWindow",
    "Strategy",
    "run_backtest",
]


class LookaheadError(RuntimeError):
    """Raised when something tries to reach past a sealed window's edge.

    This should never fire in correct code. It exists because a guarantee
    that is never tested is a guarantee nobody can trust.
    """


class SealedWindow:
    """Price history truncated at a point in time, with no way past the edge.

    The defining property: **the future is absent, not merely hidden.** The
    frame this object holds ends at `as_of`. A strategy cannot index past
    it, cannot reach a private attribute holding more, and cannot ask the
    window for a date it does not contain without raising.

    Constructing one is cheap -- pandas slices are views, not copies -- so
    the engine builds a fresh window per bar rather than mutating a cursor,
    which would let a stale reference outlive its seal.
    """

    __slots__ = ("_as_of", "_frame")

    def __init__(self, frame: pd.DataFrame, end_position: int) -> None:
        if end_position < 0 or end_position >= len(frame):
            raise IndexError(
                f"end_position {end_position} is outside the frame (length {len(frame)})"
            )
        # The slice happens here, once. Everything downstream sees only this.
        self._frame = frame.iloc[: end_position + 1]
        self._as_of = frame.index[end_position]

    @property
    def frame(self) -> pd.DataFrame:
        """Every bar up to and including `as_of`, and nothing after it."""
        return self._frame

    @property
    def as_of(self) -> pd.Timestamp:
        """The most recent bar the strategy is allowed to know about."""
        return self._as_of

    @property
    def close(self) -> pd.Series:
        return self._frame["Close"]

    @property
    def latest_close(self) -> float:
        return float(self._frame["Close"].iloc[-1])

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return f"SealedWindow(bars={len(self._frame)}, as_of={self._as_of.date()})"

    def value_at(self, when: pd.Timestamp, column: str = "Close") -> float:
        """Look up one bar, refusing anything at or after the seal.

        The explicit accessor exists so that an attempt to read the future
        fails loudly instead of returning a KeyError that a caller might
        paper over.
        """
        stamp = pd.Timestamp(when)
        if stamp > self._as_of:
            raise LookaheadError(
                f"asked for {column} at {stamp.date()}, which is after this window's "
                f"seal at {self._as_of.date()}; the future is not available here"
            )
        if stamp not in self._frame.index:
            raise KeyError(f"no bar at {stamp.date()} in this window")
        return float(self._frame.loc[stamp, column])


@runtime_checkable
class Strategy(Protocol):
    """Anything that turns a sealed window into a target exposure.

    `warmup_bars` tells the engine how much history the strategy needs
    before its output means anything; the engine skips those bars rather
    than recording decisions the strategy could not really make.
    """

    name: str
    warmup_bars: int

    def signal(self, window: SealedWindow) -> float:
        """Target exposure in [-1, 1] for the *next* bar."""
        ...


@dataclass(frozen=True)
class BacktestConfig:
    """Execution assumptions. All costs are in basis points of traded notional.

    Attributes
    ----------
    initial_cash : starting portfolio value.
    commission_bps : broker commission per trade, on the notional traded.
    slippage_bps : the gap between the price you saw and the price you got.
        Defaulted to a non-zero value on purpose -- a backtest run at zero
        slippage is a different and much more flattering experiment, and
        making the user opt into that is the right default.
    allow_short : whether negative exposures are permitted. Off by default,
        since shorting has borrow costs and assignment risk this engine does
        not model.
    max_exposure : ceiling on absolute exposure, capping leverage at 1x.
    """

    initial_cash: float = 10_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    allow_short: bool = False
    max_exposure: float = 1.0

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {self.initial_cash}")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")
        if not 0 < self.max_exposure <= 3.0:
            raise ValueError(
                f"max_exposure must lie in (0, 3]; above 3x this engine's assumptions "
                f"(no margin calls, no financing) stop being defensible. Got "
                f"{self.max_exposure}"
            )

    @property
    def cost_rate(self) -> float:
        """Total round-trip-free cost per unit of notional traded."""
        return (self.commission_bps + self.slippage_bps) / 10_000.0


@dataclass
class BacktestResult:
    """Everything one backtest produced.

    Attributes
    ----------
    equity : portfolio value per bar, starting at the first traded bar.
    exposure : target exposure actually held per bar.
    benchmark_equity : buy-and-hold over the identical window, with the same
        entry cost applied. Comparing against a costless benchmark would
        flatter every strategy.
    stats, benchmark_stats : the headline statistics for each.
    trades : number of exposure changes.
    total_costs : cash paid in commission and slippage.
    strategy_name : for display.
    warmup_bars : bars skipped before trading began.
    """

    equity: pd.Series
    exposure: pd.Series
    benchmark_equity: pd.Series
    stats: PerformanceStats
    benchmark_stats: PerformanceStats
    trades: int
    total_costs: float
    strategy_name: str
    warmup_bars: int
    notes: list[str] = field(default_factory=list)

    @property
    def excess_return_pct(self) -> float:
        """Strategy total return minus buy-and-hold, in percentage points."""
        return self.stats.total_return_pct - self.benchmark_stats.total_return_pct

    @property
    def beat_benchmark(self) -> bool:
        return self.excess_return_pct > 0.0


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the price frame the backtest will run over."""
    required = {"Open", "Close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"price frame is missing {sorted(missing)}; the engine fills orders at the "
            "next bar's open, so Open is required as well as Close"
        )
    clean = frame.sort_index()
    clean = clean[np.isfinite(clean["Open"]) & np.isfinite(clean["Close"])]
    if (clean["Open"] <= 0).any() or (clean["Close"] <= 0).any():
        raise ValueError("price frame contains non-positive prices")
    return clean


def run_backtest(
    frame: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Run one strategy over one price frame.

    The loop is deliberately explicit. At bar *i* the strategy sees a window
    sealed at *i*, and the resulting exposure is filled at bar *i+1*'s open
    and marked at its close. No vectorized shortcut is used, because the
    vectorized forms of this calculation are exactly where an off-by-one
    turns into a lookahead that nobody notices.

    Raises
    ------
    ValueError : malformed price data, or a frame too short for the
        strategy's warm-up.
    LookaheadError : propagated from `SealedWindow` if a strategy reaches
        past its seal.
    """
    settings = config or BacktestConfig()
    prices = _prepare(frame)
    warmup = max(int(getattr(strategy, "warmup_bars", 1)), 1)

    if len(prices) < warmup + 2:
        raise ValueError(
            f"strategy {getattr(strategy, 'name', type(strategy).__name__)!r} needs "
            f"{warmup} warm-up bars plus at least 2 to trade; the frame has {len(prices)}"
        )

    opens = prices["Open"].to_numpy(dtype=float)
    closes = prices["Close"].to_numpy(dtype=float)
    index = prices.index

    cash = settings.initial_cash
    units = 0.0
    exposure_held = 0.0
    trades = 0
    total_costs = 0.0
    notes: list[str] = []

    equity_values: list[float] = []
    exposure_values: list[float] = []
    equity_index: list[pd.Timestamp] = []

    # Bar i produces a decision filled at bar i+1, so the loop stops one
    # short of the end. The final bar is marked, never traded on.
    for i in range(warmup - 1, len(prices) - 1):
        window = SealedWindow(prices, i)
        raw_signal = strategy.signal(window)

        if raw_signal is None or not np.isfinite(raw_signal):
            target = exposure_held
            notes.append(
                f"{index[i].date()}: strategy returned a non-finite signal; holding the "
                "previous exposure"
            )
        else:
            target = float(raw_signal)

        lower = -settings.max_exposure if settings.allow_short else 0.0
        target = float(np.clip(target, lower, settings.max_exposure))

        fill_price = opens[i + 1]
        mark_price = closes[i + 1]

        # Mark the portfolio at the fill price before resizing, so the trade
        # is sized against what the account is actually worth right then.
        portfolio_value = cash + units * fill_price
        desired_units = target * portfolio_value / fill_price
        delta_units = desired_units - units

        if abs(delta_units) > 1e-12:
            notional = abs(delta_units) * fill_price
            cost = notional * settings.cost_rate
            cash -= delta_units * fill_price + cost
            units = desired_units
            total_costs += cost
            if abs(target - exposure_held) > 1e-12:
                trades += 1
            exposure_held = target

        equity_values.append(cash + units * mark_price)
        exposure_values.append(exposure_held)
        equity_index.append(index[i + 1])

    if len(equity_values) < 3:
        raise ValueError(
            f"the backtest produced only {len(equity_values)} bars after warm-up; "
            "use a longer price history"
        )

    equity = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index), name="equity")
    exposure = pd.Series(exposure_values, index=equity.index, name="exposure")

    # Buy and hold over the identical window, paying the same entry cost.
    first_fill = opens[warmup]
    benchmark_units = (settings.initial_cash * (1.0 - settings.cost_rate)) / first_fill
    benchmark_equity = pd.Series(
        benchmark_units * closes[warmup : len(prices)][: len(equity)],
        index=equity.index,
        name="benchmark",
    )

    return BacktestResult(
        equity=equity,
        exposure=exposure,
        benchmark_equity=benchmark_equity,
        stats=compute_stats(equity, n_trades=trades, risk_free_rate=risk_free_rate),
        benchmark_stats=compute_stats(benchmark_equity, n_trades=1, risk_free_rate=risk_free_rate),
        trades=trades,
        total_costs=total_costs,
        strategy_name=getattr(strategy, "name", type(strategy).__name__),
        warmup_bars=warmup,
        notes=notes,
    )
