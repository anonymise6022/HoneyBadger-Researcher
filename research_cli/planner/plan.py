"""Build two or three plan options for a decision the user has already made.

Not one "optimal" answer: each option trades something for something else,
and the trade is stated. Every figure is computed here from price history;
the report only formats them, and the validator checks the report against
`Plan.numbers()`.

    A  Size to your loss limit   invest only what a bad stretch would cost
                                 your limit; keep the rest in cash
    B  Full amount, exit rule    invest it all, with an exit if it falls by
                                 your limit; replayed on history, gaps included
    C  Spread the entry out      option A's amount in parts, a few weeks
                                 apart; compared with putting it in at once
                                 using the backtest engine (next-open fills,
                                 costs included), over many past start dates

The tool never places an order. These are descriptions of plans.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestConfig, SealedWindow, run_backtest
from .inputs import PlanRequest
from .sizing import HorizonRisk, StopOutcome, horizon_risk, size_for_loss, stop_outcomes

__all__ = ["DollarRange", "Plan", "PlanOption", "StagedComparison", "build_plan"]

_TRANCHES = 4
_MAX_STARTS = 120


@dataclass(frozen=True)
class DollarRange:
    """Gain or loss in dollars on the invested amount: bad, typical, good, worst."""

    bad: float
    typical: float
    good: float
    worst: float

    @classmethod
    def from_pct(cls, invested: float, p10: float, median: float, p90: float, worst: float):
        return cls(*(invested * x / 100.0 for x in (p10, median, p90, worst)))


@dataclass(frozen=True)
class StagedComparison:
    starts: int
    tranches: int
    spacing_days: int
    staged_beat_share_pct: float
    staged_median: float
    lump_median: float
    staged_worst: float
    lump_worst: float


@dataclass(frozen=True)
class PlanOption:
    key: str
    title: str
    invest: float
    cash: float
    outcome: DollarRange
    stop_pct: float | None = None
    stop_price: float | None = None
    stop: StopOutcome | None = None
    staged: StagedComparison | None = None


@dataclass(frozen=True)
class Plan:
    request: PlanRequest
    risk: HorizonRisk
    options: tuple[PlanOption, ...]
    history_sessions: int
    is_synthetic: bool

    def numbers(self) -> frozenset[float]:
        """Every figure this plan holds -- the planner's `extra_numbers`."""
        found: set[float] = set()

        def walk(value: object) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, (int, float)):
                found.add(float(value))
            elif is_dataclass(value):
                for item in fields(value):
                    walk(getattr(value, item.name))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    walk(item)

        walk(self)
        found.add(self.request.max_loss_pct)
        return frozenset(found)


class _StagedEntry:
    """Exposure that steps up in equal parts, `spacing` bars apart."""

    name = "Staged entry"
    warmup_bars = 1

    def __init__(self, tranches: int, spacing: int) -> None:
        self.tranches, self.spacing = tranches, spacing

    def signal(self, window: SealedWindow) -> float:
        elapsed = len(window) - 1
        return min(1.0, (elapsed // self.spacing + 1) / self.tranches)


def _compare_staged(
    frame: pd.DataFrame, horizon: int, invest: float
) -> tuple[StagedComparison, DollarRange]:
    spacing = max(1, min(21, horizon // (_TRANCHES + 1)))
    last_start = len(frame) - horizon - 2
    step = max(spacing, last_start // _MAX_STARTS)
    staged, lump = [], []
    for start in range(0, last_start, step):
        window = frame.iloc[start : start + horizon + 2]
        result = run_backtest(window, _StagedEntry(_TRANCHES, spacing), BacktestConfig(initial_cash=invest))
        staged.append(float(result.equity.iloc[-1]))
        lump.append(float(result.benchmark_equity.iloc[-1]))
    s, b = np.array(staged), np.array(lump)
    comparison = StagedComparison(
        starts=len(s), tranches=_TRANCHES, spacing_days=spacing,
        staged_beat_share_pct=float((s > b).mean() * 100.0),
        staged_median=float(np.median(s)), lump_median=float(np.median(b)),
        staged_worst=float(s.min()), lump_worst=float(b.min()),
    )
    gains = s - invest
    outcome = DollarRange(
        float(np.percentile(gains, 10)), float(np.median(gains)),
        float(np.percentile(gains, 90)), float(gains.min()),
    )
    return comparison, outcome


def build_plan(request: PlanRequest, frame: pd.DataFrame, is_synthetic: bool) -> Plan:
    """All options for one request. Raises ValueError for too little history."""
    closes = frame["Close"].astype(float)
    risk = horizon_risk(closes, request.horizon_days)
    amount, max_loss = request.amount, request.max_loss

    invest_a = round(size_for_loss(max_loss, amount, risk.bad_stretch_loss_pct), 2)
    option_a = PlanOption(
        "A", "Size the position to your loss limit", invest_a, round(amount - invest_a, 2),
        DollarRange.from_pct(invest_a, risk.p10_pct, risk.median_pct, risk.p90_pct, risk.worst_pct),
    )

    stop_pct = min(request.max_loss_pct, 95.0)
    if stop_pct >= 95.0:
        option_b = PlanOption(
            "B", "Put in the full amount, with no exit rule", amount, 0.0,
            DollarRange.from_pct(amount, risk.p10_pct, risk.median_pct, risk.p90_pct, risk.worst_pct),
        )
    else:
        stop = stop_outcomes(closes, request.horizon_days, stop_pct)
        option_b = PlanOption(
            "B", "Put in the full amount, with an exit rule", amount, 0.0,
            DollarRange.from_pct(amount, stop.p10_pct, stop.median_pct, stop.p90_pct, stop.worst_pct),
            stop_pct=stop_pct, stop_price=round(risk.current_price * (1 - stop_pct / 100.0), 2),
            stop=stop,
        )

    staged, staged_outcome = _compare_staged(frame, request.horizon_days, invest_a)
    option_c = PlanOption(
        "C", "Spread your entry out over several weeks", invest_a, round(amount - invest_a, 2),
        staged_outcome, staged=staged,
    )
    return Plan(request, risk, (option_a, option_b, option_c), len(closes), is_synthetic)
