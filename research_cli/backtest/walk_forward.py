"""Walk-forward testing: choose parameters on the past, measure on the future.

A backtest that picks its parameters by trying every combination on the whole
history and reporting the best one is not a test. It is a search, and the
number it produces is the best of N random draws -- which is high whether or
not the rule works. Walk-forward is the standard remedy: split the history
into consecutive folds, fit on each fold's training window, and measure on
the untouched window that follows.

**The leakage guarantee, and how it is enforced.**

Selection and evaluation are given *physically different slices* of the price
frame. `_select_parameters` receives `frame.iloc[train_start:train_end + 1]`
and literally cannot see anything after `train_end` -- the rows are not in the
object it holds. The evaluation frame then starts at `test_start`, and the
only bars before it that the engine touches are warm-up bars, which are in
the past relative to every decision made.

Because "we were careful" is exactly the assurance that failed on the options
side of this repository, `_assert_no_leakage` re-checks the invariant at
runtime on every fold, comparing the last date the selector could have seen
against the first date it is scored on. A fold that violates it raises rather
than reporting a number.

**What walk-forward still cannot fix.** Running several folds and reporting
the best fold is the same error one level up. The headline figure here is the
*stitched* curve across all test windows, never a single fold. And selecting
from a grid on each training window still overfits that window; the honest
reading of a walk-forward result is "this is what the rule would have earned
had it been chosen this way in real time", which is usually much worse than
the in-sample figure and is the point of running it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestResult, run_backtest
from .risk_metrics import PerformanceStats, compute_stats
from .strategies import STRATEGY_TEMPLATES, build_strategy

__all__ = [
    "Fold",
    "FoldResult",
    "WalkForwardConfig",
    "WalkForwardResult",
    "generate_folds",
    "walk_forward",
]


class LeakageError(RuntimeError):
    """Raised when a fold's selection window overlaps its evaluation window."""


@dataclass(frozen=True)
class Fold:
    """One train/test split, in integer positions into the price frame."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end < self.test_start <= self.test_end:
            raise ValueError(
                f"fold {self.index} is malformed: train {self.train_start}-"
                f"{self.train_end}, test {self.test_start}-{self.test_end}"
            )

    @property
    def train_bars(self) -> int:
        return self.train_end - self.train_start + 1

    @property
    def test_bars(self) -> int:
        return self.test_end - self.test_start + 1


@dataclass(frozen=True)
class WalkForwardConfig:
    """How the history is split.

    Attributes
    ----------
    train_bars : bars used to choose parameters. 504 is about two years.
    test_bars : bars each fold is scored on. 126 is about six months.
    anchored : when True every training window starts at bar zero and grows;
        when False it rolls at a fixed length. Rolling adapts to regime
        change and uses less data; anchored uses everything and is slower to
        adapt. Rolling is the default because a fixed-length window makes
        folds comparable to each other.
    """

    train_bars: int = 504
    test_bars: int = 126
    anchored: bool = False

    def __post_init__(self) -> None:
        if self.train_bars < 60:
            raise ValueError(
                f"train_bars must be at least 60 for parameter selection to mean "
                f"anything, got {self.train_bars}"
            )
        if self.test_bars < 20:
            raise ValueError(
                f"test_bars must be at least 20 to produce measurable statistics, got "
                f"{self.test_bars}"
            )


@dataclass
class FoldResult:
    """One fold's chosen parameters and out-of-sample outcome."""

    fold: Fold
    parameters: dict
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    result: BacktestResult

    @property
    def excess_return_pct(self) -> float:
        return self.result.excess_return_pct


@dataclass
class WalkForwardResult:
    """The stitched out-of-sample record across every fold.

    `stats` describes the *combined* curve, which is the only figure that
    should be quoted. Per-fold numbers are kept for inspection, and picking
    the best of them would reintroduce exactly the selection bias this
    procedure exists to remove.
    """

    template: str
    folds: list[FoldResult]
    equity: pd.Series
    benchmark_equity: pd.Series
    stats: PerformanceStats
    benchmark_stats: PerformanceStats
    total_trades: int
    total_costs: float
    notes: list[str] = field(default_factory=list)

    @property
    def excess_return_pct(self) -> float:
        return self.stats.total_return_pct - self.benchmark_stats.total_return_pct

    #: A fold counts as beating the benchmark only by a margin wider than
    #: floating-point noise. Without this, buy-and-hold -- which *is* the
    #: benchmark -- reported winning roughly half its folds.
    _MEANINGFUL_EDGE_PCT = 0.01

    @property
    def folds_beating_benchmark(self) -> int:
        return sum(
            1 for fold in self.folds
            if fold.excess_return_pct > self._MEANINGFUL_EDGE_PCT
        )

    @property
    def parameter_stability(self) -> str:
        """How often selection changed its mind between folds.

        Parameters that change every fold mean the grid search is fitting
        noise, which is worth seeing next to the headline return.
        """
        if len(self.folds) < 2:
            return "not enough folds to judge"
        distinct = len({tuple(sorted(f.parameters.items())) for f in self.folds})
        if distinct == 1:
            return f"stable: the same parameters won all {len(self.folds)} folds"
        return (
            f"unstable: {distinct} different parameter sets won across "
            f"{len(self.folds)} folds, which suggests the search is fitting noise"
        )


def generate_folds(n_bars: int, config: WalkForwardConfig) -> list[Fold]:
    """Lay out consecutive, non-overlapping test windows over the history.

    Test windows never overlap, so the stitched curve is a genuine
    out-of-sample track record rather than the same days counted twice.

    Raises ValueError when the history is too short for even one fold, with
    the shortfall quantified -- "not enough data" is a useless message when
    the fix is to ask for a longer history.
    """
    needed = config.train_bars + config.test_bars
    if n_bars < needed:
        raise ValueError(
            f"need at least {needed} bars ({config.train_bars} to train + "
            f"{config.test_bars} to test) but only {n_bars} are available; "
            f"fetch about {needed - n_bars} more bars, or reduce train_bars/test_bars"
        )

    folds: list[Fold] = []
    test_start = config.train_bars
    while test_start + config.test_bars <= n_bars:
        train_start = 0 if config.anchored else test_start - config.train_bars
        folds.append(
            Fold(
                index=len(folds),
                train_start=train_start,
                train_end=test_start - 1,
                test_start=test_start,
                test_end=test_start + config.test_bars - 1,
            )
        )
        test_start += config.test_bars
    return folds


def _assert_no_leakage(
    selection_frame: pd.DataFrame, evaluation_frame: pd.DataFrame, first_traded: pd.Timestamp
) -> None:
    """Re-check at runtime that selection could not have seen the test period.

    The slicing above already makes this true. It is verified anyway because
    the failure it guards against is silent, produces flattering numbers, and
    has happened before in this repository.
    """
    last_seen = selection_frame.index.max()
    if last_seen >= first_traded:
        raise LeakageError(
            f"parameter selection saw data up to {last_seen.date()}, but the fold is "
            f"scored from {first_traded.date()}. The selection window must end strictly "
            "before the first bar it is judged on."
        )
    overlap = selection_frame.index.intersection(
        evaluation_frame.index[evaluation_frame.index >= first_traded]
    )
    if len(overlap):
        raise LeakageError(
            f"{len(overlap)} bar(s) appear in both the selection and scored windows, "
            f"starting at {overlap.min().date()}"
        )


def _select_parameters(
    selection_frame: pd.DataFrame,
    template: str,
    backtest_config: BacktestConfig,
) -> tuple[dict, str | None]:
    """Choose the grid point with the best Sharpe on the training window.

    Sharpe rather than total return: return alone picks whichever setting
    happened to hold the largest position during the window's best stretch,
    which is a statement about that stretch and not about the rule.

    Returns (parameters, note). Grid points that cannot run on this window --
    a 200-bar average on a 150-bar window -- are skipped rather than scored,
    and the note records it.
    """
    grid = STRATEGY_TEMPLATES[template]["grid"]
    best_params, best_score = grid[0], -np.inf
    skipped = 0

    for params in grid:
        try:
            outcome = run_backtest(
                selection_frame, build_strategy(template, **params), backtest_config
            )
        except ValueError:
            skipped += 1
            continue
        score = outcome.stats.sharpe_ratio
        if score is not None and np.isfinite(score) and score > best_score:
            best_params, best_score = params, score

    note = None
    if skipped:
        note = (
            f"{skipped} of {len(grid)} parameter settings needed more history than the "
            f"{len(selection_frame)}-bar training window allowed and were not considered"
        )
    return dict(best_params), note


def walk_forward(
    frame: pd.DataFrame,
    template: str,
    config: WalkForwardConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    risk_free_rate: float = 0.0,
) -> WalkForwardResult:
    """Run a full walk-forward evaluation of one strategy template.

    Each fold selects parameters on its training window and is scored on the
    window that follows. The scored windows are stitched into one continuous
    equity curve, which is the result that should be quoted.

    Raises
    ------
    KeyError : unknown template.
    ValueError : history too short for a single fold.
    LeakageError : a fold's windows overlap. This should be unreachable.
    """
    if template not in STRATEGY_TEMPLATES:
        raise KeyError(
            f"unknown strategy {template!r}; available: "
            f"{', '.join(sorted(STRATEGY_TEMPLATES))}"
        )

    settings = config or WalkForwardConfig()
    execution = backtest_config or BacktestConfig()
    prices = frame.sort_index()
    folds = generate_folds(len(prices), settings)

    fold_results: list[FoldResult] = []
    notes: list[str] = []
    stitched_returns: list[pd.Series] = []
    stitched_benchmark: list[pd.Series] = []
    total_trades = 0
    total_costs = 0.0

    for fold in folds:
        # Physically separate slices. The selector holds no rows past
        # train_end, so it cannot consult the future even by accident.
        selection_frame = prices.iloc[fold.train_start : fold.train_end + 1]
        parameters, note = _select_parameters(selection_frame, template, execution)
        if note:
            notes.append(f"fold {fold.index}: {note}")

        strategy = build_strategy(template, **parameters)
        warmup = max(int(getattr(strategy, "warmup_bars", 1)), 1)

        # Warm-up bars come from before the test window. They are in the
        # past relative to every decision the fold is scored on, so they are
        # history, not leakage -- and the engine skips them before trading.
        evaluation_start = max(0, fold.test_start - warmup)
        evaluation_frame = prices.iloc[evaluation_start : fold.test_end + 1]
        if len(evaluation_frame) < warmup + 2:
            notes.append(f"fold {fold.index}: too few bars to evaluate; skipped")
            continue

        try:
            outcome = run_backtest(evaluation_frame, strategy, execution, risk_free_rate)
        except ValueError as exc:
            notes.append(f"fold {fold.index}: could not be evaluated ({exc}); skipped")
            continue

        _assert_no_leakage(selection_frame, evaluation_frame, outcome.equity.index[0])

        fold_results.append(
            FoldResult(
                fold=fold,
                parameters=parameters,
                train_start_date=prices.index[fold.train_start],
                train_end_date=prices.index[fold.train_end],
                test_start_date=outcome.equity.index[0],
                test_end_date=outcome.equity.index[-1],
                result=outcome,
            )
        )
        # Stitch by return, not by level: each fold restarts at the same
        # notional, so concatenating raw equity would create false jumps.
        stitched_returns.append(outcome.equity.pct_change().dropna())
        stitched_benchmark.append(outcome.benchmark_equity.pct_change().dropna())
        total_trades += outcome.trades
        total_costs += outcome.total_costs

    if not fold_results:
        raise ValueError(
            "no fold could be evaluated; the history is too short or the strategy needs "
            "more warm-up than the test windows allow"
        )

    initial = execution.initial_cash
    combined = pd.concat(stitched_returns).sort_index()
    benchmark_combined = pd.concat(stitched_benchmark).sort_index()
    equity = (1.0 + combined).cumprod() * initial
    benchmark_equity = (1.0 + benchmark_combined).cumprod() * initial

    return WalkForwardResult(
        template=template,
        folds=fold_results,
        equity=equity,
        benchmark_equity=benchmark_equity,
        stats=compute_stats(equity, n_trades=total_trades, risk_free_rate=risk_free_rate),
        benchmark_stats=compute_stats(
            benchmark_equity, n_trades=len(fold_results), risk_free_rate=risk_free_rate
        ),
        total_trades=total_trades,
        total_costs=total_costs,
        notes=notes,
    )
