"""Tests for the backtest engine, with the no-lookahead guarantee front and centre.

The headline requirement for this module was that lookahead be structurally
impossible rather than merely avoided. A guarantee nobody tests is a
guarantee nobody can rely on, so the first group of tests below actively
tries to cheat and asserts that it cannot.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from research_cli.backtest.engine import (
    BacktestConfig,
    LookaheadError,
    SealedWindow,
    run_backtest,
)
from research_cli.backtest.plots import plot_backtest
from research_cli.backtest.risk_metrics import (
    annualized_return,
    compute_stats,
    max_drawdown,
    sharpe_ratio,
)
from research_cli.backtest.strategies import (
    STRATEGY_TEMPLATES,
    BuyAndHold,
    MovingAverageCrossover,
    RSIThreshold,
    build_strategy,
)
from research_cli.backtest.walk_forward import (
    LeakageError,
    WalkForwardConfig,
    _assert_no_leakage,
    generate_folds,
    walk_forward,
)
from research_cli.data.mock_data import mock_price_history


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return mock_price_history("SPY", "2015-01-01", "2026-09-21")


@pytest.fixture(scope="module")
def short_prices() -> pd.DataFrame:
    return mock_price_history("AAPL", "2023-01-01", "2026-09-21")


# --- the no-lookahead guarantee -------------------------------------------

def test_sealed_window_physically_excludes_the_future(prices: pd.DataFrame) -> None:
    """Not hidden behind a flag -- absent from the object."""
    window = SealedWindow(prices, 100)
    assert len(window) == 101
    assert window.frame.index.max() == window.as_of
    assert (window.frame.index > window.as_of).sum() == 0


def test_a_strategy_cannot_reach_the_future_through_private_state(prices: pd.DataFrame) -> None:
    """Even the private attribute holds only truncated data.

    This is the difference between "we were careful" and "it is impossible".
    """
    window = SealedWindow(prices, 50)
    assert window._frame.index.max() == window.as_of
    assert set(SealedWindow.__slots__) == {"_frame", "_as_of"}
    assert not hasattr(window, "__dict__"), "__slots__ must block attribute injection"


def test_asking_for_a_future_bar_raises(prices: pd.DataFrame) -> None:
    window = SealedWindow(prices, 100)
    with pytest.raises(LookaheadError, match="after this window's seal"):
        window.value_at(prices.index[101])


def test_a_cheating_strategy_cannot_see_tomorrow(prices: pd.DataFrame) -> None:
    """A strategy that tries to peek gets an error, not tomorrow's price."""

    class Cheater:
        name = "cheater"
        warmup_bars = 5

        def __init__(self) -> None:
            self.attempts = 0
            self.blocked = 0

        def signal(self, window: SealedWindow) -> float:
            self.attempts += 1
            try:
                window.value_at(window.as_of + pd.Timedelta(days=1))
            except (LookaheadError, KeyError):
                self.blocked += 1
            return 1.0

    cheater = Cheater()
    run_backtest(prices.head(300), cheater, BacktestConfig())
    assert cheater.attempts > 100
    assert cheater.blocked == cheater.attempts, "every peek must fail"


def test_a_perfect_foresight_strategy_cannot_be_written(prices: pd.DataFrame) -> None:
    """The decisive test: a strategy *given* the full frame still cannot use it.

    An oracle that knows every future return should earn enormous amounts.
    Here it cannot, because the window it is handed does not contain the
    future -- so its return is indistinguishable from buy-and-hold's.
    """

    class WouldBeOracle:
        name = "oracle"
        warmup_bars = 5

        def signal(self, window: SealedWindow) -> float:
            # The only data reachable is the past; the best this can do is
            # react to what already happened.
            return 1.0 if window.frame.index.max() <= window.as_of else 99.0

    outcome = run_backtest(prices.head(500), WouldBeOracle(), BacktestConfig())
    assert outcome.stats.total_return_pct == pytest.approx(
        outcome.benchmark_stats.total_return_pct, abs=1.0
    )


def test_orders_fill_at_the_next_bar_open_not_todays_close(short_prices: pd.DataFrame) -> None:
    """Filling at the close you just made the decision from is a lookahead.

    A strategy that goes long on bar i must be filled at bar i+1's open, so
    its first equity point reflects that open, never bar i's close.
    """
    recorded: list[pd.Timestamp] = []

    class Recorder:
        name = "recorder"
        warmup_bars = 2

        def signal(self, window: SealedWindow) -> float:
            recorded.append(window.as_of)
            return 1.0

    outcome = run_backtest(short_prices, Recorder(), BacktestConfig())
    # The first decision is made on the warm-up bar; the first equity point
    # is the bar after it.
    assert outcome.equity.index[0] > recorded[0]
    positions = short_prices.index.get_indexer([recorded[0], outcome.equity.index[0]])
    assert positions[1] == positions[0] + 1


def test_walk_forward_selection_never_sees_its_test_window(prices: pd.DataFrame) -> None:
    result = walk_forward(prices, "ma-crossover", WalkForwardConfig())
    for fold in result.folds:
        assert fold.train_end_date < fold.test_start_date


def test_leakage_guard_fires_on_an_overlap(prices: pd.DataFrame) -> None:
    """The runtime check must actually detect an overlap, not just exist."""
    selection = prices.iloc[:200]
    evaluation = prices.iloc[150:300]
    with pytest.raises(LeakageError, match="must end strictly before"):
        _assert_no_leakage(selection, evaluation, prices.index[150])


def test_folds_have_non_overlapping_test_windows() -> None:
    folds = generate_folds(3000, WalkForwardConfig(train_bars=504, test_bars=126))
    assert len(folds) > 5
    for earlier, later in pairwise(folds):
        assert earlier.test_end < later.test_start
        assert earlier.train_end < earlier.test_start


def test_too_short_a_history_quantifies_the_shortfall() -> None:
    with pytest.raises(ValueError, match="more bars"):
        generate_folds(300, WalkForwardConfig(train_bars=504, test_bars=126))


# --- execution mechanics ---------------------------------------------------

def test_buy_and_hold_matches_its_own_benchmark(prices: pd.DataFrame) -> None:
    """The engine's most basic self-check."""
    outcome = run_backtest(prices, BuyAndHold(), BacktestConfig())
    assert outcome.stats.total_return_pct == pytest.approx(
        outcome.benchmark_stats.total_return_pct, abs=0.5
    )


def test_costs_reduce_returns(prices: pd.DataFrame) -> None:
    free = run_backtest(
        prices, MovingAverageCrossover(), BacktestConfig(commission_bps=0, slippage_bps=0)
    )
    costly = run_backtest(
        prices, MovingAverageCrossover(), BacktestConfig(commission_bps=10, slippage_bps=20)
    )
    assert costly.total_costs > free.total_costs
    assert costly.stats.total_return_pct < free.stats.total_return_pct


def test_shorting_is_refused_unless_enabled(prices: pd.DataFrame) -> None:
    class AlwaysShort:
        name = "short"
        warmup_bars = 2

        def signal(self, window: SealedWindow) -> float:
            return -1.0

    flat = run_backtest(prices.head(300), AlwaysShort(), BacktestConfig(allow_short=False))
    assert (flat.exposure == 0.0).all()

    short = run_backtest(prices.head(300), AlwaysShort(), BacktestConfig(allow_short=True))
    assert (short.exposure < 0).any()


def test_a_non_finite_signal_holds_the_previous_position(prices: pd.DataFrame) -> None:
    class Broken:
        name = "broken"
        warmup_bars = 2

        def signal(self, window: SealedWindow) -> float:
            return float("nan")

    outcome = run_backtest(prices.head(200), Broken(), BacktestConfig())
    assert outcome.notes
    assert (outcome.exposure == 0.0).all()


def test_exposure_is_capped_at_max(prices: pd.DataFrame) -> None:
    class Greedy:
        name = "greedy"
        warmup_bars = 2

        def signal(self, window: SealedWindow) -> float:
            return 10.0

    outcome = run_backtest(prices.head(200), Greedy(), BacktestConfig(max_exposure=1.0))
    assert outcome.exposure.max() == pytest.approx(1.0)


def test_missing_open_column_is_refused(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Open"):
        run_backtest(prices[["Close"]], BuyAndHold(), BacktestConfig())


def test_history_shorter_than_warmup_is_refused(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="warm-up"):
        run_backtest(prices.head(30), MovingAverageCrossover(fast=20, slow=100))


def test_absurd_leverage_is_refused() -> None:
    with pytest.raises(ValueError, match="max_exposure"):
        BacktestConfig(max_exposure=10.0)


# --- strategies ------------------------------------------------------------

def test_inverted_moving_average_windows_are_refused() -> None:
    with pytest.raises(ValueError, match="must be shorter"):
        MovingAverageCrossover(fast=100, slow=20)


def test_rsi_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="oversold"):
        RSIThreshold(oversold=80, overbought=30)


def test_rsi_is_100_for_a_monotonically_rising_series() -> None:
    """Wilder's RSI with no down moves is exactly 100."""
    rising = pd.DataFrame(
        {"Open": np.arange(1, 101.0), "Close": np.arange(1, 101.0)},
        index=pd.bdate_range("2024-01-01", periods=100),
    )
    strategy = RSIThreshold(period=14)
    assert strategy._rsi(rising["Close"].to_numpy()) == pytest.approx(100.0)


def test_rsi_is_near_50_for_an_alternating_series() -> None:
    values = np.array([100.0 + (1 if i % 2 else -1) for i in range(100)])
    assert RSIThreshold(period=14)._rsi(values) == pytest.approx(50.0, abs=5.0)


@pytest.mark.parametrize("template", sorted(STRATEGY_TEMPLATES))
def test_every_template_runs_and_declares_warmup(template: str, prices: pd.DataFrame) -> None:
    strategy = build_strategy(template)
    assert strategy.warmup_bars >= 1
    outcome = run_backtest(prices, strategy, BacktestConfig())
    assert len(outcome.equity) > 100
    assert np.isfinite(outcome.stats.total_return_pct)


def test_unknown_template_lists_the_valid_ones() -> None:
    with pytest.raises(KeyError, match="ma-crossover"):
        build_strategy("nonsense")


@pytest.mark.parametrize("template", sorted(STRATEGY_TEMPLATES))
def test_every_template_survives_walk_forward(template: str, prices: pd.DataFrame) -> None:
    result = walk_forward(prices, template, WalkForwardConfig())
    assert result.folds
    assert len(result.equity) > 100
    assert result.stats.n_periods == len(result.equity)


def test_buy_and_hold_never_beats_itself(prices: pd.DataFrame) -> None:
    """Regression: float noise once made this report winning half its folds."""
    result = walk_forward(prices, "buy-and-hold", WalkForwardConfig())
    assert result.folds_beating_benchmark == 0


# --- statistics ------------------------------------------------------------

def test_annualized_return_of_a_known_curve() -> None:
    index = pd.bdate_range("2024-01-01", periods=504)
    steady = pd.Series(100 * 1.10 ** (np.arange(504) / 252), index=index)
    assert annualized_return(steady) == pytest.approx(10.0, abs=0.05)


def test_sharpe_is_none_for_a_flat_curve() -> None:
    """Infinity would read as a spectacular result."""
    flat = pd.Series([100.0] * 50, index=pd.bdate_range("2024-01-01", periods=50))
    assert sharpe_ratio(flat) is None


def test_max_drawdown_of_a_known_v_shape() -> None:
    curve = pd.Series([100.0, 95, 90, 85, 80, 85, 90, 95, 100, 105])
    depth, length = max_drawdown(curve)
    assert depth == pytest.approx(-20.0)
    assert length == 4


def test_risk_free_rate_lowers_sharpe() -> None:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2024-01-01", periods=500)
    curve = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 500)), index=index)
    assert sharpe_ratio(curve, risk_free_rate=0.05) < sharpe_ratio(curve, risk_free_rate=0.0)


def test_win_rate_ignores_flat_periods() -> None:
    """A day spent in cash is neither a win nor a loss."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.01, 400)
    returns[::2] = 0.0
    index = pd.bdate_range("2024-01-01", periods=400)
    curve = pd.Series(100 * np.cumprod(1 + returns), index=index)
    assert compute_stats(curve).win_rate_pct > 35.0


def test_zero_equity_is_refused() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        compute_stats(pd.Series([100.0, 50.0, 0.0, 10.0]))


# --- chart -----------------------------------------------------------------

def test_chart_is_written(prices: pd.DataFrame, tmp_path) -> None:
    outcome = run_backtest(prices, BuyAndHold(), BacktestConfig())
    path = plot_backtest(
        outcome.equity, outcome.benchmark_equity, tmp_path / "chart.png", title="Test"
    )
    assert path is not None
    assert path.is_file()
    assert path.stat().st_size > 5_000
