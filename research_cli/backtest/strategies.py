"""Three built-in strategy templates, deliberately simple and honest.

These exist so the backtest harness can be exercised and so a beginner can
see what testing a rule actually involves. They are textbook rules, not
edges: moving-average crossovers and RSI thresholds have been public for
decades and any advantage they had was arbitraged away long ago. Presenting
them as strategies to trade would be the same mistake this whole tool is
built to avoid.

What they *are* good for is showing a beginner the shape of the problem --
that a rule which looks obviously profitable on a chart usually loses to
buying and holding once costs and execution timing are applied, and that
the gap between those two facts is where most retail money goes.

Every strategy declares `warmup_bars`, the history it needs before its
output means anything, and the engine skips exactly that many bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import SealedWindow

__all__ = [
    "STRATEGY_TEMPLATES",
    "BuyAndHold",
    "MovingAverageCrossover",
    "RSIThreshold",
    "build_strategy",
]


@dataclass
class BuyAndHold:
    """Hold the asset, always. The benchmark every other rule must beat.

    Included as a selectable strategy, not only as an implicit comparison,
    because seeing it win is the most useful result a beginner can get from
    this module.
    """

    name: str = "Buy and hold"
    warmup_bars: int = 1

    def signal(self, window: SealedWindow) -> float:
        return 1.0


@dataclass
class MovingAverageCrossover:
    """Long while the fast average is above the slow one, flat otherwise.

    The canonical trend rule. Its weakness is structural rather than a
    matter of tuning: in a range-bound market the averages cross repeatedly,
    each crossing costs the spread, and the rule bleeds. It tends to look
    best precisely on the samples where the asset trended anyway -- which is
    where buy-and-hold also did well, and without the costs.
    """

    fast: int = 20
    slow: int = 100
    name: str = "Moving-average crossover"

    def __post_init__(self) -> None:
        if self.fast < 2 or self.slow < 2:
            raise ValueError(f"windows must be at least 2 bars, got {self.fast}/{self.slow}")
        if self.fast >= self.slow:
            raise ValueError(
                f"fast window ({self.fast}) must be shorter than slow ({self.slow}); "
                "otherwise the rule is inverted and means the opposite of its name"
            )
        self.name = f"MA crossover ({self.fast}/{self.slow})"

    @property
    def warmup_bars(self) -> int:
        return self.slow + 1

    def signal(self, window: SealedWindow) -> float:
        closes = window.close
        if len(closes) < self.slow:
            return 0.0
        fast_mean = float(closes.iloc[-self.fast :].mean())
        slow_mean = float(closes.iloc[-self.slow :].mean())
        return 1.0 if fast_mean > slow_mean else 0.0


@dataclass
class RSIThreshold:
    """Buy when the relative strength index is low, sell when it is high.

    RSI measures how much of recent movement has been upward, on a 0-100
    scale. The rule buys weakness and sells strength, which is the opposite
    bet to the crossover above -- useful here precisely because the two
    disagree, so a beginner can see that two plausible rules on the same data
    give opposite answers.

    Wilder's smoothing is used rather than a simple mean, matching the
    original definition; a simple mean gives noticeably different values and
    is a common source of "my RSI doesn't match my broker's".
    """

    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    name: str = "RSI threshold"

    def __post_init__(self) -> None:
        if self.period < 2:
            raise ValueError(f"period must be at least 2, got {self.period}")
        if not 0 < self.oversold < self.overbought < 100:
            raise ValueError(
                f"need 0 < oversold < overbought < 100, got "
                f"{self.oversold}/{self.overbought}"
            )
        self.name = f"RSI ({self.period}, {self.oversold:g}/{self.overbought:g})"

    @property
    def warmup_bars(self) -> int:
        return self.period * 3

    def _rsi(self, closes: np.ndarray) -> float:
        deltas = np.diff(closes)
        if len(deltas) < self.period:
            return float("nan")
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Wilder's smoothing: seed with a simple mean, then decay.
        average_gain = float(gains[: self.period].mean())
        average_loss = float(losses[: self.period].mean())
        for position in range(self.period, len(deltas)):
            average_gain = (average_gain * (self.period - 1) + gains[position]) / self.period
            average_loss = (average_loss * (self.period - 1) + losses[position]) / self.period

        if average_loss <= 1e-12:
            return 100.0 if average_gain > 0 else 50.0
        return float(100.0 - 100.0 / (1.0 + average_gain / average_loss))

    def signal(self, window: SealedWindow) -> float:
        rsi = self._rsi(window.close.to_numpy(dtype=float))
        if not np.isfinite(rsi):
            return 0.0
        if rsi <= self.oversold:
            return 1.0
        if rsi >= self.overbought:
            return 0.0
        # Between the thresholds the rule has no opinion; holding the
        # previous position would make the strategy stateful and therefore
        # dependent on where the backtest happened to start.
        return 0.5


#: Selectable templates, with the parameter grids walk-forward searches.
STRATEGY_TEMPLATES: dict[str, dict] = {
    "buy-and-hold": {
        "factory": lambda **_: BuyAndHold(),
        "grid": [{}],
        "description": "Hold the asset for the whole period. The benchmark.",
    },
    "ma-crossover": {
        "factory": lambda **kwargs: MovingAverageCrossover(**kwargs),
        "grid": [
            {"fast": fast, "slow": slow}
            for fast, slow in ((10, 50), (20, 100), (50, 200), (5, 20))
        ],
        "description": "Long while a short average is above a long one.",
    },
    "rsi": {
        "factory": lambda **kwargs: RSIThreshold(**kwargs),
        "grid": [
            {"period": period, "oversold": low, "overbought": high}
            for period, low, high in ((14, 30, 70), (14, 20, 80), (7, 30, 70), (21, 35, 65))
        ],
        "description": "Buy when recent movement has been mostly downward.",
    },
}


def _merge_quant_templates() -> None:
    """Fold the quant strategies into the registry.

    Imported lazily and merged here rather than defined alongside the simple
    rules, so that `strategies.py` has no dependency on the quant package --
    a failure to import `arch` or `scipy` should cost the quant rules, not
    the moving-average one.
    """
    try:
        from .quant_strategies import QUANT_TEMPLATES
    except ImportError:  # pragma: no cover - only when scipy/pandas are absent
        return
    STRATEGY_TEMPLATES.update(QUANT_TEMPLATES)


_merge_quant_templates()


def build_strategy(template: str, **params):
    """Construct a strategy by template name.

    Raises KeyError naming the available templates, since a typo here is the
    most likely CLI mistake.
    """
    if template not in STRATEGY_TEMPLATES:
        raise KeyError(
            f"unknown strategy {template!r}; available: "
            f"{', '.join(sorted(STRATEGY_TEMPLATES))}"
        )
    return STRATEGY_TEMPLATES[template]["factory"](**params)
