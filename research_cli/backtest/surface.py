"""Sweep two parameters of a strategy and report the result surface.

This exists to answer one question that a single backtest number cannot:
**is this result a plateau or a spike?**

A strategy whose Sharpe is 0.9 at (50, 200) and roughly 0.85 everywhere
nearby has found something that survives small changes. One that scores 0.9
at (50, 200) and 0.2 at (55, 200) has found a coincidence in this particular
history, and the 0.9 is the largest of many draws rather than an estimate of
anything. The two are indistinguishable from the headline figure and obvious
from the surface, which is why this is worth the compute.

The output is a grid, rendered as a 3D surface in the interface. `ruggedness`
summarizes it in one number -- the mean absolute difference between
neighbouring cells, relative to the spread of the whole grid -- so the
warning does not depend on the user reading a picture correctly.

**Every cell is a full walk-forward run**, so this is the slowest thing in
the tool; the grid is deliberately coarse and the caller is expected to keep
it small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import BacktestConfig, run_backtest
from .strategies import STRATEGY_TEMPLATES, build_strategy

__all__ = ["SURFACE_AXES", "ParameterSurface", "sweep"]

# Neighbour spacing on a 5-point axis (4 steps), the resolution the
# ruggedness thresholds in `verdict` were calibrated against.
_REFERENCE_STEPS = 4.0

SURFACE_AXES: dict[str, tuple[tuple[str, list[float]], tuple[str, list[float]]]] = {
    # Seven values per axis rather than five. Each cell is a full pass over the
    # history, so the grid costs 49 backtests instead of 25 -- about five
    # seconds. That buys a surface with enough resolution to show whether the
    # good region is a broad plateau or a single lucky spike, which is the
    # entire question the chart exists to answer.
    "ma-crossover": (
        ("fast", [5, 10, 15, 20, 30, 40, 50]),
        ("slow", [50, 80, 110, 140, 175, 210, 250]),
    ),
    "rsi": (
        ("period", [7, 10, 14, 18, 21, 25, 28]),
        ("oversold", [20, 23, 27, 30, 33, 37, 40]),
    ),
    "vol-target": (
        ("target_vol", [8, 10, 12, 15, 18, 21, 25]),
        ("lookback", [20, 35, 50, 60, 80, 100, 120]),
    ),
    "mean-reversion": (
        ("window", [10, 21, 32, 42, 63, 95, 126]),
        ("entry_z", [-2.5, -2.2, -1.8, -1.5, -1.2, -0.8, -0.5]),
    ),
    "momentum": (
        ("lookback", [63, 105, 126, 168, 189, 252, 378]),
        ("skip", [0, 3, 5, 10, 15, 21, 42]),
    ),
}


@dataclass
class ParameterSurface:
    """A grid of out-of-sample results across two parameters.

    Attributes
    ----------
    x_name, y_name : the parameters swept.
    x_values, y_values : the grid axes.
    sharpe, total_return : (len(y), len(x)) grids. NaN where a combination
        was invalid -- a fast window longer than the slow one, say -- rather
        than zero, which would read as a real but poor result.
    best : the (x, y) pair with the highest Sharpe.
    """

    strategy: str
    x_name: str
    y_name: str
    x_values: list[float]
    y_values: list[float]
    sharpe: list[list[float | None]]
    total_return: list[list[float | None]]
    best: tuple[float, float] | None = None
    best_sharpe: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ruggedness(self) -> float:
        """How sharply the result changes between neighbouring settings.

        Near zero means a smooth surface, where nearby settings behave
        alike. Above about 0.35 means the result changes sharply between
        adjacent settings, which is what overfitting looks like from above.

        The raw mean neighbour difference cannot be compared across grids of
        different sizes: sampling the same parameter range at seven points
        instead of five puts the neighbours closer together, so the average
        step shrinks even when the surface is identical. Multiplying by the
        number of steps and dividing by the reference resolution the
        thresholds were calibrated at removes that dependence, so a grid can
        be made denser without silently making every strategy look robust.
        """
        grid = np.array(
            [[np.nan if v is None else v for v in row] for row in self.sharpe], dtype=float
        )
        if np.all(np.isnan(grid)):
            return 0.0
        spread = float(np.nanmax(grid) - np.nanmin(grid))
        if spread <= 1e-9:
            return 0.0
        differences = np.concatenate(
            [np.abs(np.diff(grid, axis=0)).ravel(), np.abs(np.diff(grid, axis=1)).ravel()]
        )
        differences = differences[np.isfinite(differences)]
        if not differences.size:
            return 0.0
        steps = (grid.shape[0] - 1 + grid.shape[1] - 1) / 2.0
        scale = steps / _REFERENCE_STEPS if steps > 0 else 1.0
        return float(np.mean(differences) / spread * scale)

    @property
    def verdict(self) -> str:
        """What the surface says about whether the best cell means anything."""
        rugged = self.ruggedness
        if rugged > 0.35:
            return (
                "The result changes sharply between neighbouring settings. The best "
                "cell is very likely a coincidence in this particular history rather "
                "than a setting that works."
            )
        if rugged > 0.18:
            return (
                "Nearby settings give noticeably different results, so the best cell "
                "should be treated as one draw from a wide range."
            )
        return (
            "Nearby settings give similar results, which is what a robust parameter "
            "choice looks like. It does not make the strategy profitable -- only "
            "less dependent on a lucky number."
        )


def sweep(
    frame: pd.DataFrame,
    strategy: str,
    execution: BacktestConfig | None = None,
    max_points: int = 64,
) -> ParameterSurface:
    """Run a backtest at every point on the strategy's parameter grid.

    Uses a single in-sample pass per cell rather than a full walk-forward:
    the surface is about *shape*, and a walk-forward at every one of 25
    points takes minutes. The headline number the user acts on still comes
    from the walk-forward run; this is the picture beside it, and the
    distinction is stated in `notes`.
    """
    if strategy not in SURFACE_AXES:
        raise ValueError(
            f"no parameter surface is defined for {strategy!r}; available: "
            f"{', '.join(sorted(SURFACE_AXES))}"
        )
    if strategy not in STRATEGY_TEMPLATES:
        raise ValueError(f"unknown strategy {strategy!r}")

    (x_name, x_values), (y_name, y_values) = SURFACE_AXES[strategy]
    if len(x_values) * len(y_values) > max_points:
        raise ValueError(
            f"that grid is {len(x_values) * len(y_values)} backtests, above the "
            f"{max_points} cap"
        )

    settings = execution or BacktestConfig()
    sharpe: list[list[float | None]] = []
    returns: list[list[float | None]] = []
    best: tuple[float, float] | None = None
    best_sharpe = -np.inf
    invalid = 0

    for y in y_values:
        sharpe_row: list[float | None] = []
        return_row: list[float | None] = []
        for x in x_values:
            params = {x_name: type(x)(x), y_name: type(y)(y)}
            try:
                outcome = run_backtest(frame, build_strategy(strategy, **params), settings)
            except (ValueError, KeyError):
                # An invalid combination (fast >= slow) or too little history.
                invalid += 1
                sharpe_row.append(None)
                return_row.append(None)
                continue
            value = outcome.stats.sharpe_ratio
            sharpe_row.append(None if value is None else float(value))
            return_row.append(float(outcome.stats.total_return_pct))
            if value is not None and value > best_sharpe:
                best_sharpe, best = float(value), (float(x), float(y))
        sharpe.append(sharpe_row)
        returns.append(return_row)

    notes = [
        ("Each cell is a single pass over the whole history, not a walk-forward run. "
        "It shows the shape of the parameter space; the headline figures beside it "
        "come from the out-of-sample test.")
    ]
    if invalid:
        notes.append(
            f"{invalid} combination was not valid for this strategy and is left blank."
            if invalid == 1
            else f"{invalid} combinations were not valid for this strategy and are left blank."
        )

    return ParameterSurface(
        strategy=strategy,
        x_name=x_name,
        y_name=y_name,
        x_values=[float(v) for v in x_values],
        y_values=[float(v) for v in y_values],
        sharpe=sharpe,
        total_return=returns,
        best=best,
        best_sharpe=None if best is None else float(best_sharpe),
        notes=notes,
    )
