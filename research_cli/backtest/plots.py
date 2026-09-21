"""Equity-curve and drawdown chart for a backtest result.

One figure, two stacked panels sharing an x-axis: portfolio value on top,
depth below the running peak underneath. They are stacked rather than
overlaid because they have different units -- dollars and percent -- and a
second y-axis on one chart is the fastest way to mislead somebody about
which line they are reading.

The drawdown panel is not decoration. A strategy's return is what gets
quoted and its drawdown is what decides whether anyone actually holds it to
collect that return, so the two are shown at the same scale, at the same
time, on the same dates.

Colours come from the repository's data-visualization palette: blue for the
strategy, orange for the benchmark -- adjacent slots in that palette's
validated ordering, so they stay distinguishable under colour-vision
deficiency -- and a red fill for drawdown. Every series is also labelled in
the legend, so colour never carries meaning alone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .risk_metrics import drawdown_series

__all__ = ["plot_backtest"]

COLOR_STRATEGY = "#2a78d6"
COLOR_BENCHMARK = "#eb6834"
COLOR_DRAWDOWN = "#e34948"
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"


def plot_backtest(
    equity: pd.Series,
    benchmark_equity: pd.Series,
    path: Path,
    title: str = "Backtest",
    subtitle: str = "",
) -> Path | None:
    """Write the equity and drawdown chart to `path`.

    Returns the path, or None if matplotlib is unavailable -- charting is a
    presentation dependency and its absence must not fail a backtest whose
    numbers are already computed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display needed; write straight to file
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(12.0, 7.0), sharex=True, height_ratios=[2.2, 1.0]
    )
    figure.patch.set_facecolor(COLOR_SURFACE)

    for axes in (top, bottom):
        axes.set_facecolor(COLOR_SURFACE)
        axes.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(COLOR_GRID)
        axes.tick_params(colors=COLOR_MUTED, labelsize=9)

    dates = equity.index.to_pydatetime()
    top.plot(dates, equity.to_numpy(), color=COLOR_STRATEGY, linewidth=2.0, label="Strategy")
    top.plot(
        benchmark_equity.index.to_pydatetime(),
        benchmark_equity.to_numpy(),
        color=COLOR_BENCHMARK,
        linewidth=2.0,
        linestyle="--",
        label="Buy and hold",
    )
    top.set_ylabel("Portfolio value", color=COLOR_INK)
    top.set_title(title, color=COLOR_INK, fontsize=13, loc="left", pad=30)
    if subtitle:
        top.text(
            0.0, 1.012, subtitle, transform=top.transAxes,
            color=COLOR_MUTED, fontsize=9.5, va="bottom",
        )
    legend = top.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR_INK)

    drawdown = drawdown_series(equity)
    bottom.fill_between(
        drawdown.index.to_pydatetime(), drawdown.to_numpy(), 0.0,
        color=COLOR_DRAWDOWN, alpha=0.30, linewidth=0,
    )
    bottom.plot(
        drawdown.index.to_pydatetime(), drawdown.to_numpy(),
        color=COLOR_DRAWDOWN, linewidth=1.4, label="Fall from peak",
    )
    bottom.set_ylabel("Below peak (%)", color=COLOR_INK)
    bottom.set_ylim(min(float(drawdown.min()) * 1.15, -1.0), 1.0)
    bottom_legend = bottom.legend(loc="lower left", frameon=False, fontsize=9)
    for text in bottom_legend.get_texts():
        text.set_color(COLOR_INK)

    locator = mdates.AutoDateLocator()
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(figure)
    return path
