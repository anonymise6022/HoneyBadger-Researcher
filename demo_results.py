#!/usr/bin/env python3
"""Run the whole pipeline on mock data and show what it produced.

    python demo_results.py

No API keys, no network, no configuration. Every module runs against the
synthetic generators in `quant_pipeline/mock_data`, and the run produces
three artifacts:

    console       a daily table and a summary block
    results/pipeline_output.csv    one row per day, every intermediate
    results/conviction_over_time.png   conviction, regime bands, trades

**What this does and does not demonstrate.** It shows that the pieces fit
together and that each produces numbers of a sensible magnitude on data
whose true parameters are known -- the Hurst estimate should land near the
0.14 the volatility path was generated with, the implied distribution's
mean should sit near the forward, the regime flags should cluster near the
injected crashes. That is a wiring check and a plausibility check.

It is emphatically **not a backtest**. The simulated world plants a
relationship between a macro feature and the underlying's drift so that the
pipeline has something to find (see `mock_data.world`), at an effect size
far beyond anything real macro data offers. Any apparent profitability here
is circular by construction.

**Failure handling.** Each module is wrapped per day. A module that raises
records the failure, writes NaN for its columns, and the run continues, so
one broken component shows up as a column of NaNs and a named entry in the
summary rather than as a traceback with nothing to look at.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from quant_pipeline.config import PipelineConfig
from quant_pipeline.ensemble import PremiumBaseline, divergence_signal
from quant_pipeline.macro_model import (
    QuantileRegressionForecaster,
    align_to_daily,
    build_macro_features,
    forward_return_target,
    load_macro_panel,
)
from quant_pipeline.macro_model.feature_engineering import MacroFeatureConfig
from quant_pipeline.mock_data import simulate_world
from quant_pipeline.quant_model import market_view_from_chain
from quant_pipeline.regime_detection import RoughVolConfig, TDAConfig
from quant_pipeline.regime_detection.rough_vol import rolling_rough_vol
from quant_pipeline.regime_detection.tda_signal import rolling_tda_signal

T = TypeVar("T")

DEFAULT_DAYS = 90
MACRO_START = "2018-01-01"
MACRO_END = "2024-12-31"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Colors from the repository's data-visualization palette. Long and short
#: markers sit in the 6-8 CVD separation band, which is permitted only with
#: a secondary encoding -- here the marker shape (up/down triangle), the
#: legend labels, and the decision column in the CSV and console table.
COLOR_SURFACE = "#fcfcfb"
COLOR_LINE = "#2a78d6"
COLOR_LONG = "#008300"
COLOR_SHORT = "#e34948"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
#: Regime bands are greyscale on purpose: an ordinal severity ramp with no
#: hue leaves the entire color channel to the marks, and reads correctly in
#: print and under every form of color vision deficiency.
REGIME_BAND = {"stable": None, "elevated": "#e1e0d9", "unstable": "#c3c2b7"}


@dataclass
class FailureLog:
    """Per-day, per-module failures, so a broken module is visible, not fatal."""

    entries: list[tuple[pd.Timestamp, str, str]] = field(default_factory=list)

    def record(self, date: pd.Timestamp, module: str, error: BaseException) -> None:
        self.entries.append((date, module, f"{type(error).__name__}: {error}"))

    def counts(self) -> Counter:
        return Counter(module for _, module, _ in self.entries)

    def first_per_module(self) -> dict[str, tuple[pd.Timestamp, str]]:
        """One representative failure per module -- the first one seen."""
        seen: dict[str, tuple[pd.Timestamp, str]] = {}
        for date, module, message in self.entries:
            seen.setdefault(module, (date, message))
        return seen


def attempt(
    log: FailureLog,
    date: pd.Timestamp,
    module: str,
    action: Callable[..., T],
    *args: Any,
) -> T | None:
    """Run `action(*args)`, returning None and recording a failure if it raises.

    Arguments are passed explicitly rather than captured in a closure, so
    nothing here can depend on a loop variable's value at call time -- the
    classic late-binding bug, and a nasty one in a loop that is meant to
    isolate failures.

    Deliberately catches `Exception` rather than a narrow set: the point is
    to survive an *unanticipated* bug in one module, and a list of expected
    exception types would by definition not include it. KeyboardInterrupt
    and SystemExit are not caught, since those are the user asking to stop.
    """
    try:
        return action(*args)
    except Exception as error:  # noqa: BLE001 - see docstring
        log.record(date, module, error)
        return None


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def build_macro(seed: int, config: PipelineConfig) -> tuple[pd.DataFrame, Any]:
    """Load the macro panel (mock), align it point-in-time, build features."""
    panel = load_macro_panel(start=MACRO_START, end=MACRO_END, source="mock", seed=seed)
    daily = align_to_daily(panel)
    features = build_macro_features(
        daily,
        MacroFeatureConfig(
            zscore_window=config.macro_zscore_window,
            surprise_window=config.macro_surprise_window,
            lags=config.macro_lags,
        ),
        consensus_is_modelled=(panel.source == "fred"),
    )
    return daily, features


def build_world(features: Any, seed: int, config: PipelineConfig) -> Any:
    """Simulate a market whose drift is driven by one macro feature.

    The driver is the curve-slope z-score: a flattening or inverted curve
    tilts the underlying's drift down. The sign is the conventional one and
    the magnitude is not -- see `mock_data.world` on why a planted signal
    makes this a wiring check rather than evidence.
    """
    index = features.frame.index
    driver = features.frame["curve_slope_z"]
    return simulate_world(
        n_days=len(index),
        start=str(index[0].date()),
        drift_driver=driver,
        expiry_days=config.forecast_horizon_days,
        seed=seed,
    )


def precompute_regime(
    world: Any, n_days: int, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll the two regime diagnostics over just enough history.

    Both are windowed estimators, so they are computed over the demo window
    plus the lookback each needs, rather than over the whole simulated
    history. The topological signal is the expensive one -- a boundary-matrix
    reduction per day -- and this keeps it to a few hundred reductions.
    """
    tda_config = TDAConfig(
        window=config.tda_window,
        zscore_window=config.tda_zscore_window,
        elevated_z=config.tda_elevated_z,
        unstable_z=config.tda_unstable_z,
    )
    rough_config = RoughVolConfig(
        window=config.rough_vol_window, forecast_horizon=config.forecast_horizon_days
    )

    tda_span = tda_config.window + tda_config.zscore_window + n_days
    tda = rolling_tda_signal(world.returns.iloc[-tda_span:], tda_config)

    rough_span = rough_config.window + rough_config.baseline_window + n_days
    rough = rolling_rough_vol(world.realized_vol.iloc[-rough_span:], rough_config)
    return tda, rough


# --------------------------------------------------------------------------
# The daily loop
# --------------------------------------------------------------------------

def _fit_forecaster(
    features: Any,
    target: pd.Series,
    position: int,
    config: PipelineConfig,
    max_features: int,
) -> QuantileRegressionForecaster:
    """Fit the macro forecaster on data that was fully realized by `position`.

    Point-in-time discipline lives here. The target at row t spans the
    returns over t+1..t+h, so on date d the last row whose target is known
    is d-h. Slicing to d and relying on NaN handling would leak h days of
    future returns into training, which is the single easiest way to make a
    pipeline like this look brilliant.
    """
    horizon = config.forecast_horizon_days
    train_end = position - horizon
    train_start = max(0, train_end - config.macro_lookback_days + 1)
    return QuantileRegressionForecaster(
        quantile_levels=config.quantile_levels, max_features=max_features
    ).fit(features.frame.iloc[train_start : train_end + 1], target.iloc[train_start : train_end + 1])


def _forecast_for(
    forecaster: QuantileRegressionForecaster, features: Any, date: pd.Timestamp, horizon: int
) -> Any:
    return forecaster.forecast(features.frame.loc[date], horizon)


def _market_view_for(world: Any, date: pd.Timestamp, rate: float) -> Any:
    return market_view_from_chain(
        world.chains[date],
        spot=float(world.prices.loc[date]),
        expiry_years=world.expiry_years,
        rate=rate,
    )


_NAN_MACRO = {
    "macro_mean": np.nan, "macro_var": np.nan, "macro_std": np.nan,
    "macro_skew": np.nan, "macro_n_train": np.nan, "macro_nonzero": np.nan,
}
_NAN_MARKET = {
    "mkt_mean": np.nan, "mkt_var": np.nan, "mkt_std": np.nan, "mkt_skew": np.nan,
    "mkt_atm_iv": np.nan, "mkt_forward_error": np.nan, "mkt_density_mass": np.nan,
}
_NAN_SIGNAL = {
    "gap_mean_raw": np.nan, "gap_var_raw": np.nan, "gap_skew_raw": np.nan,
    "gap_mean_adj": np.nan, "gap_var_adj": np.nan, "gap_skew_adj": np.nan,
    "divergence_score": np.nan, "base_conviction": np.nan, "conviction": np.nan,
    "regime_flag": "stable", "tda_flag": "stable", "rough_flag": "stable",
    "decision": "no-trade", "baseline_ready": False,
    "reason": "upstream module failed; no signal computed",
}


def _macro_columns(forecast: Any) -> dict[str, Any]:
    distribution = forecast.distribution
    return {
        "macro_mean": distribution.mean,
        "macro_var": distribution.variance,
        "macro_std": distribution.std,
        "macro_skew": distribution.skew,
        "macro_n_train": float(forecast.n_train),
        "macro_nonzero": float(forecast.nonzero_coefficients),
    }


def _market_columns(view: Any) -> dict[str, Any]:
    return {
        "mkt_mean": view.mean_return,
        "mkt_var": view.variance_return,
        "mkt_std": view.std_return,
        "mkt_skew": view.skew_return,
        "mkt_atm_iv": view.implied_vol_atm,
        "mkt_forward_error": view.forward_error,
        "mkt_density_mass": view.distribution.total_mass,
    }


def _signal_columns(signal: Any) -> dict[str, Any]:
    return {
        "gap_mean_raw": signal.raw_gaps.mean_gap,
        "gap_var_raw": signal.raw_gaps.variance_log_ratio,
        "gap_skew_raw": signal.raw_gaps.skew_gap,
        "gap_mean_adj": signal.adjusted_gaps.mean_gap,
        "gap_var_adj": signal.adjusted_gaps.variance_log_ratio,
        "gap_skew_adj": signal.adjusted_gaps.skew_gap,
        "divergence_score": signal.divergence_score,
        "base_conviction": signal.base_conviction,
        "conviction": signal.conviction,
        "regime_flag": signal.regime.regime_flag,
        "tda_flag": signal.regime.tda_flag,
        "rough_flag": signal.regime.rough_flag,
        "decision": signal.decision,
        "baseline_ready": signal.baseline_ready,
        "reason": signal.reason,
    }


def _regime_columns(
    tda: pd.DataFrame, rough: pd.DataFrame, date: pd.Timestamp
) -> tuple[dict[str, float], list[str]]:
    """Read the two precomputed diagnostics for one date.

    A date missing from either frame is a gap in that diagnostic, not a
    crash: the columns come back NaN and the module is named as failed, and
    `regime_modulation` will read the NaNs as "no baseline yet, do not
    modulate".
    """
    failed: list[str] = []
    columns: dict[str, float] = {}

    if date in tda.index:
        row = tda.loc[date]
        columns.update(
            tda_l1=float(row["tda_l1"]),
            tda_n_loops=float(row["tda_n_loops"]),
            tda_z=float(row["tda_z"]),
        )
    else:
        failed.append("regime_detection.tda")
        columns.update(tda_l1=np.nan, tda_n_loops=np.nan, tda_z=np.nan)

    if date in rough.index:
        row = rough.loc[date]
        columns.update(
            hurst=float(row["hurst"]),
            hurst_drop=float(row["hurst_drop"]),
            rv_forecast=float(row["rv_forecast"]),
        )
    else:
        failed.append("regime_detection.rough_vol")
        columns.update(hurst=np.nan, hurst_drop=np.nan, rv_forecast=np.nan)

    return columns, failed


def run_pipeline(
    world: Any,
    features: Any,
    tda: pd.DataFrame,
    rough: pd.DataFrame,
    n_days: int,
    config: PipelineConfig,
    refit_every: int,
    max_features: int,
    log: FailureLog,
    burn_in: int = 60,
    rate: float = 0.03,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk forward day by day, recording every intermediate quantity.

    `burn_in` days are processed before the reported window and used only
    to fill the premium baseline. Without them the first twenty reported
    days would be forced to "no-trade" while the baseline accumulates, and
    -- worse -- the days just after would be scored against a baseline built
    from three or four observations. The burn-in runs through exactly the
    same code path; the only difference is that its rows are not recorded.

    Every module call goes through `attempt`, so a failure in one component
    on one day costs that day's column and nothing else.
    """
    horizon = config.forecast_horizon_days
    index = world.index
    total_days = n_days + burn_in
    if total_days > len(index):
        raise ValueError(
            f"need {total_days} days ({n_days} reported + {burn_in} burn-in) but only "
            f"{len(index)} were simulated"
        )
    demo_dates = index[-total_days:]
    target = forward_return_target(world.underlying_returns, horizon)
    baseline = PremiumBaseline(window=60, min_observations=20)

    forecaster: QuantileRegressionForecaster | None = None
    rows: list[dict[str, Any]] = []
    started = time.time()

    for step, date in enumerate(demo_dates):
        recording = step >= burn_in
        position = index.get_loc(date)
        failed: list[str] = []
        row: dict[str, Any] = {
            "date": date,
            "spot": float(world.prices.loc[date]),
            "realized_vol": float(world.realized_vol.loc[date]),
            "implied_atm_vol": float(world.implied_atm_vol.loc[date]),
            "true_regime": str(world.regime.loc[date]),
        }

        # --- macro forecast -------------------------------------------------
        if forecaster is None or step % refit_every == 0:
            fitted = attempt(
                log, date, "macro_model.fit",
                _fit_forecaster, features, target, position, config, max_features,
            )
            if fitted is not None:
                forecaster = fitted

        forecast = None
        if forecaster is not None:
            forecast = attempt(
                log, date, "macro_model.forecast",
                _forecast_for, forecaster, features, date, horizon,
            )
        if forecast is None:
            failed.append("macro_model")
            row.update(_NAN_MACRO)
        else:
            row.update(_macro_columns(forecast))

        # --- market-implied distribution ------------------------------------
        view = attempt(log, date, "quant_model", _market_view_for, world, date, rate)
        if view is None:
            failed.append("quant_model")
            row.update(_NAN_MARKET)
        else:
            row.update(_market_columns(view))

        # --- regime diagnostics (precomputed; read point-in-time) -----------
        regime_columns, regime_failed = _regime_columns(tda, rough, date)
        row.update(regime_columns)
        failed.extend(regime_failed)

        # --- divergence and decision ----------------------------------------
        signal = None
        if forecast is not None and view is not None:
            signal = attempt(
                log, date, "ensemble",
                divergence_signal, forecast, view,
                row["tda_z"], row["hurst"], row["hurst_drop"], baseline, config,
            )
            if signal is None:
                failed.append("ensemble")
        if signal is None:
            row.update(_NAN_SIGNAL)
        else:
            row.update(_signal_columns(signal))

        row["failed_modules"] = ";".join(sorted(set(failed)))
        if recording:
            rows.append(row)

        if verbose and (step + 1) % 20 == 0:
            phase = "burn-in" if not recording else "reported"
            print(
                f"  ... {step + 1}/{len(demo_dates)} days ({phase})  "
                f"({time.time() - started:.1f}s)",
                file=sys.stderr,
            )

    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

_TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("date", "date", "{:%Y-%m-%d}"),
    ("macro_mean", "mac_mu", "{:+.4f}"),
    ("macro_std", "mac_sd", "{:.4f}"),
    ("macro_skew", "mac_sk", "{:+.2f}"),
    ("mkt_mean", "mkt_mu", "{:+.4f}"),
    ("mkt_std", "mkt_sd", "{:.4f}"),
    ("mkt_skew", "mkt_sk", "{:+.2f}"),
    ("divergence_score", "diverg", "{:+.3f}"),
    ("tda_z", "tda_z", "{:+.2f}"),
    ("hurst", "hurst", "{:.3f}"),
    ("regime_flag", "regime", "{:s}"),
    ("conviction", "conv", "{:.3f}"),
    ("decision", "decision", "{:s}"),
)


def _format_cell(value: Any, fmt: str) -> str:
    """Format one cell, rendering NaN and None as a visible dash."""
    if value is None:
        return "--"
    if isinstance(value, float) and not np.isfinite(value):
        return "--"
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def print_table(frame: pd.DataFrame, limit: int | None = None) -> None:
    """Print the daily results as fixed-width text, no dependencies."""
    display = frame if limit is None else frame.tail(limit)
    headers = [header for _, header, _ in _TABLE_COLUMNS]
    rendered: list[list[str]] = []
    for date, row in display.iterrows():
        cells = []
        for column, _, fmt in _TABLE_COLUMNS:
            value = date if column == "date" else row.get(column)
            cells.append(_format_cell(value, fmt))
        rendered.append(cells)

    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rendered), default=0))
        for i in range(len(headers))
    ]
    line = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for cells in rendered:
        print("  ".join(c.rjust(w) for c, w in zip(cells, widths)))
    print("-" * len(line))
    print(
        "mac_* = macro forecast moments over the horizon; mkt_* = option-implied\n"
        "moments over the same horizon; diverg = signed divergence in implied\n"
        "standard deviations; conv = conviction after regime modulation."
    )


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write the full results, every column, one row per day."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True, float_format="%.6f")
    return path


def plot_conviction(frame: pd.DataFrame, path: Path) -> Path | None:
    """Plot conviction over time with regime bands and trade markers.

    One axes, by design. Conviction is the line; the regime flag is a
    greyscale background band, ordered by severity so that darker always
    means worse; trade decisions are triangles at the conviction value,
    pointing the way the trade points. The conviction floor is drawn as a
    dashed rule, so a day sitting below it visibly explains its own
    "no-trade".
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed; write straight to file
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as error:
        print(f"  [skip] matplotlib unavailable ({error}); no plot written", file=sys.stderr)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(13.0, 5.2))
    figure.patch.set_facecolor(COLOR_SURFACE)
    axes.set_facecolor(COLOR_SURFACE)

    dates = frame.index.to_pydatetime()
    flags = frame["regime_flag"].fillna("stable").to_numpy()
    # Shade contiguous runs of one flag as a single span, so the bands read
    # as episodes rather than as 90 adjacent slivers.
    start = 0
    for position in range(1, len(flags) + 1):
        if position == len(flags) or flags[position] != flags[start]:
            color = REGIME_BAND.get(str(flags[start]))
            if color is not None:
                left = dates[start]
                right = dates[min(position, len(dates) - 1)]
                axes.axvspan(left, right, color=color, linewidth=0, zorder=0)
            start = position

    conviction = frame["conviction"].to_numpy(dtype=float)
    axes.plot(dates, conviction, color=COLOR_LINE, linewidth=2.0, zorder=3, label="Conviction")

    floor = float(frame.attrs.get("min_conviction", np.nan))
    if np.isfinite(floor):
        axes.axhline(
            floor, color=COLOR_MUTED, linewidth=1.0, linestyle="--", zorder=2,
            label=f"Conviction floor ({floor:.2f})",
        )

    for decision, color, marker in (("long", COLOR_LONG, "^"), ("short", COLOR_SHORT, "v")):
        mask = (frame["decision"] == decision).to_numpy()
        if mask.any():
            axes.scatter(
                np.asarray(dates)[mask], conviction[mask], s=70, marker=marker,
                color=color, edgecolors=COLOR_SURFACE, linewidths=1.2, zorder=4,
                label=f"{decision.capitalize()} signal",
            )

    axes.set_ylim(0.0, max(1.0, float(np.nanmax(conviction)) * 1.12 if len(conviction) else 1.0))
    axes.set_ylabel("Conviction", color=COLOR_INK)
    axes.set_title(
        "Pipeline conviction over time, with regime bands and trade signals",
        color=COLOR_INK, fontsize=13, pad=12, loc="left",
    )
    axes.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(COLOR_GRID)
    axes.tick_params(colors=COLOR_MUTED, labelsize=9)
    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    handles, labels = axes.get_legend_handles_labels()
    # A hairline edge keeps the swatches legible: the band fills are close
    # in lightness to the surface by design, so an unbordered patch can
    # read as an empty gap in the legend.
    handles += [
        Patch(facecolor=REGIME_BAND["elevated"], edgecolor=COLOR_MUTED,
              linewidth=0.8, label="Regime: elevated"),
        Patch(facecolor=REGIME_BAND["unstable"], edgecolor=COLOR_MUTED,
              linewidth=0.8, label="Regime: unstable"),
    ]
    labels += ["Regime: elevated", "Regime: unstable"]
    legend = axes.legend(
        handles, labels, loc="upper left", frameon=False, fontsize=9, ncol=3
    )
    for text in legend.get_texts():
        text.set_color(COLOR_INK)

    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(figure)
    return path


def print_summary(frame: pd.DataFrame, log: FailureLog, config: PipelineConfig) -> None:
    """Print the final block: signal counts, regime mix, conviction, breakage."""
    total = len(frame)
    decisions = frame["decision"].value_counts()
    trades = frame[frame["decision"].isin(["long", "short"])]

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Days simulated                 {total}")
    print(
        f"Trade signals                  {len(trades)} "
        f"({len(trades) / total:.0%} of days) -- "
        f"{int(decisions.get('long', 0))} long, {int(decisions.get('short', 0))} short"
    )
    print(f"No-trade days                  {int(decisions.get('no-trade', 0))}")

    warming = int((~frame["baseline_ready"].astype(bool)).sum())
    if warming:
        print(
            f"  of which premium baseline still warming up: {warming} "
            "(no decision is possible on these days by construction)"
        )

    print()
    print("Regime flag breakdown (inferred):")
    for flag in ("stable", "elevated", "unstable"):
        subset = frame[frame["regime_flag"] == flag]
        if subset.empty:
            print(f"  {flag:<10} {0:>4} days")
            continue
        flag_trades = int(subset["decision"].isin(["long", "short"]).sum())
        print(
            f"  {flag:<10} {len(subset):>4} days   "
            f"mean conviction {subset['conviction'].mean():.3f}   "
            f"trades {flag_trades}"
        )

    if "true_regime" in frame.columns:
        print()
        print("Ground-truth regime for the same days (the pipeline never sees this):")
        for label, count in frame["true_regime"].value_counts().items():
            print(f"  {label!s:<10} {count:>4} days")

    print()
    print(
        f"Thresholds in force            |divergence| >= "
        f"{config.divergence_trade_threshold:.2f} and conviction >= {config.min_conviction:.2f}"
    )
    print(f"Mean conviction                {frame['conviction'].mean():.3f}")
    print(f"Mean conviction before regime  {frame['base_conviction'].mean():.3f}")
    print(
        f"Divergence score               mean {frame['divergence_score'].mean():+.3f}, "
        f"sd {frame['divergence_score'].std():.3f}, "
        f"|max| {frame['divergence_score'].abs().max():.3f}"
    )
    print(
        f"Hurst exponent                 mean {frame['hurst'].mean():.3f} "
        f"(generated at H = 0.14; measurement noise biases this down and the\n"
        f"                               smooth regime schedule underneath biases it up, so "
        f"the net direction\n                               varies by window)"
    )
    print(
        f"Implied density total mass     mean {frame['mkt_density_mass'].mean():.4f} "
        f"(1.0 would mean no tail truncation)"
    )

    print()
    print("Data quality -- NaN counts by column (non-zero only):")
    nan_counts = frame.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if nan_counts.empty:
        print("  none: every column is populated on every day")
    else:
        for column, count in nan_counts.items():
            print(f"  {column!s:<24} {int(count):>4} / {total} days")

    print()
    print("Module failures:")
    counts = log.counts()
    if not counts:
        print("  none: every module returned a result on every day")
    else:
        representatives = log.first_per_module()
        for module, count in counts.most_common():
            date, message = representatives[module]
            print(f"  {module:<26} {count:>4} day(s); first on {date:%Y-%m-%d}: {message}")

    print()
    print(
        "Reminder: the simulated world plants a macro-to-return relationship so the\n"
        "pipeline has something to find, at an effect size far larger than anything\n"
        "real macro data offers. This run demonstrates that the components fit\n"
        "together and produce sane magnitudes. It is not evidence of edge."
    )
    print("=" * 78)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the quant pipeline end to end on mock data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="days to simulate")
    parser.add_argument("--seed", type=int, default=7, help="seed for every generator")
    parser.add_argument(
        "--refit-every", type=int, default=5,
        help="refit the macro forecaster every N days; 1 refits daily and is slower",
    )
    parser.add_argument(
        "--max-features", type=int, default=40,
        help="cap on macro features entering the quantile regression",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR, help="where the CSV and PNG go"
    )
    parser.add_argument(
        "--burn-in", type=int, default=60,
        help="days run before the reported window to fill the premium baseline",
    )
    parser.add_argument(
        "--table-rows", type=int, default=0,
        help="print only the last N rows of the daily table; 0 prints all",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip the PNG")
    args = parser.parse_args(argv)

    if args.days < 1:
        parser.error(f"--days must be at least 1, got {args.days}")
    if args.refit_every < 1:
        parser.error(f"--refit-every must be at least 1, got {args.refit_every}")
    if args.max_features < 1:
        parser.error(f"--max-features must be at least 1, got {args.max_features}")
    if args.burn_in < 0:
        parser.error(f"--burn-in must be non-negative, got {args.burn_in}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig()
    log = FailureLog()
    started = time.time()

    print("Quant pipeline demo -- mock data, no API keys required")
    print(
        f"  days={args.days}  seed={args.seed}  refit_every={args.refit_every}  "
        f"burn_in={args.burn_in}"
    )
    print()

    print("[1/5] Building macro panel and features ...", file=sys.stderr)
    _, features = build_macro(args.seed, config)
    usable = features.complete()
    print(
        f"      {len(features.feature_names)} features, "
        f"{len(usable)} complete rows of {len(features.frame)}",
        file=sys.stderr,
    )

    print("[2/5] Simulating market, volatility path and option chains ...", file=sys.stderr)
    world = build_world(features, args.seed, config)
    print(
        f"      {len(world.index)} days, price {world.prices.iloc[0]:.1f} -> "
        f"{world.prices.iloc[-1]:.1f}, {len(world.chains)} chains",
        file=sys.stderr,
    )

    if args.days + args.burn_in > len(world.index):
        print(
            f"error: asked for {args.days} days plus {args.burn_in} burn-in, but only "
            f"{len(world.index)} were simulated",
            file=sys.stderr,
        )
        return 2

    print("[3/5] Rolling regime diagnostics (persistent homology is the slow part) ...",
          file=sys.stderr)
    tda, rough = precompute_regime(world, args.days + args.burn_in, config)
    print(
        f"      TDA over {len(tda)} days, rough-vol over {len(rough)} days",
        file=sys.stderr,
    )

    print(
        f"[4/5] Running the pipeline over {args.days} days "
        f"(plus {args.burn_in} burn-in days to fill the premium baseline) ...",
        file=sys.stderr,
    )
    results = run_pipeline(
        world, features, tda, rough, args.days, config,
        args.refit_every, args.max_features, log, burn_in=args.burn_in,
    )
    results.attrs["min_conviction"] = config.min_conviction

    print("[5/5] Writing output ...", file=sys.stderr)
    print()
    print_table(results, args.table_rows or None)

    csv_path = write_csv(results, args.output_dir / "pipeline_output.csv")
    png_path = None if args.no_plot else plot_conviction(
        results, args.output_dir / "conviction_over_time.png"
    )

    print_summary(results, log, config)
    print()
    print(f"CSV   {csv_path}")
    if png_path is not None:
        print(f"PNG   {png_path}")
    print(f"Ran in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
