"""Turn the point-in-time macro panel into conditioning features.

Four transformations, each answering a different question about the same
data:

**Surprise vs. consensus.** A CPI print of 0.4% is not news; a 0.4% print
against a 0.2% consensus is. The level is already in the price, the
surprise is not. Normalizing by the trailing standard deviation of past
surprises puts inflation and payrolls on one scale and, more importantly,
makes the feature comparable across regimes -- a 0.1pp miss means something
different when the recent forecast dispersion is 0.05pp than when it is
0.3pp.

**Rolling z-scores of levels.** Macro levels are persistent and
non-stationary; regressing returns on a raw unemployment rate mostly fits
the slow trend. A z-score against a trailing window asks the stationary
question -- how unusual is this level relative to the recent past -- at the
cost of a window-length choice that is arbitrary and matters.

**Yield-curve slope.** 10y - 2y. Given separately from its two legs
because the slope carries information neither leg does, and because a
linear model given only the legs would have to discover the difference,
spending a degree of freedom on something already known.

**Lags.** Macro variables affect asset prices with delays, and a monthly
series read daily is flat between releases, so the lag structure also
encodes "how long since this last changed".

A caution the feature count makes concrete: five series expand to roughly
forty columns after z-scores, changes, surprises and three lag depths each.
On a few hundred training rows that is more than enough to fit noise, which
is why `forecast_model` regularizes rather than running plain least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "MacroFeatureConfig",
    "MacroFeatures",
    "add_lags",
    "build_macro_features",
    "normalized_surprise",
    "rolling_zscore",
    "yield_curve_slope",
]

_MIN_STD = 1e-10  # below this a window is constant; a z-score is meaningless


@dataclass(frozen=True)
class MacroFeatureConfig:
    """Window lengths for the feature transformations.

    Attributes
    ----------
    zscore_window : trailing business days for level z-scores. 250 is about
        a year, long enough to be stable and short enough to adapt.
    surprise_window : number of *releases* (not days) used for the surprise
        standard deviation. 24 is two years of monthly prints.
    change_windows : horizons, in business days, for level differences.
    lags : lag depths, in business days, applied to every feature.
    min_periods_fraction : fraction of a window that must be populated
        before a rolling statistic is emitted, rather than accepting a
        z-score computed from three observations.
    """

    zscore_window: int = 250
    surprise_window: int = 24
    change_windows: tuple[int, ...] = (21, 63)
    lags: tuple[int, ...] = (1, 5, 21)
    min_periods_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.zscore_window < 5:
            raise ValueError(f"zscore_window must be at least 5, got {self.zscore_window}")
        if self.surprise_window < 3:
            raise ValueError(
                f"surprise_window must span at least 3 releases to have a standard "
                f"deviation worth dividing by, got {self.surprise_window}"
            )
        if any(w < 1 for w in self.change_windows):
            raise ValueError(f"change_windows must be positive, got {self.change_windows}")
        if any(lag < 1 for lag in self.lags):
            raise ValueError(f"lags must be positive; lag 0 is the feature itself, got {self.lags}")
        if not 0.0 < self.min_periods_fraction <= 1.0:
            raise ValueError(
                f"min_periods_fraction must lie in (0, 1], got {self.min_periods_fraction}"
            )

    def min_periods(self, window: int) -> int:
        return max(2, round(window * self.min_periods_fraction))


@dataclass(frozen=True)
class MacroFeatures:
    """The feature matrix plus what a downstream model needs to trust it.

    Attributes
    ----------
    frame : the features, indexed by date. Leading rows are NaN while the
        rolling windows fill; they are not dropped here so the index still
        lines up with the price data the caller will join against.
    feature_names : column order, so a fitted model can reindex later data
        onto exactly the columns it was trained on.
    consensus_is_modelled : True when the surprise features were computed
        against a random-walk proxy rather than a survey consensus. When
        True, they are change features wearing a surprise's name -- see
        `data_ingest._model_consensus`.
    """

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    consensus_is_modelled: bool

    def complete(self) -> pd.DataFrame:
        """Rows with every feature populated -- what a model can be fitted on."""
        return self.frame.dropna(how="any")


def rolling_zscore(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """(x - rolling mean) / rolling std, over a trailing window ending at t.

    Uses pandas' default `ddof=1` sample standard deviation. Windows whose
    standard deviation is numerically zero -- a series that has not moved,
    which happens to a monthly print read daily -- yield NaN rather than
    inf, since "infinitely unusual" is not what a flat window means.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2 for a standard deviation, got {window}")
    periods = min_periods if min_periods is not None else max(2, window // 2)
    rolling = series.rolling(window, min_periods=periods)
    std = rolling.std()
    return ((series - rolling.mean()) / std.where(std > _MIN_STD)).rename(f"{series.name}_z")


def normalized_surprise(
    actual: pd.Series, consensus: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """(actual - consensus) scaled by the trailing std of past surprises.

    The scaling window counts *distinct surprise values*, not days: on a
    daily index a monthly release repeats about 21 times, and a rolling
    window over the repeated series would measure the repetition rather
    than the dispersion. So the raw surprise is deduplicated to one value
    per release, the standard deviation is computed there, and the result
    is forward-filled back onto the daily index.

    The standard deviation is shifted by one release, so the divisor uses
    only surprises known before the one being scaled -- without the shift,
    a large surprise inflates its own denominator and scales itself down.
    """
    raw = (actual - consensus).rename("surprise")
    # One row per release: keep the first day each new value appears.
    changed = raw.ne(raw.shift(1)) & raw.notna()
    per_release = raw[changed]
    if len(per_release) < 2:
        return pd.Series(np.nan, index=actual.index, name=f"{actual.name}_surprise_z")

    periods = min_periods if min_periods is not None else max(3, window // 2)
    scale = per_release.rolling(window, min_periods=periods).std().shift(1)
    normalized = (per_release / scale.where(scale > _MIN_STD)).reindex(actual.index).ffill()
    return normalized.rename(f"{actual.name}_surprise_z")


def yield_curve_slope(frame: pd.DataFrame, short: str = "yield_2y", long: str = "yield_10y") -> pd.Series:
    """Term spread, long minus short, in the units of the underlying yields.

    Negative values are an inverted curve. Raises KeyError naming the
    missing column, since a silently absent leg would produce an all-NaN
    feature that only shows up as a mysteriously empty training set.
    """
    for column in (short, long):
        if column not in frame.columns:
            raise KeyError(
                f"column {column!r} not in the macro frame (have: {list(frame.columns)}); "
                "align_to_daily produces yield_2y and yield_10y"
            )
    return (frame[long] - frame[short]).rename("curve_slope")


def add_lags(
    frame: pd.DataFrame, columns: list[str] | None = None, lags: tuple[int, ...] = (1, 5, 21)
) -> pd.DataFrame:
    """Append `<column>_lag<k>` for each column and lag, without dropping rows.

    Lags shift *forward* in time (`shift(k)`), so row t carries the value
    from t-k. Row t never sees row t+1.
    """
    target = list(frame.columns) if columns is None else columns
    unknown = [c for c in target if c not in frame.columns]
    if unknown:
        raise KeyError(f"cannot lag columns not present in the frame: {unknown}")
    if any(lag < 1 for lag in lags):
        raise ValueError(f"lags must be positive, got {lags}")

    lagged = {f"{column}_lag{lag}": frame[column].shift(lag) for column in target for lag in lags}
    return pd.concat([frame, pd.DataFrame(lagged, index=frame.index)], axis=1)


def build_macro_features(
    daily: pd.DataFrame,
    config: MacroFeatureConfig | None = None,
    consensus_is_modelled: bool = False,
) -> MacroFeatures:
    """Assemble the full feature matrix from an aligned daily macro frame.

    Parameters
    ----------
    daily : output of `data_ingest.align_to_daily` -- columns `cpi`,
        `unemployment`, `yield_2y`, `yield_10y`, `credit_spread`, plus
        `cpi_consensus` and `unemployment_consensus` where available.
    config : window lengths; defaults to `MacroFeatureConfig()`.
    consensus_is_modelled : pass through from the data source, recorded on
        the result so the surprise features carry their own caveat.

    Returns
    -------
    MacroFeatures whose frame holds, before lagging:

        cpi_yoy                  year-over-year inflation, percent
        cpi_yoy_z                its z-score
        unemployment, _z         level and z-score
        yield_2y, _z             level and z-score
        yield_10y, _z            level and z-score
        curve_slope, _z          10y - 2y and its z-score
        credit_spread, _z        level and z-score
        <series>_chg<w>          w-day change of each level
        cpi_surprise_z           normalized surprise, where consensus exists
        unemployment_surprise_z

    and then `_lag<k>` copies of every one of them.

    Raises KeyError if a required column is missing from `daily`.
    """
    cfg = config or MacroFeatureConfig()
    required = ["cpi", "unemployment", "yield_2y", "yield_10y", "credit_spread"]
    missing = [c for c in required if c not in daily.columns]
    if missing:
        raise KeyError(
            f"macro frame is missing {missing}; expected the output of "
            f"data_ingest.align_to_daily (have: {list(daily.columns)})"
        )
    if daily.empty:
        raise ValueError("macro frame is empty; nothing to build features from")

    features: dict[str, pd.Series] = {}

    # Inflation enters as a year-over-year rate, not an index level: the
    # level is a near-deterministic upward trend and carries no signal.
    # 252 business days approximates a calendar year on this index.
    cpi_yoy = ((daily["cpi"] / daily["cpi"].shift(252) - 1.0) * 100.0).rename("cpi_yoy")
    features["cpi_yoy"] = cpi_yoy

    levels = {
        "cpi_yoy": cpi_yoy,
        "unemployment": daily["unemployment"],
        "yield_2y": daily["yield_2y"],
        "yield_10y": daily["yield_10y"],
        "credit_spread": daily["credit_spread"],
        "curve_slope": yield_curve_slope(daily),
    }
    zscore_periods = cfg.min_periods(cfg.zscore_window)
    for name, series in levels.items():
        if name != "cpi_yoy":  # already added above
            features[name] = series.rename(name)
        features[f"{name}_z"] = rolling_zscore(
            series.rename(name), cfg.zscore_window, zscore_periods
        )
        for window in cfg.change_windows:
            features[f"{name}_chg{window}"] = (series - series.shift(window)).rename(
                f"{name}_chg{window}"
            )

    for column, consensus_column in (
        ("cpi", "cpi_consensus"),
        ("unemployment", "unemployment_consensus"),
    ):
        if consensus_column in daily.columns:
            features[f"{column}_surprise_z"] = normalized_surprise(
                daily[column].rename(column), daily[consensus_column], cfg.surprise_window
            )

    base = pd.DataFrame(features, index=daily.index)
    full = add_lags(base, list(base.columns), cfg.lags)
    full.index.name = "date"
    return MacroFeatures(
        frame=full,
        feature_names=tuple(full.columns),
        consensus_is_modelled=consensus_is_modelled,
    )
