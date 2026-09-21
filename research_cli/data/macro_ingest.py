"""FRED macro series and the release calendar, ported from the quant pipeline.

Ported with minimal changes from `quant_pipeline/macro_model/data_ingest.py`
and the surprise logic from `feature_engineering.py`. The series table, the
publication lags, the FRED client and `align_to_daily` are carried over
essentially verbatim; what is new is `releases_on`, which answers the
question this tool actually asks -- *what macro data came out on the day the
user is asking about, and was it a surprise?*

**Publication lag, restated because it is the whole point.** FRED dates an
observation by its *reference period*, not its release date. CPI for
January is dated 1 January and published around 13 February. A tool that
joined on the reference date would tell a user that January's inflation
print "coincided with" a move on 2 January -- six weeks before anyone knew
the number. `align_to_daily` shifts each series by its release lag first.

**The econ-calendar limitation, stated plainly.** There is no free,
terms-of-service-clean API for a forward-looking economic calendar with
consensus forecasts. `investpy`, the package usually reached for, has been
blocked by its upstream source since 2022 and is unmaintained. So release
*dates* here are derived from each series' typical publication lag, not
from an actual calendar, and they can be off by a few days around holidays
and revisions. Consensus forecasts are not available at all on the free
tier, so surprises are measured against a **random-walk proxy** -- last
month's value -- which makes them *change* measures rather than true
surprises. Both facts are carried on `MacroRelease.is_estimated_date` and
`MacroRelease.consensus_is_modelled` and are surfaced in the report, rather
than being buried here.

With a FRED API key in `FRED_API_KEY` the series are real. Without one,
everything falls back to the synthetic panel in `mock_data`, clearly
labelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import requests

from ..settings import get_key
from .mock_data import mock_macro_panel

__all__ = [
    "FRED_SERIES",
    "MacroDataError",
    "MacroPanel",
    "MacroRelease",
    "SeriesSpec",
    "align_to_daily",
    "fetch_fred_series",
    "load_macro_panel",
    "normalized_surprise",
    "releases_on",
    "yield_moves_on",
]

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT_SEC = 15.0
_MISSING_VALUE = "."  # FRED's sentinel for an unavailable observation
_MIN_STD = 1e-10

Source = Literal["auto", "live", "mock"]


@dataclass(frozen=True)
class SeriesSpec:
    """One FRED series and everything needed to use it without look-ahead.

    Ported unchanged from the quant pipeline, with `label` and `units_label`
    added so a beginner-facing report can name the series in English rather
    than as "BAMLH0A0HYM2".
    """

    series_id: str
    column: str
    label: str
    units_label: str
    release_lag_days: int
    has_consensus: bool


FRED_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("CPIAUCSL", "cpi", "Consumer price index (inflation)", "index", 43, True),
    SeriesSpec("UNRATE", "unemployment", "Unemployment rate", "%", 36, True),
    SeriesSpec("DGS2", "yield_2y", "2-year Treasury yield", "%", 1, False),
    SeriesSpec("DGS10", "yield_10y", "10-year Treasury yield", "%", 1, False),
    # FRED serves this one as a rolling ~3-year window (ICE licence), so
    # its history is much shorter than the others'. See the module docstring.
    SeriesSpec(
        "BAMLH0A0HYM2", "credit_spread", "High-yield credit spread", "%", 1, False
    ),
)

_BY_COLUMN = {spec.column: spec for spec in FRED_SERIES}


class MacroDataError(RuntimeError):
    """Raised when macro data cannot be retrieved or parsed."""


@dataclass(frozen=True)
class MacroPanel:
    """Raw observations and consensus proxies, at native frequencies."""

    observations: dict[str, pd.Series]
    consensus: dict[str, pd.Series]
    source: str

    def __post_init__(self) -> None:
        missing = [s.series_id for s in FRED_SERIES if s.series_id not in self.observations]
        if missing:
            raise ValueError(f"MacroPanel is missing required series {missing}")


@dataclass(frozen=True)
class MacroRelease:
    """One macro number that became public on a given day.

    Attributes
    ----------
    series_id, label, units_label : what was released.
    reference_period : the month or day the number describes.
    release_date : when it became public (see the estimation caveat).
    value : the released value.
    previous_value : the prior period's value, or None if unavailable.
    change : value minus previous, in the series' own units.
    surprise_z : the change divided by the trailing standard deviation of
        past changes. Roughly "how many normal moves was this". None when
        there is not enough history.
    is_estimated_date : True when the release date was derived from a
        typical publication lag rather than an actual calendar.
    consensus_is_modelled : True when the "surprise" is measured against a
        random-walk proxy rather than a survey consensus -- which is always,
        on free data.
    """

    series_id: str
    label: str
    units_label: str
    reference_period: date
    release_date: date
    value: float
    previous_value: float | None
    change: float | None
    surprise_z: float | None
    is_estimated_date: bool = True
    consensus_is_modelled: bool = True

    @property
    def formatted_value(self) -> str:
        """The value with its units attached, spaced the way English is."""
        if self.units_label == "%":
            return f"{self.value:.2f}%"
        return f"{self.value:.2f} {self.units_label}"

    def describe(self) -> str:
        """A sentence a beginner can read without knowing what a z-score is."""
        if self.change is None:
            return f"{self.label} was published at {self.formatted_value}."
        direction = (
            "higher than" if self.change > 0
            else "lower than" if self.change < 0
            else "unchanged from"
        )
        sentence = (
            f"{self.label} came in at {self.formatted_value}, "
            f"{abs(self.change):.2f} {direction} the previous reading"
        )
        if self.surprise_z is not None and abs(self.surprise_z) >= 1.0:
            sentence += (
                f" -- a bigger step than its recent average, by about "
                f"{abs(self.surprise_z):.1f} times the usual month-to-month variation"
            )
        return sentence + "."


# --- fetching (ported) -----------------------------------------------------

def fetch_fred_series(
    series_id: str,
    start: date,
    end: date,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> pd.Series:
    """Fetch one series from FRED as a float Series indexed by reference date.

    Ported unchanged from the quant pipeline. Missing observations (FRED
    sends ".") are dropped rather than filled: a holiday has no 10-year
    yield, and inventing one would put a fake print into the change windows.
    """
    if not series_id:
        raise ValueError("series_id must be a non-empty FRED identifier")
    key = api_key or get_key("FRED_API_KEY")
    if not key:
        raise MacroDataError(
            "no FRED API key: set FRED_API_KEY (free at "
            "https://fred.stlouisfed.org/docs/api/api_key.html)"
        )

    http = session or requests.Session()
    try:
        response = http.get(
            FRED_BASE_URL,
            params={
                "series_id": series_id,
                "api_key": key,
                "file_type": "json",
                "observation_start": start.isoformat(),
                "observation_end": end.isoformat(),
            },
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise MacroDataError(f"FRED request for {series_id} timed out") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        hint = " (check the API key)" if status in (400, 403) else ""
        raise MacroDataError(f"FRED returned HTTP {status} for {series_id}{hint}") from exc
    except requests.RequestException as exc:
        raise MacroDataError(f"FRED request for {series_id} failed: {exc}") from exc
    except ValueError as exc:
        raise MacroDataError(f"FRED returned a non-JSON response for {series_id}") from exc

    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise MacroDataError(
            f"FRED payload for {series_id} has no observations: "
            f"{payload.get('error_message', 'no detail')}"
        )

    dates, values = [], []
    for row in observations:
        raw = row.get("value")
        if raw is None or raw == _MISSING_VALUE:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
        dates.append(pd.Timestamp(row["date"]))

    if not dates:
        raise MacroDataError(f"FRED returned no usable observations for {series_id}")
    return pd.Series(
        values, index=pd.DatetimeIndex(dates, name="date"), name=series_id
    ).sort_index()


def _random_walk_consensus(observations: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Last period's value as the stand-in for a survey consensus.

    Ported from the quant pipeline, where the same substitution is made and
    the same caveat applies: differencing against a random walk gives the
    *change*, not the surprise. A 0.4% CPI print is a surprise only relative
    to what forecasters expected, and a random walk expects last month.
    """
    return {
        spec.series_id: observations[spec.series_id].shift(1)
        for spec in FRED_SERIES
        if spec.has_consensus and spec.series_id in observations
    }


def load_macro_panel(
    start: date, end: date, source: Source = "auto", api_key: str | None = None
) -> MacroPanel:
    """Load all five series, from FRED when a key exists and mock otherwise.

    A missing key is not an error under source="auto" -- it falls back, so a
    user with no accounts still gets a working tool with a clear label.
    """
    if source not in ("auto", "live", "mock"):
        raise ValueError(f"source must be 'auto', 'live' or 'mock', got {source!r}")

    if source == "mock" or (source == "auto" and not (api_key or get_key("FRED_API_KEY"))):
        observations = mock_macro_panel(start, end)
        return MacroPanel(observations, _random_walk_consensus(observations), "mock")

    try:
        http = requests.Session()
        observations = {
            spec.series_id: fetch_fred_series(spec.series_id, start, end, api_key, http)
            for spec in FRED_SERIES
        }
    except MacroDataError:
        if source == "live":
            raise
        observations = mock_macro_panel(start, end)
        return MacroPanel(observations, _random_walk_consensus(observations), "mock")

    return MacroPanel(observations, _random_walk_consensus(observations), "fred")


def align_to_daily(panel: MacroPanel, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Lag each series by its publication delay, then forward-fill to daily.

    Ported unchanged. The result is point-in-time: the row for date d holds,
    for each series, the most recent value that had actually been published
    on or before d. Leading rows before a series' first publication stay
    NaN rather than being back-filled, which would reintroduce look-ahead.
    """
    if calendar is None:
        first = min(s.index.min() for s in panel.observations.values())
        last = max(s.index.max() for s in panel.observations.values()) + pd.Timedelta(
            days=max(spec.release_lag_days for spec in FRED_SERIES)
        )
        calendar = pd.bdate_range(start=first, end=last, name="date")
    if len(calendar) == 0:
        raise ValueError("calendar is empty; nothing to align onto")

    columns: dict[str, pd.Series] = {}
    for spec in FRED_SERIES:
        lag = pd.Timedelta(days=spec.release_lag_days)
        for suffix, store in (("", panel.observations), ("_consensus", panel.consensus)):
            series = store.get(spec.series_id)
            if series is None:
                continue
            published = series.copy()
            published.index = published.index + lag
            published = published.groupby(level=0).last().sort_index()
            columns[f"{spec.column}{suffix}"] = (
                published.reindex(published.index.union(calendar)).ffill().reindex(calendar)
            )

    frame = pd.DataFrame(columns, index=calendar)
    frame.index.name = "date"
    return frame.astype(np.float64)


def normalized_surprise(
    actual: pd.Series, consensus: pd.Series, window: int = 24
) -> pd.Series:
    """(actual - consensus) scaled by the trailing std of past surprises.

    Ported from `quant_pipeline/macro_model/feature_engineering.py`. The
    scaling window counts *distinct releases*, not days: on a daily index a
    monthly release repeats about 21 times, and a rolling window over the
    repeated series would measure the repetition rather than the dispersion.

    The standard deviation is shifted by one release, so the divisor uses
    only surprises known before the one being scaled -- without the shift, a
    large surprise inflates its own denominator and scales itself down.
    """
    raw = (actual - consensus).rename("surprise")
    changed = raw.ne(raw.shift(1)) & raw.notna()
    per_release = raw[changed]
    if len(per_release) < 3:
        return pd.Series(np.nan, index=actual.index, name="surprise_z")

    scale = per_release.rolling(window, min_periods=3).std().shift(1)
    return (
        (per_release / scale.where(scale > _MIN_STD))
        .reindex(actual.index)
        .ffill()
        .rename("surprise_z")
    )


# --- what this tool actually asks -----------------------------------------

def releases_on(
    panel: MacroPanel, target: date, window_days: int = 0
) -> list[MacroRelease]:
    """Which macro numbers became public on (or near) a given day.

    Parameters
    ----------
    panel : the loaded series.
    target : the day being asked about.
    window_days : also include releases this many days either side. Useful
        because the derived release dates are approximate -- a CPI print
        "on" the 13th may really have landed on the 11th or 14th.

    Returns a list ordered by absolute surprise, largest first, so the
    caller can take the most notable releases without re-sorting.

    Only monthly series produce releases. Daily market series (yields,
    spreads) are prices that trade continuously, not scheduled
    announcements, and are handled by `yield_moves_on`.
    """
    if window_days < 0:
        raise ValueError(f"window_days must be non-negative, got {window_days}")

    releases: list[MacroRelease] = []
    lower = pd.Timestamp(target - timedelta(days=window_days))
    upper = pd.Timestamp(target + timedelta(days=window_days))

    for spec in FRED_SERIES:
        if not spec.has_consensus:  # monthly scheduled releases only
            continue
        series = panel.observations.get(spec.series_id)
        if series is None or len(series) < 4:
            continue

        release_dates = series.index + pd.Timedelta(days=spec.release_lag_days)
        hits = np.flatnonzero((release_dates >= lower) & (release_dates <= upper))
        if len(hits) == 0:
            continue

        changes = series.diff()
        # A surprise is a deviation from what was *expected*, and for a
        # trending series the expectation is not zero. CPI is an index that
        # rises nearly every month, so dividing a raw change by its standard
        # deviation would score every ordinary print as a multi-sigma shock.
        # The baseline is therefore the trailing mean change -- a
        # drift-aware random walk -- and the deviation from it is what gets
        # scaled. Both statistics are shifted by one so a print cannot
        # normalize itself; same discipline as `normalized_surprise`.
        expected = changes.rolling(24, min_periods=3).mean().shift(1)
        dispersion = changes.rolling(24, min_periods=3).std().shift(1)

        for position in hits:
            value = float(series.iloc[position])
            previous = float(series.iloc[position - 1]) if position > 0 else None
            change = None if previous is None else value - previous
            scale = float(dispersion.iloc[position]) if position < len(dispersion) else np.nan
            baseline = float(expected.iloc[position]) if position < len(expected) else np.nan
            surprise_z = (
                float((change - baseline) / scale)
                if change is not None
                and np.isfinite(scale)
                and np.isfinite(baseline)
                and scale > _MIN_STD
                else None
            )
            releases.append(
                MacroRelease(
                    series_id=spec.series_id,
                    label=spec.label,
                    units_label=spec.units_label,
                    reference_period=series.index[position].date(),
                    release_date=release_dates[position].date(),
                    value=value,
                    previous_value=previous,
                    change=change,
                    surprise_z=surprise_z,
                )
            )

    releases.sort(key=lambda r: abs(r.surprise_z) if r.surprise_z is not None else 0.0, reverse=True)
    return releases


def yield_moves_on(daily: pd.DataFrame, target: date) -> dict[str, float]:
    """One-day changes in the daily market series, in their own units.

    Returns a mapping like {"yield_2y": 0.04, "yield_10y": -0.02,
    "credit_spread": 0.11} -- changes in percentage points, positive meaning
    the yield or spread rose. Series with no session at or before `target`,
    or with no prior session to difference against, are omitted rather than
    reported as zero, because "unchanged" and "unknown" are different facts.
    """
    if daily.empty:
        return {}
    at_or_before = daily.index[daily.index <= pd.Timestamp(target)]
    if len(at_or_before) < 2:
        return {}

    session = at_or_before[-1]
    position = daily.index.get_loc(session)
    moves: dict[str, float] = {}
    for spec in FRED_SERIES:
        if spec.has_consensus or spec.column not in daily.columns:
            continue
        column = daily[spec.column]
        current, previous = column.iloc[position], column.iloc[position - 1]
        if np.isfinite(current) and np.isfinite(previous):
            moves[spec.column] = float(current - previous)
    return moves
