"""Retrieval of the macro series the forecast is conditioned on.

Five FRED series, chosen to span the four channels that plausibly move an
equity return distribution over a one-month horizon:

    CPIAUCSL      inflation           monthly level, released ~13th of M+1
    UNRATE        labour slack        monthly rate, released first Friday of M+1
    DGS2          policy expectations daily, same-day
    DGS10         term premium/growth daily, same-day
    BAMLH0A0HYM2  credit risk         daily, one-day lag

Two things in here matter more than the HTTP plumbing.

**Publication lag.** FRED dates an observation by its *reference period*,
not its release date. CPI for January is dated 1 January and published
around 13 February. Joining on the reference date would hand the model six
weeks of information it could not have had, and macro features are exactly
where that mistake is easy to make and expensive: a look-ahead inflation
print is close to a look-ahead return. `align_to_daily` shifts every series
forward by its release lag before forward-filling, so a given row contains
only what had actually printed by that date.

**Revisions.** This module fetches the *current* vintage of each series.
CPI is revised for seasonal factors and unemployment for benchmark
population controls, so historical values here are not what the market saw
at the time. Doing this properly needs FRED's ALFRED vintage API
(`realtime_start`/`realtime_end`); the release lag handles the timing half
of the problem, and the revision half is a known, unaddressed limitation.

No API key is required. Without one -- or with `source="mock"` -- the module
returns a synthetic panel from `mock_data.macro` with the same series IDs,
frequencies and units, so downstream code cannot tell the difference.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import requests

from ..mock_data.macro import mock_consensus, mock_macro_panel

__all__ = [
    "FRED_SERIES",
    "MacroDataError",
    "MacroPanel",
    "SeriesSpec",
    "align_to_daily",
    "fetch_fred_series",
    "load_macro_panel",
]

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT_SEC = 15.0
_MISSING_VALUE = "."  # FRED's sentinel for an unavailable observation

Source = Literal["fred", "mock"]


@dataclass(frozen=True)
class SeriesSpec:
    """One FRED series and everything the pipeline needs to use it safely.

    Attributes
    ----------
    series_id : FRED identifier, used verbatim in the API call.
    column : the name this series takes in the aligned daily frame.
    units : human-readable units, for error messages and plots.
    release_lag_days : calendar days from the observation's reference date
        to its publication. Applied by `align_to_daily`. These are typical
        lags, not exact release-calendar dates -- a schedule-accurate
        implementation would join against FRED's release calendar, which is
        a separate endpoint.
    has_consensus : whether a survey consensus exists for this release.
        Market-priced series (yields, spreads) have none: they are not
        forecast and then announced, they trade continuously.
    """

    series_id: str
    column: str
    units: str
    release_lag_days: int
    has_consensus: bool


FRED_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("CPIAUCSL", "cpi", "index 1982-84=100", 43, True),
    SeriesSpec("UNRATE", "unemployment", "percent", 36, True),
    SeriesSpec("DGS2", "yield_2y", "percent", 1, False),
    SeriesSpec("DGS10", "yield_10y", "percent", 1, False),
    SeriesSpec("BAMLH0A0HYM2", "credit_spread", "percent", 1, False),
)

_BY_ID: Mapping[str, SeriesSpec] = {spec.series_id: spec for spec in FRED_SERIES}


class MacroDataError(RuntimeError):
    """Raised when macro data cannot be retrieved or parsed."""


@dataclass(frozen=True)
class MacroPanel:
    """Raw observations and consensus forecasts, at native frequencies.

    Attributes
    ----------
    observations : FRED series ID -> Series dated by *reference period*.
    consensus : FRED series ID -> Series of survey consensus, dated
        identically to the corresponding observation. Only the monthly
        releases appear; the rest have no consensus to speak of.
    source : "fred" if fetched live, "mock" if synthesized.
    """

    observations: dict[str, pd.Series]
    consensus: dict[str, pd.Series]
    source: Source

    def __post_init__(self) -> None:
        missing = [spec.series_id for spec in FRED_SERIES if spec.series_id not in self.observations]
        if missing:
            raise ValueError(f"MacroPanel is missing required series {missing}")


def fetch_fred_series(
    series_id: str,
    start: str,
    end: str,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> pd.Series:
    """Fetch one series from FRED as a float Series indexed by reference date.

    Parameters
    ----------
    series_id : e.g. "CPIAUCSL".
    start, end : ISO dates bounding the observation period.
    api_key : FRED key, else the FRED_API_KEY environment variable.
        Free at https://fred.stlouisfed.org/docs/api/api_key.html.
    session : optional requests.Session for connection reuse across series.

    Missing observations (FRED sends ".") are dropped rather than filled:
    a holiday has no 10-year yield, and inventing one would put a fake
    print into the lag/z-score windows.

    Raises MacroDataError on a missing key, a failed request, a malformed
    payload, or a series that comes back empty.
    """
    if not series_id:
        raise ValueError("series_id must be a non-empty FRED identifier")
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise MacroDataError(
            "no FRED API key: pass api_key= or set FRED_API_KEY (free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html). Use "
            "load_macro_panel(source='mock') to run without one."
        )

    http = session or requests.Session()
    try:
        response = http.get(
            FRED_BASE_URL,
            params={
                "series_id": series_id,
                "api_key": key,
                "file_type": "json",
                "observation_start": start,
                "observation_end": end,
            },
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise MacroDataError(f"FRED request for {series_id} timed out after {REQUEST_TIMEOUT_SEC}s") from exc
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
            f"FRED payload for {series_id} has no observations list: "
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
            continue  # one unparseable print should not void the series
        dates.append(pd.Timestamp(row["date"]))

    if not dates:
        raise MacroDataError(
            f"FRED returned no usable observations for {series_id} between {start} and {end}"
        )
    return pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name=series_id).sort_index()


def _model_consensus(observations: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Random-walk stand-in for survey consensus on the monthly releases.

    Real consensus figures are licensed and have no free API, so with live
    FRED data there is nothing to difference against. The standard
    substitute is a naive forecast: consensus for month M is the value
    published for M-1. Differencing against it yields the *change*, not the
    surprise, and the two are different objects -- a 0.3% CPI print is a
    surprise only relative to what forecasters expected, and a random walk
    expects last month.

    So this makes the surprise feature a change feature when running on
    live data, which is weaker but not wrong, and is flagged in
    `MacroFeatures.consensus_is_modelled`. Point `load_macro_panel` at a
    real consensus source before trusting the surprise channel.
    """
    return {
        spec.series_id: observations[spec.series_id].shift(1).rename(f"{spec.series_id}_consensus")
        for spec in FRED_SERIES
        if spec.has_consensus and spec.series_id in observations
    }


def load_macro_panel(
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    api_key: str | None = None,
    source: Source | None = None,
    seed: int | None = None,
    session: requests.Session | None = None,
) -> MacroPanel:
    """Load all five series, from FRED when possible and mock data otherwise.

    Parameters
    ----------
    start, end : ISO dates bounding the observation period.
    api_key : FRED key; falls back to FRED_API_KEY.
    source : force "fred" or "mock". When None (the default), FRED is used
        if a key is available and mock data otherwise, so a clean checkout
        runs without configuration and a configured one silently upgrades.
    seed : seed for the mock generator; ignored for live data.
    session : optional requests.Session shared across the five fetches.

    Raises MacroDataError when source="fred" is forced and the fetch fails.
    A missing key with source=None is not an error -- it falls back.
    """
    resolved: Source
    if source is not None:
        if source not in ("fred", "mock"):
            raise ValueError(f"source must be 'fred', 'mock' or None, got {source!r}")
        resolved = source
    else:
        resolved = "fred" if (api_key or os.environ.get("FRED_API_KEY")) else "mock"

    if resolved == "mock":
        observations = mock_macro_panel(start=start, end=end, seed=seed)
        return MacroPanel(observations, mock_consensus(observations, seed=seed), "mock")

    http = session or requests.Session()
    observations = {
        spec.series_id: fetch_fred_series(spec.series_id, start, end, api_key, http)
        for spec in FRED_SERIES
    }
    return MacroPanel(observations, _model_consensus(observations), "fred")


def align_to_daily(
    panel: MacroPanel, calendar: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Lag each series by its publication delay, then forward-fill to daily.

    The result is a point-in-time frame: the row for date d holds, for each
    series, the most recent value that had actually been published on or
    before d. Monthly series therefore step once a month and then hold flat
    -- which is correct, because that is all a forecaster knows between
    releases -- and the step lands on the release date, not the reference
    date.

    Parameters
    ----------
    panel : the observations and consensus to align.
    calendar : business-day index to align onto. Defaults to every business
        day spanned by the observations, extended by the largest release
        lag so the final print has somewhere to land.

    Returns
    -------
    DataFrame indexed by date with one column per `SeriesSpec.column`, plus
    `<column>_consensus` for series that have one. Leading rows before the
    first publication of a series are NaN; they are left in place rather
    than back-filled, since dropping or filling them is the caller's choice
    and back-filling would reintroduce look-ahead.
    """
    if calendar is None:
        first = min(s.index.min() for s in panel.observations.values())
        last = max(s.index.max() for s in panel.observations.values())
        last = last + pd.Timedelta(days=max(spec.release_lag_days for spec in FRED_SERIES))
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
            # groupby(level=0).last() collapses the rare case of two prints
            # landing on the same business day after shifting.
            published = published.groupby(level=0).last().sort_index()
            columns[f"{spec.column}{suffix}"] = published.reindex(
                published.index.union(calendar)
            ).ffill().reindex(calendar)

    frame = pd.DataFrame(columns, index=calendar)
    frame.index.name = "date"
    return frame.astype(np.float64)
