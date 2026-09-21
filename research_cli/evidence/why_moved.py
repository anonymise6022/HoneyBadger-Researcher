"""Assemble the evidence for "why did X move" -- without ever claiming a cause.

The honest answer to "why did SPY fall today" is that nobody knows. What
*can* be said is: here is what else happened at the same time, and here is
how often the symbol has moved this way on the other days when that same
thing happened. That is what this module builds.

**How a hit rate is defined here.** For each candidate factor -- the VIX
move, the 10-year yield move, sector breadth, and so on -- the factor's
daily history is cut into quintiles. Today's value falls in one of them.
The hit rate is the fraction of *other* historical days in that same
quintile on which the symbol moved in the same direction as today.

    "On the 20% of days when the VIX rose the most, SPY fell on 68% of
     them. Today's VIX move is in that group."

Quintiles rather than a hand-tuned threshold, for two reasons. A threshold
like "VIX up more than 5%" is a free parameter that can be tuned until the
hit rate looks impressive, and quintiles have none. And a quintile always
yields about a fifth of the history -- roughly 200 days over four years --
which is a large enough sample that the binomial error is around 3.5 points
rather than the 15 points a rare-threshold sample would give.

**Why the base rate is always shown beside it.** A 68% hit rate sounds
decisive until you learn the symbol fell on 64% of *all* days in that
period. `CandidateFactor.lift` is the difference, and `beats_base_rate`
requires it to clear two standard errors. Most factors on most days do not
clear it, and the report says so rather than ranking noise.

**The day itself is excluded from its own sample.** History runs up to the
session before the one being explained. Including it would let a large move
contribute to the statistic used to explain it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..data.macro_ingest import (
    MacroDataError,
    load_macro_panel,
    releases_on,
)
from ..data.price_ingest import (
    PriceObservation,
    Source,
    build_observation,
    fetch_many,
)
from ..query_parser import ParsedQuery
from .schema import (
    CandidateFactor,
    Counterevidence,
    EvidenceBundle,
    MacroReleaseRecord,
    Observation,
)

__all__ = [
    "DEFAULT_HISTORY_YEARS",
    "FACTOR_SYMBOLS",
    "MEGACAP_WEIGHTS",
    "SECTOR_ETFS",
    "build_why_moved_bundle",
    "quintile_hit_rate",
]

#: History used for hit rates. Four years rather than three: quintile
#: conditioning keeps a fifth of the days, and four years leaves ~200 in the
#: conditioning bin, where the binomial error on a hit rate is about 3.5
#: points. Three years gives ~150 and an error near 4 points; both are
#: usable, and the extra year is nearly free once cached.
DEFAULT_HISTORY_YEARS = 4.0

_MIN_HISTORY_SESSIONS = 250  # roughly a year; below this no rate is quoted
_N_QUINTILES = 5

#: Sector ETFs, used for breadth. The eleven GICS sectors.
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health care", "XLY": "Consumer discretionary",
    "XLP": "Consumer staples", "XLI": "Industrials", "XLU": "Utilities",
    "XLB": "Materials", "XLRE": "Real estate", "XLC": "Communication services",
}

#: Approximate S&P 500 index weights for the largest constituents, used to
#: estimate how much of an index move one stock accounted for. These are
#: **static and approximate** -- real weights drift daily with price and
#: change on rebalance -- so the contribution figures are indicative, and
#: the bundle says so in its limitations. Exact weights need a paid
#: constituents feed.
MEGACAP_WEIGHTS: dict[str, float] = {
    "AAPL": 0.070, "MSFT": 0.065, "NVDA": 0.062, "AMZN": 0.038,
    "META": 0.025, "GOOGL": 0.021, "AVGO": 0.017, "TSLA": 0.016,
}

#: Cross-asset context symbols, with the label and units each carries.
FACTOR_SYMBOLS: dict[str, tuple[str, str]] = {
    "^VIX": ("Volatility index (VIX)", "%"),
    "^TNX": ("10-year Treasury yield", "%"),
    "^FVX": ("5-year Treasury yield", "%"),
    "HYG": ("High-yield corporate bonds (HYG)", "%"),
    "UUP": ("US dollar (UUP)", "%"),
}

#: Index symbols get breadth and single-stock attribution; single stocks get
#: market-beta decomposition instead.
_INDEX_SYMBOLS = frozenset({"SPY", "QQQ", "DIA", "IWM", "VOO", "VTI", "IVV"})


@dataclass(frozen=True)
class _FactorInput:
    """A factor's daily history plus today's reading, before scoring."""

    factor_id: str
    label: str
    units: str
    series: pd.Series          # daily values, indexed by date
    today_value: float
    what_happened: str
    note: str | None = None
    relationship: str = "external"


def quintile_hit_rate(
    factor: pd.Series,
    target_returns: pd.Series,
    today_value: float,
    target_sign: int,
    n_bins: int = _N_QUINTILES,
) -> tuple[float | None, float | None, int, str]:
    """How often the target moved `target_sign` on days like today.

    Parameters
    ----------
    factor : the factor's daily values over the history window.
    target_returns : the symbol's daily returns, same index.
    today_value : the factor's reading on the day being explained.
    target_sign : +1 if the symbol rose, -1 if it fell.
    n_bins : quantile bins. Five keeps ~20% of days per bin.

    Returns
    -------
    (hit_rate, base_rate, sample_size, condition_description). The hit rate
    is None when there is too little history or the bin cannot be formed --
    a discrete factor with heavy ties, for instance, where `qcut` collapses
    bins and the remaining ones are not comparable.

    Notes
    -----
    Bin edges come from the history itself, so the conditioning set is
    defined by what the factor actually did rather than by a threshold
    somebody chose. Ties are dropped rather than jittered: silently
    perturbing data to force five bins would make the sample sizes a
    fiction.
    """
    aligned = pd.concat(
        [factor.rename("factor"), target_returns.rename("target")], axis=1
    ).dropna()
    if len(aligned) < _MIN_HISTORY_SESSIONS:
        return None, None, len(aligned), "not enough history"
    if not np.isfinite(today_value):
        return None, None, 0, "today's value is unavailable"

    base_rate = float((np.sign(aligned["target"]) == target_sign).mean())

    try:
        bins, edges = pd.qcut(
            aligned["factor"], n_bins, retbins=True, labels=False, duplicates="drop"
        )
    except (ValueError, IndexError):
        return None, base_rate, 0, "factor has too few distinct values to bin"
    if bins.nunique() < 2:
        return None, base_rate, 0, "factor has too few distinct values to bin"

    # Place today in a bin using the same edges. np.searchsorted on the
    # interior edges reproduces qcut's assignment and extends naturally to a
    # value beyond the historical range, which lands in the outer bin.
    today_bin = int(np.clip(np.searchsorted(edges[1:-1], today_value, side="right"), 0, bins.max()))
    mask = (bins == today_bin).to_numpy()
    sample_size = int(mask.sum())
    if sample_size == 0:
        return None, base_rate, 0, "no comparable historical days"

    hit_rate = float((np.sign(aligned["target"].to_numpy()[mask]) == target_sign).mean())

    n_actual = int(bins.nunique())
    lower, upper = edges[today_bin], edges[today_bin + 1]
    percentile_lower = 100.0 * today_bin / n_actual
    percentile_upper = 100.0 * (today_bin + 1) / n_actual
    description = (
        f"days when this factor was between {lower:+.2f} and {upper:+.2f} "
        f"(the {percentile_lower:.0f}-{percentile_upper:.0f}th percentile of its history)"
    )
    return hit_rate, base_rate, sample_size, description


def _score_factor(
    factor_input: _FactorInput,
    target_returns: pd.Series,
    target_sign: int,
    history_start: date,
    history_end: date,
    direction_agrees: bool = True,
) -> CandidateFactor:
    """Turn a factor's history into a scored `CandidateFactor`."""
    hit_rate, base_rate, sample_size, condition = quintile_hit_rate(
        factor_input.series, target_returns, factor_input.today_value, target_sign
    )
    return CandidateFactor(
        factor_id=factor_input.factor_id,
        label=factor_input.label,
        what_happened=factor_input.what_happened,
        observed_value=float(factor_input.today_value),
        observed_units=factor_input.units,
        condition_description=condition,
        historical_hit_rate=hit_rate,
        base_rate=base_rate,
        sample_size=sample_size,
        history_start=history_start,
        history_end=history_end,
        direction_agrees=direction_agrees,
        relationship=factor_input.relationship,  # type: ignore[arg-type]
        note=factor_input.note,
    )


def _daily_returns(frame: pd.DataFrame) -> pd.Series:
    """Close-to-close percentage returns, indexed by date."""
    closes = frame["Close"].astype(float)
    return (closes.pct_change() * 100.0).dropna()


def _value_on(series: pd.Series, target: date) -> float:
    """The series value on the last index date at or before `target`."""
    at_or_before = series.index[series.index <= pd.Timestamp(target)]
    if len(at_or_before) == 0:
        return float("nan")
    return float(series.loc[at_or_before[-1]])


def _cross_asset_factors(
    histories: dict[str, pd.DataFrame], session: date
) -> list[_FactorInput]:
    """Build the VIX / rates / credit / dollar factors from fetched prices."""
    inputs: list[_FactorInput] = []
    for symbol, (label, units) in FACTOR_SYMBOLS.items():
        frame = histories.get(symbol)
        if frame is None or len(frame) < 30:
            continue
        returns = _daily_returns(frame)
        today = _value_on(returns, session)
        if not np.isfinite(today):
            continue
        # ^TNX and ^FVX quote the yield itself, so a percentage change in
        # them is a change in the yield level, not a price move. Saying
        # "the 10-year yield rose 1.8%" is right but easy to misread, so
        # the sentence spells out that it is the yield that moved.
        is_yield = symbol in ("^TNX", "^FVX")
        verb = "rose" if today > 0 else "fell"
        # The label is used verbatim: lowercasing it turns "(HYG)" into
        # "(hyg)", which reads as a typo to anyone who knows the ticker.
        what = (
            f"{label} {verb} {abs(today):.2f}% on {session.isoformat()}"
            + (" -- that is a move in the yield itself, not in bond prices" if is_yield else "")
        )
        inputs.append(
            _FactorInput(
                factor_id=symbol.lstrip("^").lower(),
                label=label,
                units=units,
                series=returns,
                today_value=today,
                what_happened=what,
            )
        )
    return inputs


def _sector_breadth_factor(
    histories: dict[str, pd.DataFrame], session: date, target_sign: int, is_index: bool
) -> _FactorInput | None:
    """Fraction of sector ETFs that moved the same way as the symbol.

    Breadth separates "the whole market moved" from "one corner of it did",
    which is the distinction a beginner most often misses when a headline
    blames a single story for an index move.
    """
    returns = {
        symbol: _daily_returns(frame)
        for symbol, frame in histories.items()
        if symbol in SECTOR_ETFS and len(frame) >= 30
    }
    if len(returns) < 5:
        return None

    panel = pd.DataFrame(returns).dropna(how="all")
    if len(panel) < _MIN_HISTORY_SESSIONS:
        return None

    # Breadth is signed toward the *symbol's* direction, so the factor means
    # the same thing whether the symbol rose or fell.
    breadth = (np.sign(panel) == target_sign).mean(axis=1) * 100.0
    today = _value_on(breadth, session)
    if not np.isfinite(today):
        return None

    session_row = panel.loc[panel.index[panel.index <= pd.Timestamp(session)][-1]]
    same_way = int((np.sign(session_row.dropna()) == target_sign).sum())
    total = int(session_row.notna().sum())
    word = "rose" if target_sign > 0 else "fell"
    return _FactorInput(
        factor_id="sector_breadth",
        label="Sector breadth (% of sectors moving alike)",
        units="%",
        series=breadth,
        today_value=today,
        what_happened=(
            f"{same_way} of {total} stock-market sectors {word} on the same day "
            f"({today:.0f}% of them), which shows how broad the move was"
        ),
        # For an index, breadth is a description of the same move seen a
        # different way -- the index is built from these sectors -- so its
        # hit rate is near-tautological. For a single stock it is genuine
        # outside context.
        relationship="mechanical" if is_index else "external",
        note=(
            "an index is built from its sectors, so this describes the shape of the "
            "move rather than explaining it"
            if is_index else None
        ),
    )


def _megacap_factors(
    histories: dict[str, pd.DataFrame], session: date, index_return_pct: float
) -> tuple[list[_FactorInput], list[str]]:
    """Largest single-stock contributions to an index move.

    Contribution is approximated as (static index weight) x (stock return).
    The weights drift daily and are stale the moment they are written down,
    so these figures are indicative; the caller records that limitation.
    """
    contributions: list[tuple[str, float, float]] = []
    for symbol, weight in MEGACAP_WEIGHTS.items():
        frame = histories.get(symbol)
        if frame is None or len(frame) < 30:
            continue
        returns = _daily_returns(frame)
        today = _value_on(returns, session)
        if np.isfinite(today):
            contributions.append((symbol, today, weight * today))

    if not contributions:
        return [], []

    contributions.sort(key=lambda item: abs(item[2]), reverse=True)
    inputs: list[_FactorInput] = []
    notes: list[str] = []
    for symbol, stock_return, contribution in contributions[:2]:
        share = (
            abs(contribution / index_return_pct) * 100.0
            if abs(index_return_pct) > 1e-9
            else float("nan")
        )
        share_phrase = (
            f", roughly {share:.0f}% of the index's move"
            if np.isfinite(share) and share < 200.0
            else ""
        )
        inputs.append(
            _FactorInput(
                factor_id=f"megacap_{symbol.lower()}",
                label=f"{symbol} (large index member)",
                units="%",
                series=_daily_returns(histories[symbol]),
                today_value=stock_return,
                what_happened=(
                    f"{symbol} moved {stock_return:+.2f}%. At an index weight of about "
                    f"{MEGACAP_WEIGHTS[symbol] * 100:.1f}% that contributed roughly "
                    f"{contribution:+.2f} percentage points{share_phrase}"
                ),
                relationship="mechanical",
                note="a component of the index; this is arithmetic, not explanation",
            )
        )
    notes.append(
        "Single-stock contributions use approximate, static index weights, so they "
        "indicate scale rather than exact attribution."
    )
    return inputs, notes


def _market_factor(
    market_history: pd.DataFrame | None,
    target_returns: pd.Series,
    session: date,
    observation: PriceObservation,
) -> tuple[_FactorInput | None, list[Counterevidence], str | None]:
    """Decompose a single stock's move into market and stock-specific parts.

    Usually the single most useful thing a beginner can be told. If the S&P
    fell 1.8% and their stock fell 2.0%, almost nothing happened to their
    company -- they own a slice of the market, and the market moved. Beta is
    estimated by ordinary least squares over the history window, ending
    before the session being explained.
    """
    if market_history is None or len(market_history) < _MIN_HISTORY_SESSIONS:
        return None, [], None

    market_returns = _daily_returns(market_history)
    aligned = pd.concat(
        [market_returns.rename("market"), target_returns.rename("stock")], axis=1
    ).dropna()
    # Estimate beta strictly before the session, so the day being explained
    # cannot influence the yardstick used to explain it.
    estimation = aligned[aligned.index < pd.Timestamp(session)]
    if len(estimation) < _MIN_HISTORY_SESSIONS:
        return None, [], None

    variance = float(estimation["market"].var(ddof=1))
    if variance <= 1e-12:
        return None, [], None
    beta = float(estimation.cov().loc["market", "stock"] / variance)

    market_today = _value_on(market_returns, session)
    if not np.isfinite(market_today):
        return None, [], None

    explained = beta * market_today
    specific = observation.period_return_pct - explained
    counterevidence: list[Counterevidence] = []

    if abs(observation.period_return_pct) > 1e-9:
        explained_share = abs(explained / observation.period_return_pct) * 100.0
        if explained_share >= 60.0 and np.sign(explained) == np.sign(
            observation.period_return_pct
        ):
            counterevidence.append(
                Counterevidence(
                    label="Most of this was the whole market, not the company",
                    detail=(
                        f"The S&P 500 moved {market_today:+.2f}% the same day. "
                        f"{observation.display_symbol} typically moves about "
                        f"{beta:.2f}x the market, which accounts for roughly "
                        f"{explained:+.2f} of its {observation.period_return_pct:+.2f}% move "
                        f"({explained_share:.0f}%). Only about {specific:+.2f}% was specific "
                        f"to {observation.display_symbol}. Company-specific explanations "
                        "should be weighed against that."
                    ),
                    related_factor_id="market_move",
                )
            )

    factor = _FactorInput(
        factor_id="market_move",
        label="The overall US stock market (S&P 500)",
        units="%",
        series=market_returns,
        today_value=market_today,
        what_happened=(
            f"The S&P 500 moved {market_today:+.2f}%. {observation.display_symbol} has "
            f"historically moved about {beta:.2f}x the market, which accounts for roughly "
            f"{explained:+.2f} percentage points of its {observation.period_return_pct:+.2f}% "
            f"move; about {specific:+.2f} points were specific to {observation.display_symbol}"
        ),
    )
    note = (
        f"Beta of {beta:.2f} estimated over {len(estimation)} sessions ending before "
        f"{session.isoformat()}."
    )
    return factor, counterevidence, note


def _macro_release_records(
    session: date, start: date, end: date, source: Source
) -> tuple[list[MacroReleaseRecord], list[str], list[str], bool]:
    """Scheduled macro numbers published on or near the session.

    Returns (records, limitations, warnings, is_mock). A macro failure never
    aborts the report -- the price evidence stands on its own -- so problems
    come back as warnings.
    """
    limitations: list[str] = []
    warnings: list[str] = []
    try:
        panel = load_macro_panel(start, end, source=source)
    except (MacroDataError, ValueError) as exc:
        return [], [], [f"macro data unavailable ({exc}); macro factors are omitted"], True

    try:
        releases = releases_on(panel, session, window_days=2)
    except (ValueError, KeyError) as exc:
        return [], [], [f"could not read the macro release calendar ({exc})"], panel.source == "mock"

    records = [
        MacroReleaseRecord(
            series_id=release.series_id,
            label=release.label,
            reference_period=release.reference_period,
            release_date=release.release_date,
            value=release.value,
            units_label=release.units_label,
            change=release.change,
            surprise_z=release.surprise_z,
            description=release.describe(),
        )
        for release in releases
    ]
    if records:
        limitations.append(
            "Macro release dates are derived from each series' typical publication lag, "
            "not from a live economic calendar (no free, terms-compliant calendar API "
            "exists), so they can be off by a few days."
        )
        limitations.append(
            "'Surprise' here means the change versus the recent average change, not versus "
            "a survey consensus -- consensus forecasts are not available on free data."
        )
    if panel.source == "mock":
        warnings.append(
            "macro figures are synthetic (no FRED_API_KEY set) -- set one for real data"
        )
    return records, limitations, warnings, panel.source == "mock"


def _counterevidence_for(
    observation: PriceObservation, factors: list[CandidateFactor]
) -> list[Counterevidence]:
    # NOTE: only external factors are weighed here. A mechanical factor
    # cannot be evidence for or against an explanation, so it can neither
    # support one nor contradict one.
    """Everything that argues against reading too much into this move."""
    items: list[Counterevidence] = []

    if observation.outlier_label == "typical":
        items.append(
            Counterevidence(
                label="This move was not unusual",
                detail=(
                    f"{observation.display_symbol} moved {observation.period_return_pct:+.2f}%, "
                    f"which is {abs(observation.sigma_multiple):.1f} standard deviations against "
                    f"a typical daily swing of {observation.daily_vol_pct:.2f}%. Moves this size "
                    "happen constantly and usually have no specific explanation. Treat any "
                    "story about this day with scepticism."
                ),
            )
        )

    if np.isfinite(observation.volume_ratio) and observation.volume_ratio < 0.8:
        items.append(
            Counterevidence(
                label="Trading volume was below average",
                detail=(
                    f"Volume was {observation.volume_ratio:.2f}x its recent average. Moves on "
                    "light volume reflect fewer participants and are weaker evidence that "
                    "anything material changed."
                ),
            )
        )

    external = [f for f in factors if f.relationship == "external"]
    supported = [f for f in external if f.beats_base_rate]
    scored = [f for f in external if f.historical_hit_rate is not None]
    if scored and not supported:
        best = max(scored, key=lambda f: f.rank_score)
        items.append(
            Counterevidence(
                label="No factor here has a track record better than chance",
                detail=(
                    f"The strongest factor, {best.label}, matched this direction "
                    f"{best.historical_hit_rate:.0%} of the time historically, against a base "
                    f"rate of {best.base_rate:.0%} on all days -- a difference inside the "
                    "margin of error. On this evidence, none of the factors below explains "
                    "the move better than a coin flip."
                ),
                related_factor_id=best.factor_id,
            )
        )

    disagreeing = [
        f for f in external
        if f.historical_hit_rate is not None and f.lift is not None and f.lift < -0.05
    ]
    for factor in disagreeing[:2]:
        items.append(
            Counterevidence(
                label=f"{factor.label} points the other way",
                detail=(
                    f"On days like this one, {factor.label} matched "
                    f"{observation.display_symbol}'s direction only "
                    f"{factor.historical_hit_rate:.0%} of the time, which is below the "
                    f"{factor.base_rate:.0%} base rate -- historically it has leaned against "
                    "moves like today's."
                ),
                related_factor_id=factor.factor_id,
            )
        )
    return items


def build_why_moved_bundle(
    parsed: ParsedQuery,
    source: Source = "auto",
    history_years: float = DEFAULT_HISTORY_YEARS,
    lookback: int = 20,
) -> EvidenceBundle:
    """Build the evidence bundle for an attribution question.

    Parameters
    ----------
    parsed : the structured query from `query_parser.parse_query`.
    source : "auto", "live" or "mock", passed through to the data layer.
    history_years : years of history used for hit rates.
    lookback : sessions used for the volatility baseline.

    Raises
    ------
    PriceDataError : only when the *target symbol* cannot be fetched at all.
        Every other data source is optional; losing one costs a factor, not
        the report.
    """
    if history_years < 1.0:
        raise ValueError(
            f"history_years must be at least 1 to compute any hit rate, got {history_years}"
        )

    observation = build_observation(
        parsed.yf_symbol,
        parsed.start,
        parsed.end,
        display_symbol=parsed.ticker,
        lookback=lookback,
        source=source,
    )
    session = observation.session_date
    history_start = session - timedelta(days=int(history_years * 365.25))
    target_sign = 1 if observation.period_return_pct > 0 else -1

    warnings = list(parsed.warnings) + list(observation.warnings)
    limitations: list[str] = []
    data_sources = [f"Prices: {'Yahoo Finance' if observation.source == 'yfinance' else 'synthetic'}"]

    # History for the target, over the hit-rate window, excluding the day
    # being explained so it cannot enter its own statistics.
    target_history, _, _ = (
        (observation.history, observation.source, ())
        if observation.history.index.min() <= pd.Timestamp(history_start)
        else (None, None, None)
    )
    if target_history is None:
        from ..data.price_ingest import fetch_history

        target_history, _, extra = fetch_history(
            parsed.yf_symbol, history_start, session, source
        )
        warnings.extend(extra)
    target_returns = _daily_returns(target_history)
    target_returns = target_returns[target_returns.index < pd.Timestamp(session)]

    # Which context symbols are worth fetching depends on the asset class.
    is_index = parsed.ticker.upper() in _INDEX_SYMBOLS
    wanted = list(FACTOR_SYMBOLS)
    if not parsed.is_fx:
        wanted += list(SECTOR_ETFS)
        if is_index:
            wanted += list(MEGACAP_WEIGHTS)
        else:
            wanted.append("SPY")
    histories = fetch_many(wanted, history_start, session, source)

    factor_inputs = _cross_asset_factors(histories, session)
    counterevidence: list[Counterevidence] = []

    if not parsed.is_fx:
        breadth = _sector_breadth_factor(histories, session, target_sign, is_index)
        if breadth is not None:
            factor_inputs.append(breadth)

        if is_index:
            megacaps, megacap_notes = _megacap_factors(
                histories, session, observation.period_return_pct
            )
            factor_inputs.extend(megacaps)
            limitations.extend(megacap_notes)
        else:
            market_factor, market_counter, beta_note = _market_factor(
                histories.get("SPY"), target_returns, session, observation
            )
            if market_factor is not None:
                factor_inputs.insert(0, market_factor)
                counterevidence.extend(market_counter)
                if beta_note:
                    limitations.append(beta_note)
    else:
        limitations.append(
            "Sector breadth and single-stock attribution do not apply to a currency pair, "
            "so they are omitted."
        )

    factors = [
        _score_factor(item, target_returns, target_sign, history_start, session)
        for item in factor_inputs
    ]

    macro_records, macro_limitations, macro_warnings, macro_is_mock = _macro_release_records(
        session, history_start, session, source
    )
    limitations.extend(macro_limitations)
    warnings.extend(macro_warnings)
    if macro_records:
        data_sources.append(f"Macro: {'FRED' if not macro_is_mock else 'synthetic'}")

    counterevidence.extend(_counterevidence_for(observation, factors))

    if parsed.direction != "unspecified" and parsed.direction != observation.direction:
        counterevidence.insert(
            0,
            Counterevidence(
                label=f"{parsed.ticker} did not actually go {parsed.direction}",
                detail=(
                    f"The question assumed a move {parsed.direction}, but "
                    f"{parsed.ticker} went {observation.direction} "
                    f"{abs(observation.period_return_pct):.2f}% over this period. The evidence "
                    "below describes the move that actually happened."
                ),
            ),
        )

    if any(f.relationship == "mechanical" for f in factors):
        limitations.append(
            "Factors marked as part of the index (sector breadth, individual members) "
            "describe what the move was made of. Their high hit rates are arithmetic, "
            "not evidence about why the move happened."
        )
    limitations.append(
        "Hit rates describe how often a factor and a move have coincided in the past. "
        "They are not evidence that one causes the other, and past coincidence need not "
        "continue."
    )
    if len(target_returns) < _MIN_HISTORY_SESSIONS:
        limitations.append(
            f"Only {len(target_returns)} sessions of history were available (about "
            f"{len(target_returns) / 252:.1f} years), so hit rates are less reliable than usual."
        )

    return EvidenceBundle(
        query=parsed.raw_query,
        question_type="attribution",
        ticker=parsed.ticker,
        period_description=parsed.describe_period(),
        observation=Observation(
            symbol=observation.symbol,
            display_symbol=observation.display_symbol,
            session_date=observation.session_date,
            requested_date=observation.requested_date,
            close=observation.close,
            prior_close=observation.prior_close,
            return_pct=observation.return_pct,
            period_return_pct=observation.period_return_pct,
            daily_vol_pct=observation.daily_vol_pct,
            annualized_vol_pct=observation.annualized_vol_pct,
            sigma_multiple=observation.sigma_multiple,
            outlier_label=observation.outlier_label,
            volume_ratio=(
                float(observation.volume_ratio)
                if np.isfinite(observation.volume_ratio)
                else None
            ),
            lookback_sessions=len(observation.context),
            plain_summary=observation.describe_move(),
        ),
        factors=factors,
        counterevidence=counterevidence,
        macro_releases=macro_records,
        data_sources=data_sources,
        limitations=limitations,
        warnings=warnings,
        history_years=history_years,
        is_synthetic=observation.source == "mock",
    )
