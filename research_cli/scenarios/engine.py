"""Measure every library scenario against one symbol's own history.

For each scenario with a trigger:

* **How often it happens** -- the share of months, over the history window,
  in which the trigger held on at least one day. Months rather than days,
  because "about one month in ten" is a frequency a beginner can picture.
* **What the symbol did next** -- forward returns at 5, 21 and 63 sessions
  from every day the trigger held: median, 10th-90th percentile, worst case,
  and the share of cases that ended lower. Below two separate episodes no
  outcome is quoted at all.
* **Named examples** -- the symbol's return across each hand-picked episode,
  or None when it has no prices that far back.

The same horizons over every day form the baseline, so each scenario can be
read against "what usually happens anyway".

**Synthetic data is labelled per series.** Each proxy is fetched on its own
and its source recorded. If the symbol is live but a proxy fell back to
synthetic data, the scenarios built on that proxy are flagged and the pack
carries a warning, so real and invented figures are never mixed silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from ..data.price_ingest import PriceDataError, Source, UnknownSymbolError, fetch_history
from ..evidence.scenario_schema import (
    MIN_EPISODES,
    EpisodeOutcome,
    ScenarioBaseline,
    ScenarioResult,
)
from .library import PROXY_SYMBOLS, SCENARIOS, ScenarioSpec
from .measure import (
    HORIZONS,
    count_episodes,
    episode_return,
    forward_returns,
    horizon_outcome,
    monthly_frequency,
    trigger_series,
)

__all__ = ["DEFAULT_SCENARIO_YEARS", "ScenarioPack", "build_scenarios"]

DEFAULT_SCENARIO_YEARS = 20.0
_MAIN_HORIZON = 21


@dataclass
class ScenarioPack:
    """Everything the scenario layer adds to a bundle."""

    results: tuple[ScenarioResult, ...]
    baseline: ScenarioBaseline | None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    synthetic: bool = False


def _load_proxies(
    start: date, end: date, source: Source, symbol_source: str
) -> tuple[dict[str, pd.Series], set[str], list[str]]:
    """Fetch each proxy's closes; return (series, synthetic proxy ids, warnings)."""
    series: dict[str, pd.Series] = {}
    synthetic: set[str] = set()
    warnings: list[str] = []
    for proxy_id, symbol in PROXY_SYMBOLS.items():
        try:
            frame, used, _ = fetch_history(symbol, start, end, source)
        except (PriceDataError, UnknownSymbolError):
            continue
        closes = frame["Close"].astype(float)
        if proxy_id == "rates" and closes.median() > 20:
            # ^TNX has been quoted both as the yield and as ten times it.
            closes = closes / 10.0
        series[proxy_id] = closes
        if used == "mock":
            synthetic.add(proxy_id)
            if symbol_source != "mock":
                warnings.append(
                    f"scenario data for {symbol} is synthetic (live prices were "
                    "unavailable); scenarios built on it are marked as such"
                )
    return series, synthetic, warnings


def _measure_one(
    spec: ScenarioSpec,
    closes: pd.Series,
    forwards: dict[int, pd.Series],
    proxies: dict[str, pd.Series],
    synthetic: set[str],
    is_fund: bool,
) -> ScenarioResult:
    examples = tuple(
        EpisodeOutcome(
            label=episode.label, start=episode.start, end=episode.end,
            symbol_return_pct=(value := episode_return(closes, episode.start, episode.end)),
            note=None if value is not None else "no price history this far back",
        )
        for episode in spec.episodes
        if episode.end <= closes.index.max().date()
    )
    base = dict(
        scenario_id=spec.scenario_id, label=spec.label, plain_label=spec.plain_label,
        description=spec.description, examples=examples, watch_for=spec.watch_for,
    )
    trigger = spec.trigger
    if trigger is None:
        return ScenarioResult(
            **base, quantified=False, note="price data cannot say how often this happens",
        )
    if is_fund and not spec.applies_to_funds:
        return ScenarioResult(
            **base, quantified=False, trigger_description=trigger.description,
            note="does not apply to a fund, which holds many companies at once",
        )
    if trigger.proxy == "self":
        driver, proxy_symbol = closes, None
    else:
        driver, proxy_symbol = proxies.get(trigger.proxy), PROXY_SYMBOLS[trigger.proxy]
    if driver is None:
        return ScenarioResult(
            **base, quantified=False, trigger_description=trigger.description,
            proxy_symbol=proxy_symbol, note="the data needed to measure this was unavailable",
        )

    flags = trigger_series(
        trigger.kind, driver, trigger.window, trigger.threshold, market=proxies.get("market")
    )
    months_observed, months_triggered = monthly_frequency(flags)
    aligned = flags.reindex(closes.index, method="ffill", limit=3) if driver is not closes else flags
    mask = (aligned >= 1.0) & forwards[_MAIN_HORIZON].notna()
    episodes = count_episodes(mask)
    outcomes = (
        tuple(horizon_outcome(forwards[h][mask], h) for h in HORIZONS)
        if episodes >= MIN_EPISODES else ()
    )
    note = None
    if episodes < MIN_EPISODES:
        note = (
            f"only {episodes} separate episode(s) while this symbol was trading -- "
            "too few to describe what usually followed"
        )
    return ScenarioResult(
        **base, quantified=True, trigger_description=trigger.description,
        proxy_symbol=proxy_symbol,
        frequency_pct=(months_triggered / months_observed * 100.0) if months_observed else None,
        months_observed=months_observed, months_triggered=months_triggered,
        episodes_detected=episodes, outcomes=outcomes, note=note,
        proxy_is_synthetic=trigger.proxy in synthetic,
    )


def build_scenarios(
    symbol: str,
    display_symbol: str,
    as_of: date,
    source: Source = "auto",
    years: float = DEFAULT_SCENARIO_YEARS,
    is_fund: bool = False,
) -> ScenarioPack:
    """Measure the whole library for one symbol, history ending at `as_of`.

    Raises PriceDataError / UnknownSymbolError only for the symbol itself;
    a missing proxy costs its scenarios, not the pack.
    """
    if years < 2:
        raise ValueError(f"years must be at least 2 to see any scenario repeat, got {years}")
    start = as_of - timedelta(days=int(years * 365.25))
    frame, symbol_source, fetch_warnings = fetch_history(symbol, start, as_of, source)
    closes = frame["Close"].astype(float)
    closes = closes[closes.index <= pd.Timestamp(as_of)]
    proxies, synthetic, warnings = _load_proxies(start, as_of, source, symbol_source)
    # Flag a proxy only when it is synthetic beside real symbol data. When
    # everything is synthetic the whole report already says so at the top.
    mixed = synthetic if symbol_source != "mock" else set()
    forwards = {h: forward_returns(closes, h) for h in HORIZONS}

    results = tuple(
        _measure_one(spec, closes, forwards, proxies, mixed, is_fund) for spec in SCENARIOS
    )
    baseline = ScenarioBaseline(
        history_start=closes.index.min().date(),
        history_end=closes.index.max().date(),
        sessions=int(len(closes)),
        outcomes=tuple(horizon_outcome(forwards[h], h) for h in HORIZONS),
    )

    covered_years = (baseline.history_end - baseline.history_start).days / 365.25
    limitations = [
        f"Scenario figures describe what {display_symbol} did on past days when each "
        f"condition held, over about {covered_years:.0f} years of its history. They count "
        "how often things happened before; they are not the chance of anything happening next.",
        "Each condition uses a round, stated threshold written into its label. Example "
        "episodes are hand-picked illustrations; the statistics come only from the stated "
        "conditions.",
        "Scenarios without a measurable condition get no number at all, only what to watch for.",
    ]
    if covered_years < years - 1:
        limitations.append(
            f"{display_symbol} has only about {covered_years:.0f} years of price history here, "
            "so situations from before then cannot be measured for it."
        )
    is_synthetic = symbol_source == "mock" or bool(synthetic)
    sources = [
        "Scenario history: "
        + ("synthetic" if symbol_source == "mock" else "Yahoo Finance")
        + f" ({', '.join(sorted(PROXY_SYMBOLS.values()))} as condition proxies)"
    ]
    return ScenarioPack(
        results=results,
        baseline=baseline,
        warnings=list(fetch_warnings) + warnings,
        limitations=limitations,
        data_sources=sources,
        synthetic=is_synthetic,
    )
