"""Typed results for the "what if" scenarios, carried on the EvidenceBundle.

Every number here is a description of the past: how often a kind of market
showed up, and what the asked-about symbol did in the days after it showed
up. None of them is a forecast, and the field names say so -- there is no
"probability" here, only frequencies and outcomes with their sample sizes.

Confidence comes from the number of *separate episodes*, not the number of
days. Trigger days cluster: one oil spike can supply sixty consecutive days,
and those sixty forward returns overlap almost entirely. Sixty days from one
episode is one piece of evidence, so the label is tied to how many distinct
times the situation arose.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field

__all__ = [
    "EpisodeOutcome",
    "HorizonOutcome",
    "ScenarioBaseline",
    "ScenarioConfidence",
    "ScenarioResult",
    "confidence_for_episodes",
]

ScenarioConfidence = Literal["none", "low", "moderate", "high"]

#: Distinct episodes needed for each label. Two is the floor for quoting any
#: outcome at all: a single episode is an anecdote, which the report shows as
#: a named example rather than as a statistic.
_EPISODE_THRESHOLDS: tuple[tuple[int, ScenarioConfidence], ...] = (
    (10, "high"),
    (4, "moderate"),
    (2, "low"),
)
MIN_EPISODES = 2


def confidence_for_episodes(episodes: int) -> ScenarioConfidence:
    """Map a count of separate episodes to a confidence label."""
    for threshold, label in _EPISODE_THRESHOLDS:
        if episodes >= threshold:
            return label
    return "none"


class HorizonOutcome(BaseModel):
    """What the symbol did over one horizon after the condition held.

    All returns in percent. `share_lower_pct` is the share of cases in which
    the price ended the horizon lower than it started -- the plainest honest
    answer to "how often did it go down", and still only a frequency.
    `share_higher_pct` is measured separately rather than derived, so flat
    cases count as neither.
    """

    model_config = ConfigDict(frozen=True)

    horizon_days: int
    median_pct: float | None = None
    p10_pct: float | None = None
    p90_pct: float | None = None
    worst_pct: float | None = None
    share_lower_pct: float | None = None
    share_higher_pct: float | None = None
    n_days: int = 0


class EpisodeOutcome(BaseModel):
    """One named historical episode, and the symbol's return across it.

    `symbol_return_pct` is None, never zero, when the symbol has no prices
    that far back -- "not listed yet" and "did not move" are different facts.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    start: date
    end: date
    symbol_return_pct: float | None = None
    note: str | None = None


class ScenarioResult(BaseModel):
    """One "what if" scenario, measured against the symbol's own history.

    `quantified` is False when the scenario has no measurable trigger. Those
    carry no frequency and no outcome, only `watch_for`, because an honest
    "this cannot be measured" is worth more than an invented number.
    """

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    label: str
    plain_label: str
    description: str
    quantified: bool
    trigger_description: str | None = None
    proxy_symbol: str | None = None
    frequency_pct: float | None = None
    months_observed: int = 0
    months_triggered: int = 0
    episodes_detected: int = 0
    outcomes: tuple[HorizonOutcome, ...] = ()
    examples: tuple[EpisodeOutcome, ...] = ()
    watch_for: tuple[str, ...] = ()
    note: str | None = None
    proxy_is_synthetic: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> ScenarioConfidence:
        """Derived from separate episodes alone -- never set by the caller."""
        return confidence_for_episodes(self.episodes_detected)

    def outcome(self, horizon_days: int) -> HorizonOutcome | None:
        """The outcome at one horizon, or None when it was not measured."""
        return next((o for o in self.outcomes if o.horizon_days == horizon_days), None)


class ScenarioBaseline(BaseModel):
    """The same horizons over *every* day -- the base rate for scenarios.

    A scenario's outcome means nothing alone. "Lower 45% of the time after an
    oil spike" is only interesting beside "lower 44% of the time on any day".
    """

    model_config = ConfigDict(frozen=True)

    history_start: date
    history_end: date
    sessions: int
    outcomes: tuple[HorizonOutcome, ...] = ()

    def outcome(self, horizon_days: int) -> HorizonOutcome | None:
        return next((o for o in self.outcomes if o.horizon_days == horizon_days), None)
