"""The EvidenceBundle: the only thing a report is allowed to talk about.

Every number that reaches a user passes through this structure first. That
is the central architectural constraint of the tool: the synthesis layer --
whether it is a string template or a language model -- receives a bundle and
nothing else, and `synthesis.validator` afterwards checks that every figure
in the prose traces back to a field here. A number that is not in a bundle
cannot legitimately appear in a report.

The types encode the honesty rules rather than leaving them to prose:

* A `CandidateFactor` cannot report a `historical_hit_rate` without a
  `sample_size`, and its `confidence` is *derived* from that sample size by
  a validator, not set by whoever built it.
* `lift` -- hit rate minus base rate -- is computed here. A factor that
  "predicted" the move 62% of the time when the market went that way 60% of
  the time anyway has told you nothing, and `beats_base_rate` says so.
* Every bundle carries `limitations`, and the builders are expected to
  populate it. An empty limitations list on a real bundle is a bug.

On causal language: nothing in this schema is named "cause", "driver" or
"reason", and that is deliberate. These are things that *coincided* with a
move, ranked by how often they have coincided with similar moves before.
The vocabulary is enforced downstream in `synthesis`; keeping it out of the
field names removes the temptation at the source.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

__all__ = [
    "CandidateFactor",
    "Confidence",
    "Counterevidence",
    "EvidenceBundle",
    "MacroReleaseRecord",
    "NewsItem",
    "Observation",
    "PeerComparison",
    "PriceTrend",
    "QuestionType",
    "Relationship",
    "SnapshotFundamentals",
    "confidence_for_sample",
]

QuestionType = Literal["attribution", "snapshot"]
Confidence = Literal["none", "low", "moderate", "high"]

#: Whether a factor is *part of* the thing it is being compared against.
#: This distinction is load-bearing. Sector breadth "predicts" an index move
#: with a huge hit rate because the index is made of those sectors -- that is
#: an identity, not a finding, and a reader who is not told so will badly
#: over-read it. Mechanical factors still belong in the report: knowing a
#: move was broad rather than concentrated in one name is genuinely useful.
#: They are simply not evidence about *why*, and are presented apart from
#: the factors that are external to the symbol.
Relationship = Literal["mechanical", "external"]

#: Sample-size thresholds for the confidence label. Chosen so that the
#: binomial standard error of a hit rate near 50% is roughly: >15pp below 10
#: observations (useless), ~9pp at 30 (suggestive), ~5pp at 100 (usable).
#: These are the widths that decide whether a hit-rate difference of a few
#: points means anything, so the labels are tied to them rather than picked.
_CONFIDENCE_THRESHOLDS: tuple[tuple[int, Confidence], ...] = (
    (100, "high"),
    (30, "moderate"),
    (10, "low"),
)

#: Multipliers used when ranking factors. Ordinal, not physical: they encode
#: "a high-confidence 60% hit rate outranks a low-confidence 80% one".
CONFIDENCE_WEIGHT: dict[Confidence, float] = {
    "none": 0.0, "low": 0.4, "moderate": 0.75, "high": 1.0,
}


def confidence_for_sample(sample_size: int) -> Confidence:
    """Map a sample size to a confidence label.

    The single place this mapping exists, so a factor cannot be constructed
    with a confidence its sample size does not support.
    """
    if sample_size < 0:
        raise ValueError(f"sample_size cannot be negative, got {sample_size}")
    for threshold, label in _CONFIDENCE_THRESHOLDS:
        if sample_size >= threshold:
            return label
    return "none"


class Observation(BaseModel):
    """The primary fact: what the symbol actually did, and how unusual it was.

    This is the one section of a report that states something without
    hedging, because it is a measurement rather than an inference.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    display_symbol: str
    session_date: date
    requested_date: date
    close: float
    prior_close: float
    return_pct: float
    period_return_pct: float
    daily_vol_pct: float = Field(ge=0.0)
    annualized_vol_pct: float = Field(ge=0.0)
    sigma_multiple: float
    outlier_label: Literal["typical", "1-sigma", "2-sigma", "3-sigma+"]
    volume_ratio: float | None = None
    lookback_sessions: int = Field(gt=0)
    plain_summary: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def direction(self) -> Literal["up", "down", "flat"]:
        if abs(self.period_return_pct) < 0.005:
            return "flat"
        return "up" if self.period_return_pct > 0 else "down"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_unusual(self) -> bool:
        """Whether the move warrants looking for an explanation at all."""
        return abs(self.sigma_multiple) >= 2.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def session_differs_from_request(self) -> bool:
        return self.session_date != self.requested_date


class CandidateFactor(BaseModel):
    """Something that happened at the same time, with its historical record.

    A factor is *not* a cause and is not described as one. It is a condition
    that held on the day in question, together with how often the symbol has
    moved this way on the other days that condition held.

    Attributes
    ----------
    factor_id : stable machine name, e.g. "vix_change".
    label : human name, e.g. "Volatility index (VIX)".
    what_happened : one sentence stating the observed value in plain words.
    observed_value, observed_units : the measurement itself.
    condition_description : the historical condition the hit rate is
        computed over, e.g. "days when the VIX rose the most (top 20%)".
    historical_hit_rate : of the historical days meeting that condition, the
        fraction on which the symbol moved in the same direction as today.
        None when the sample is too small to report one.
    base_rate : the same fraction over *all* historical days -- what you
        would expect knowing nothing.
    sample_size : days meeting the condition.
    history_start, history_end : the window the rates were computed over.
    direction_agrees : whether this factor points the same way as the move.
    """

    model_config = ConfigDict(frozen=True)

    factor_id: str
    label: str
    what_happened: str
    observed_value: float
    observed_units: str
    condition_description: str
    historical_hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    base_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_size: int = Field(default=0, ge=0)
    history_start: date | None = None
    history_end: date | None = None
    direction_agrees: bool = True
    relationship: Relationship = "external"
    note: str | None = None

    @model_validator(mode="after")
    def _confidence_follows_sample_size(self) -> CandidateFactor:
        """A hit rate with too small a sample is discarded, not labelled.

        Reporting "80% hit rate (4 observations)" invites a reader to
        believe the 80% and skim the 4. Below the threshold the rate is
        removed entirely and a note explains why.
        """
        if self.historical_hit_rate is not None and self.sample_size < 10:
            object.__setattr__(self, "historical_hit_rate", None)
            object.__setattr__(
                self,
                "note",
                self.note
                or (
                    f"only {self.sample_size} comparable days in the history window -- "
                    "too few to quote a hit rate"
                ),
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> Confidence:
        """Derived from sample size alone -- never set by the caller."""
        return confidence_for_sample(self.sample_size)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lift(self) -> float | None:
        """Hit rate minus base rate: what conditioning on this factor buys.

        The number that actually matters. A 62% hit rate against a 60% base
        rate is a lift of 2 points, which is noise on any realistic sample.
        """
        if self.historical_hit_rate is None or self.base_rate is None:
            return None
        return self.historical_hit_rate - self.base_rate

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hit_rate_standard_error(self) -> float | None:
        """Binomial standard error of the hit rate, for judging the lift."""
        if self.historical_hit_rate is None or self.sample_size < 1:
            return None
        p = self.historical_hit_rate
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / self.sample_size))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def beats_base_rate(self) -> bool:
        """True when the lift exceeds two standard errors of the hit rate.

        The honest bar for "this factor told you something". Most factors on
        most days do not clear it, and a report that says so is more useful
        than one that ranks ten coin flips.
        """
        lift, se = self.lift, self.hit_rate_standard_error
        if lift is None or se is None or se <= 0.0:
            return False
        return abs(lift) > 2.0 * se

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rank_score(self) -> float:
        """Ordering key: hit rate scaled by confidence in it.

        Factors are ranked by evidential strength, never by how interesting
        they sound. A factor with no usable hit rate scores zero and sinks
        to the bottom regardless of how dramatic its observed value is.
        """
        if self.historical_hit_rate is None:
            return 0.0
        return float(self.historical_hit_rate * CONFIDENCE_WEIGHT[self.confidence])


class Counterevidence(BaseModel):
    """Something that argues against the leading explanation.

    Surfaced as its own section rather than buried, because the failure mode
    of a research tool is a confident story, and the cheapest guard against
    it is a standing requirement to publish what does not fit.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    detail: str
    related_factor_id: str | None = None


class MacroReleaseRecord(BaseModel):
    """A scheduled macro number published on or near the day in question."""

    model_config = ConfigDict(frozen=True)

    series_id: str
    label: str
    reference_period: date
    release_date: date
    value: float
    units_label: str
    change: float | None = None
    surprise_z: float | None = None
    description: str
    release_date_is_estimated: bool = True
    consensus_is_modelled: bool = True


class PeerComparison(BaseModel):
    """One peer's metric beside the subject's, for the snapshot report."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    label: str
    metric: str
    value: float | None
    subject_value: float | None = None
    peer_median: float | None = None
    peer_values: dict[str, float] = Field(default_factory=dict)
    higher_is_better: bool | None = None
    note: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def vs_median_pct(self) -> float | None:
        """The subject's distance from the peer median, in percent.

        None when either side is missing or the median is ~0, rather than a
        divide-by-zero or a misleadingly huge ratio.
        """
        if self.subject_value is None or self.peer_median is None:
            return None
        if abs(self.peer_median) < 1e-9:
            return None
        return float((self.subject_value - self.peer_median) / abs(self.peer_median) * 100.0)


class SnapshotFundamentals(BaseModel):
    """Fundamentals for the "is X good to invest" question.

    Every field is optional because real data is full of holes: a
    loss-making company has no meaningful P/E, a non-payer has no dividend
    yield, and an ETF has almost none of these. A missing field is reported
    as missing, never filled with a zero that would read as a fact.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    company_name: str | None = None
    sector: str | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    revenue_growth: float | None = None
    profit_margin: float | None = None
    debt_to_equity: float | None = None
    return_on_equity: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    missing_fields: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_profitable(self) -> bool | None:
        """None when unknown -- distinct from False, which means loss-making."""
        if self.profit_margin is None:
            return None
        return self.profit_margin > 0.0


class PriceTrend(BaseModel):
    """How a symbol has behaved over the medium term, for the snapshot report.

    Separate from `Observation`, which describes one session. A person asking
    whether something is worth owning needs the shape of the last year, not
    yesterday's tick.
    """

    model_config = ConfigDict(frozen=True)

    current_price: float
    return_1m_pct: float | None = None
    return_3m_pct: float | None = None
    return_1y_pct: float | None = None
    annualized_vol_pct: float = Field(ge=0.0)
    high_52w: float | None = None
    low_52w: float | None = None
    drawdown_from_high_pct: float | None = None
    sessions_available: int = Field(default=0, ge=0)
    summary: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_more_volatile_than_market(self) -> bool:
        """Above ~18% annualized is more volatile than a broad index fund.

        A rough yardstick, stated as one: the S&P's long-run annualized
        volatility sits in the mid-teens, so this flags "bumpier than the
        market" rather than any precise comparison.
        """
        return self.annualized_vol_pct > 18.0


class NewsItem(BaseModel):
    """One recent headline. Title and source only -- never a summary.

    A summary would be prose this tool has not verified, and the report's
    whole claim is that everything in it traces to a checked field.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    publisher: str
    published: date | None = None
    url: str | None = None


class EvidenceBundle(BaseModel):
    """Everything a report may reference, and nothing it may not.

    `numeric_fields()` is the contract the validator enforces: it flattens
    every number in the bundle into one mapping, and any figure in a report
    that does not appear there is flagged as unsupported.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    question_type: QuestionType
    ticker: str
    period_description: str
    observation: Observation | None = None
    factors: list[CandidateFactor] = Field(default_factory=list)
    counterevidence: list[Counterevidence] = Field(default_factory=list)
    macro_releases: list[MacroReleaseRecord] = Field(default_factory=list)
    fundamentals: SnapshotFundamentals | None = None
    peers: list[PeerComparison] = Field(default_factory=list)
    trend: PriceTrend | None = None
    news: list[NewsItem] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    history_years: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.now)
    is_synthetic: bool = False

    def ranked_factors(self) -> list[CandidateFactor]:
        """Factors strongest-evidence-first.

        Sorted by `rank_score`, then by absolute lift, then by sample size.
        Deterministic and independent of insertion order, so the same bundle
        always produces the same report.
        """
        return sorted(
            self.factors,
            key=lambda f: (f.rank_score, abs(f.lift or 0.0), f.sample_size),
            reverse=True,
        )

    def supported_factors(self) -> list[CandidateFactor]:
        """External factors whose lift clears two standard errors.

        Mechanical factors are excluded no matter how high their hit rate:
        an index moving with its own sectors is an identity, and letting an
        identity top the evidence ranking would be the single most
        misleading thing this tool could do.
        """
        return [
            f for f in self.ranked_factors()
            if f.beats_base_rate and f.relationship == "external"
        ]

    def external_factors(self) -> list[CandidateFactor]:
        """Factors not mechanically part of the symbol, strongest first."""
        return [f for f in self.ranked_factors() if f.relationship == "external"]

    def mechanical_factors(self) -> list[CandidateFactor]:
        """Factors that describe what the move was made of, strongest first."""
        return [f for f in self.ranked_factors() if f.relationship == "mechanical"]

    def numeric_fields(self) -> dict[str, float]:
        """Every number in the bundle, flattened to dotted paths.

        The validator's whole basis. Booleans are excluded -- they are not
        figures a report quotes -- and so are dates, which are checked
        separately as strings.
        """
        numbers: dict[str, float] = {}

        def walk(value: object, path: str) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, (int, float)):
                numbers[path] = float(value)
            elif isinstance(value, BaseModel):
                for name in type(value).model_fields:
                    walk(getattr(value, name), f"{path}.{name}" if path else name)
                for name in type(value).model_computed_fields:
                    walk(getattr(value, name), f"{path}.{name}" if path else name)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")

        walk(self, "")
        return numbers
