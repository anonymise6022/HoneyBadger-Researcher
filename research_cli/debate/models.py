"""Structured debate results, kept separate from any rendering.

Everything an agent said that survives into a `DebateResult` has already
passed the checks in `debate.checks`: valid evidence ids, numbers that trace
to those ids, and none of the banned phrasing. The renderer can therefore
print these fields without re-deciding whether they are allowed -- though the
final report is validated again as a whole anyway.

Scores here are rubric scores for ranking arguments. They are **not**
probabilities and are never shown as if they were. The only likelihood
figures the debate report shows come from code, in `historical_support_pct`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "AgentCase",
    "Challenge",
    "Claim",
    "DebateOutcome",
    "DebateResult",
    "EvidenceItem",
    "ProposedScenario",
    "Rebuttal",
    "RubricScore",
    "Stance",
]

Stance = Literal["bull", "bear", "neutral"]
DebateOutcome = Literal["clear", "several", "fallback"]


@dataclass(frozen=True)
class EvidenceItem:
    """One fact an agent may cite, written from bundle fields.

    `item_id` is short ("V1", "S3") so a model can cite it reliably; `path`
    names the bundle field it came from, so a reader can trace it back.
    `untrusted` marks quoted third-party text (headlines), which prompts
    present as data, never as instructions.
    """

    item_id: str
    group: str
    text: str
    path: str
    untrusted: bool = False


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProposedScenario:
    """An extra "what if" an agent suggested, mapped onto measured data or not."""

    name: str
    proposed_by: str
    mapped_to: str | None = None     # a library scenario id, or None if unquantified


@dataclass(frozen=True)
class Challenge:
    by: str
    target_persona: str
    target_claim_id: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Rebuttal:
    by: str
    text: str
    evidence_ids: tuple[str, ...]
    concedes: bool = False


@dataclass(frozen=True)
class RubricScore:
    """The mostly deterministic score used to rank and eliminate cases."""

    evidence: float
    falsifiers: float
    counter: float
    scenario_coverage: float
    judge: float | None = None

    @property
    def deterministic(self) -> float:
        return (
            0.35 * self.evidence + 0.15 * self.falsifiers
            + 0.25 * self.counter + 0.25 * self.scenario_coverage
        )

    @property
    def total(self) -> float:
        """Deterministic part, blended with the judge's small share if present."""
        if self.judge is None:
            return self.deterministic
        return 0.8 * self.deterministic + 0.2 * self.judge


@dataclass
class AgentCase:
    """One persona's argued case and everything that happened to it."""

    persona: str
    persona_label: str
    stance: Stance
    headline: str
    claims: list[Claim]
    falsifiers: list[str]
    confidence_words: str
    dropped_claims: int = 0
    challenges_received: list[Challenge] = field(default_factory=list)
    rebuttal: Rebuttal | None = None
    score: RubricScore | None = None
    eliminated_in_round: int | None = None
    plain_summary: str | None = None
    historical_support_pct: float | None = None

    def cited_ids(self) -> set[str]:
        ids = {i for claim in self.claims for i in claim.evidence_ids}
        if self.rebuttal is not None:
            ids.update(self.rebuttal.evidence_ids)
        return ids


@dataclass
class DebateResult:
    """The whole debate, ready for a renderer.

    `outcome` is "clear" when the top case leads by the stated margin,
    "several" when it does not (the top 2-3 are shown as equals), and
    "fallback" when too few checked arguments survived for a debate at all.
    """

    outcome: DebateOutcome
    cases: list[AgentCase]
    leaders: list[AgentCase]
    evidence: dict[str, EvidenceItem]
    proposed_scenarios: list[ProposedScenario] = field(default_factory=list)
    rounds_completed: int = 0
    notes: list[str] = field(default_factory=list)
    calls_used: int = 0
    calls_limit: int = 0

    @property
    def strongest(self) -> AgentCase | None:
        return self.leaders[0] if self.outcome == "clear" and self.leaders else None
