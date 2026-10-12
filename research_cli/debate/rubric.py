"""Scoring cases, eliminating the weakest, and deciding whether anyone won.

The score is a rubric of what can be counted, not a judgement of who is
right about the future:

    evidence           0.35  share of claims that survived the checks,
                             scaled by how many there are (up to three)
    falsifiers         0.15  says what would prove it wrong (two is full marks)
    counter            0.25  answered the challenge it received
    scenario coverage  0.25  engages the "what if" facts (two cited is full)

A judge model adds a small share (20%) at the very end, among the finalists
only. Scores rank arguments; they are not probabilities and are never shown
as one.

**"Clear" needs a margin.** A case is called the strongest only when it
leads the next by `CLEAR_MARGIN`. Otherwise the report shows the top two or
three as several strong views and says plainly that none won -- forcing a
winner out of a near tie would be the debate equivalent of ranking noise.

The only likelihood figure attached to a case comes from code:
`historical_support_pct` is how often history agreed with that stance's
direction over 21 trading days, from the bundle's every-day baseline.
"""

from __future__ import annotations

from ..evidence.schema import EvidenceBundle
from .models import AgentCase, DebateOutcome, RubricScore

__all__ = [
    "CLEAR_MARGIN",
    "decide",
    "eliminate_weakest",
    "historical_support_pct",
    "rank",
    "score_case",
]

CLEAR_MARGIN = 0.12
_PERSONA_ORDER = ("value", "momentum", "macro", "skeptic")


def score_case(case: AgentCase, scenario_ids: set[str], rebuttals_done: bool) -> RubricScore:
    """Deterministic rubric for one case at its current state."""
    valid = len(case.claims)
    total = valid + case.dropped_claims
    evidence = (valid / total) * min(valid, 3) / 3 if total else 0.0
    falsifiers = min(len(case.falsifiers), 2) / 2

    if not rebuttals_done or not case.challenges_received:
        counter = 0.5
    elif case.rebuttal is None:
        counter = 0.0
    elif case.rebuttal.concedes:
        counter = 0.25
    else:
        counter = 1.0

    if scenario_ids:
        coverage = min(len(case.cited_ids() & scenario_ids), 2) / 2
    else:
        coverage = 0.5
    judge = case.score.judge if case.score is not None else None
    return RubricScore(evidence, falsifiers, counter, coverage, judge)


def rank(cases: list[AgentCase]) -> list[AgentCase]:
    """Highest total first; ties broken by a fixed persona order."""
    def key(case: AgentCase) -> tuple[float, int]:
        order = _PERSONA_ORDER.index(case.persona) if case.persona in _PERSONA_ORDER else 99
        return (-(case.score.total if case.score else 0.0), order)

    return sorted(cases, key=key)


def eliminate_weakest(cases: list[AgentCase], round_number: int, keep_at_least: int = 2) -> None:
    """Mark the lowest-ranked surviving case as eliminated in this round."""
    survivors = [c for c in cases if c.eliminated_in_round is None]
    if len(survivors) > keep_at_least:
        rank(survivors)[-1].eliminated_in_round = round_number


def decide(cases: list[AgentCase]) -> tuple[DebateOutcome, list[AgentCase]]:
    """Return ("clear", [winner]) or ("several", top 2-3), from all cases.

    Finalists are ranked by total. A case eliminated in the last round can
    still join "several strong views" if its deterministic score sits within
    the margin of the leader's -- elimination keeps the debate bounded, it
    does not prove a case weak.
    """
    finalists = rank([c for c in cases if c.eliminated_in_round is None])
    if len(finalists) < 2:
        return "fallback", finalists
    lead = finalists[0].score.total - finalists[1].score.total  # type: ignore[union-attr]
    if lead >= CLEAR_MARGIN:
        return "clear", [finalists[0]]
    top = finalists[0].score.deterministic  # type: ignore[union-attr]
    close = [c for c in finalists if top - c.score.deterministic < CLEAR_MARGIN]  # type: ignore[union-attr]
    late = rank([c for c in cases if c.eliminated_in_round is not None])
    for case in late:
        if len(close) < 3 and top - case.score.deterministic < CLEAR_MARGIN:  # type: ignore[union-attr]
            close.append(case)
    return "several", close[:3] if len(close) >= 2 else finalists[:2]


def historical_support_pct(stance: str, bundle: EvidenceBundle, horizon: int = 21) -> float | None:
    """How often history agreed with a stance's direction, from the baseline.

    bull -> share of past `horizon`-day stretches that ended higher;
    bear -> share that ended lower; neutral has no direction to check.
    """
    baseline = bundle.scenario_baseline
    outcome = baseline.outcome(horizon) if baseline else None
    if outcome is None or stance == "neutral":
        return None
    return outcome.share_lower_pct if stance == "bear" else outcome.share_higher_pct
