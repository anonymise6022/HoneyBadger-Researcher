"""Run the bounded debate: open, cross-examine, rebut, judge.

    Round 1  four openings, in parallel                    4 calls
             score; drop the weakest                        (4 -> 3)
    Round 2  each survivor challenges the strongest        3 calls
             opposing claim, in parallel
    Round 3  each challenged survivor answers, in parallel up to 3 calls
             score; drop the weakest                        (3 -> 2)
    Judge    one call: a small quality score and a plain    1 call
             summary for each finalist
                                                           ---------
                                                           11, leaving 1
                                                           for a repair

Any agent answer that fails its checks gets one repair attempt with the
problems quoted back; after that its valid parts are kept and the rest
dropped. When the call limit is reached the debate stops where it is and is
decided on what it has -- and if fewer than two checked cases exist at all,
the outcome is "fallback" and the caller shows the deterministic report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ..evidence.schema import EvidenceBundle
from ..synthesis.llm_budget import BudgetExhausted, run_parallel
from ..synthesis.llm_provider import LLMProvider
from ..synthesis.llm_report import LLMUnavailable
from .checks import parse_challenge, parse_judge, parse_opening, parse_rebuttal
from .evidence_ids import build_evidence, slice_for
from .models import AgentCase, DebateResult, ProposedScenario
from .personas import (
    AGENT_SYSTEM,
    PERSONAS,
    Persona,
    challenge_prompt,
    judge_prompt,
    opening_prompt,
    rebuttal_prompt,
    repair_prompt,
)
from .rubric import decide, eliminate_weakest, historical_support_pct, rank, score_case

__all__ = ["run_debate"]

_BY_ID = {p.persona_id: p for p in PERSONAS}


def _ask_checked(provider: LLMProvider, prompt: str, parse: Callable[[str], tuple]) -> tuple:
    """Ask once; on problems, repair once. Returns the parser's tuple.

    The parser's first element is the result (or None); the second is the
    list of problems. After the repair the better of the two is kept: the
    repaired answer if it produced anything, else the original's valid parts.
    """
    forget = getattr(provider, "forget", lambda *_a, **_k: None)
    first = parse(provider.ask(prompt, system=AGENT_SYSTEM))
    if not first[1]:
        return first
    forget(prompt, system=AGENT_SYSTEM)
    fixed = repair_prompt(prompt, first[1])
    try:
        second = parse(provider.ask(fixed, system=AGENT_SYSTEM))
    except BudgetExhausted:
        return first
    if second[1]:
        forget(fixed, system=AGENT_SYSTEM)
    return second if second[0] is not None else first


def _usage(provider: Any) -> tuple[int, int]:
    budget = getattr(provider, "budget", None)
    return (budget.used, budget.max_calls) if budget is not None else (0, 0)


def _error_note(label: str, error: Exception | None) -> str:
    if isinstance(error, BudgetExhausted):
        return f"{label}: skipped, the call limit was reached"
    return f"{label}: no usable answer ({error})" if error else f"{label}: no usable answer"


def _opening_round(bundle, provider, evidence, notes) -> tuple[list[AgentCase], list[ProposedScenario]]:
    def opening(persona: Persona):
        facts = slice_for(persona.persona_id, evidence)
        prompt = opening_prompt(persona, bundle.query, bundle.ticker, facts)
        return _ask_checked(
            provider, prompt,
            lambda text: parse_opening(text, persona.persona_id, persona.label, facts),
        )

    cases, proposals = [], []
    for persona, outcome in zip(PERSONAS, run_parallel(opening, list(PERSONAS))):
        if outcome.ok and outcome.value[0] is not None:
            case, problems, proposed = outcome.value
            cases.append(case)
            seen = {p.name.lower() for p in proposals}
            proposals.extend(p for p in proposed if p.name.lower() not in seen)
            if problems:
                notes.append(f"{persona.label}: {len(problems)} part(s) failed the checks and were dropped")
        else:
            reason = outcome.error if not outcome.ok else "; ".join(outcome.value[1][:2])
            notes.append(_error_note(persona.label, outcome.error) if not outcome.ok
                         else f"{persona.label}: no usable opening ({reason})")
    return cases, proposals


def _challenge_round(cases, provider, evidence, notes) -> None:
    survivors = rank([c for c in cases if c.eliminated_in_round is None])

    def pick_target(challenger: AgentCase) -> AgentCase:
        others = [c for c in survivors if c is not challenger]
        opposed = [c for c in others if c.stance != challenger.stance]
        return (opposed or others)[0]

    def challenge(challenger: AgentCase):
        target = pick_target(challenger)
        index = max(range(len(target.claims)), key=lambda i: (len(target.claims[i].evidence_ids), -i))
        own = slice_for(challenger.persona, evidence)
        theirs = {i: evidence[i] for i in target.claims[index].evidence_ids}
        visible = {**own, **theirs}
        prompt = challenge_prompt(_BY_ID[challenger.persona], own, target, index, theirs)
        return target, _ask_checked(
            provider, prompt, lambda text: parse_challenge(text, challenger.persona_label, target, visible)
        )

    for challenger, outcome in zip(survivors, run_parallel(challenge, survivors)):
        if outcome.ok and outcome.value[1][0] is not None:
            target, (result, _) = outcome.value
            target.challenges_received.append(result)
        else:
            notes.append(_error_note(f"{challenger.persona_label}'s challenge", outcome.error))


def _rebuttal_round(cases, provider, evidence, notes) -> None:
    challenged = [c for c in cases if c.eliminated_in_round is None and c.challenges_received]

    def rebut(case: AgentCase):
        facts = slice_for(case.persona, evidence)
        prompt = rebuttal_prompt(_BY_ID[case.persona], case, facts)
        return _ask_checked(provider, prompt, lambda text: parse_rebuttal(text, case.persona_label, facts))

    for case, outcome in zip(challenged, run_parallel(rebut, challenged)):
        if outcome.ok and outcome.value[0] is not None:
            case.rebuttal = outcome.value[0]
        else:
            notes.append(_error_note(f"{case.persona_label}'s rebuttal", outcome.error))


def _judge(finalists, provider, evidence, depth, notes) -> None:
    try:
        text = provider.ask(judge_prompt(finalists, evidence, depth), system=AGENT_SYSTEM)
    except LLMUnavailable as exc:
        notes.append(_error_note("Judge", exc) + "; ranked on the rubric alone")
        return
    scores, summaries, problems = parse_judge(text, finalists, evidence)
    if problems:
        notes.append(f"Judge: {len(problems)} part(s) failed the checks and were not used")
    for case in finalists:
        if case.persona in scores and case.score is not None:
            case.score = replace(case.score, judge=scores[case.persona])
        case.plain_summary = summaries.get(case.persona)


def run_debate(bundle: EvidenceBundle, provider: LLMProvider, depth: str = "beginner") -> DebateResult:
    """Run the whole debate over one bundle. Never raises for model failures."""
    evidence = build_evidence(bundle)
    scenario_ids = {i for i, item in evidence.items() if item.group == "S"}
    notes: list[str] = []

    cases, proposals = _opening_round(bundle, provider, evidence, notes)
    if len(cases) < 2:
        used, limit = _usage(provider)
        notes.append("fewer than two checked cases survived, so no debate was held")
        return DebateResult("fallback", cases, [], evidence, proposals, 0, notes, used, limit)

    def rescore(rebuttals_done: bool) -> None:
        for case in cases:
            case.score = score_case(case, scenario_ids, rebuttals_done)

    rescore(False)
    eliminate_weakest(cases, 1)
    _challenge_round(cases, provider, evidence, notes)
    _rebuttal_round(cases, provider, evidence, notes)
    rescore(True)
    eliminate_weakest(cases, 3)

    finalists = [c for c in cases if c.eliminated_in_round is None]
    _judge(finalists, provider, evidence, depth, notes)
    outcome, leaders = decide(cases)
    for case in cases:
        case.historical_support_pct = historical_support_pct(case.stance, bundle)
    used, limit = _usage(provider)
    return DebateResult(outcome, rank(cases), leaders, evidence, proposals, 3, notes, used, limit)
