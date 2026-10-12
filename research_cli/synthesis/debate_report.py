"""Render a finished debate as a beginner-simple report, then gate it.

Layout, in the order a beginner needs it:

    QUESTION                 what was asked
    THE SIMPLE PICTURE       what this is, how bumpy, what "usual" looks like
    THE STRONGEST CASE  -or- SEVERAL STRONG VIEWS
    WHAT IF: SCENARIOS FROM HISTORY
    WHAT TO WATCH FOR
    YOU DECIDE

Agent sentences appear only where they already passed the per-sentence
checks, and the whole report is validated again -- numbers traced to the
bundle, no cause, verdict or certainty language. If that final check fails,
`build_debate_report` returns the deterministic outlook report instead, so
unvalidated prose is never shown. Rubric scores are never printed: they rank
arguments and would read as probabilities. The only likelihood figures are
computed by code from the bundle's baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..debate.models import AgentCase, DebateResult
from ..evidence.schema import EvidenceBundle
from ..explain import Lexicon
from .outlook_report import SCENARIO_HEADING, render_outlook_report, render_scenarios_section
from .validator import ValidationReport, validate_report

__all__ = ["DebateReport", "build_debate_report", "debate_sections", "render_debate_report"]

_WIDTH = 78
_STANCE_WORDS = {
    "bull": ("the bull (hopeful) case", "the hopeful view"),
    "bear": ("the bear (cautious) case", "the worried view"),
    "neutral": ("a neutral reading", "the in-between view"),
}
_ROUND_WORDS = {1: "set aside after the opening round", 3: "set aside after the rebuttals"}


def _wrap(text: str, indent: str = "  ") -> str:
    continuation = " " * len(indent)
    lines, current = [], indent
    for word in text.split():
        if current.strip() and len(current) + len(word) > _WIDTH:
            lines.append(current.rstrip())
            current = f"{continuation}{word} "
        else:
            current = f"{current}{word} "
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def _heading(title: str) -> str:
    return f"\n{title}\n{'-' * _WIDTH}"


def case_heading(result: DebateResult) -> str:
    return "THE STRONGEST CASE" if result.outcome == "clear" else "SEVERAL STRONG VIEWS"


def debate_sections(result: DebateResult) -> tuple[str, ...]:
    return ("QUESTION", "THE SIMPLE PICTURE", case_heading(result), SCENARIO_HEADING,
            "WHAT TO WATCH FOR", "YOU DECIDE")


def _simple_picture(bundle: EvidenceBundle, lex: Lexicon) -> list[str]:
    lines = [_heading("THE SIMPLE PICTURE")]
    f, t, ticker = bundle.fundamentals, bundle.trend, bundle.ticker
    name = (f.company_name if f else None) or ticker
    if f is not None and f.sector:
        lines.append(_wrap(f"{name} ({ticker}) is a company in {f.sector.lower()}."))
    else:
        lines.append(_wrap(f"{name} ({ticker}) is a fund: it holds many companies at once."))
    if t is not None:
        if t.return_1y_pct is not None:
            way = "up" if t.return_1y_pct >= 0 else "down"
            lines.append(_wrap(f"Over the past year its price went {way} {abs(t.return_1y_pct):.1f}%."))
        bumpier = "bumpier than" if t.is_more_volatile_than_market else "about as calm as, or calmer than,"
        lines.append(_wrap(
            f"Its price swings about {t.annualized_vol_pct:.0f}% a year"
            f"{lex.gloss('volatility')}, which is {bumpier} a broad index fund."
        ))
    if f is not None and f.is_profitable is False:
        lines.append(_wrap(lex.pick(
            "It currently loses money, which changes how every other figure reads.",
            "It is currently loss-making.", "Loss-making.",
            kid="Right now it spends more money than it makes.")))
    base = bundle.scenario_baseline.outcome(21) if bundle.scenario_baseline else None
    if base is not None and base.share_higher_pct is not None:
        lines.append("")
        lines.append(_wrap(
            f"What usually happens: across {bundle.scenario_baseline.sessions} past trading "
            f"days, {ticker} was higher 21 trading days later in {base.share_higher_pct:.0f}% "
            f"of cases and lower in {base.share_lower_pct:.0f}%."
            + lex.pick(
                " Any case below has to say something that beats that.",
                " The cases below are read against that.", "",
                kid=" That is like a coin that lands one way a bit more often than the other. "
                    "It describes the past, not tomorrow.")
        ))
    if bundle.counterevidence:
        lines.append("")
        lines.append(_wrap("Things to keep in mind:"))
        for item in bundle.counterevidence:
            lines.append(_wrap(f"{item.label}. {item.detail}", indent="  - "))
    return lines


def _support_line(case: AgentCase, ticker: str) -> str:
    if case.historical_support_pct is None:
        return "History check: a neutral reading has no direction to check against history."
    way = "higher" if case.stance == "bull" else "lower"
    return (
        f"History check (computed from past prices, not the agents' opinion): {ticker} "
        f"ended {way} after 21 trading days in {case.historical_support_pct:.0f}% of past "
        "cases. That is how often history agreed with this direction before, not a forecast."
    )


def _case_block(case: AgentCase, ticker: str, lex: Lexicon, full: bool) -> list[str]:
    stance = _STANCE_WORDS[case.stance][1 if lex.is_kid else 0]
    lines = [_wrap(f"{case.persona_label}: {stance}", indent="  * ")]
    pad = "      "
    lines.append(_wrap(f"In one line: {case.headline}", indent=pad))
    if case.plain_summary and full:
        lines.append(_wrap(case.plain_summary, indent=pad))
    if full:
        lines.append(_wrap("What it rests on:", indent=pad))
        for claim in case.claims:
            lines.append(_wrap(f"{claim.text} [{', '.join(claim.evidence_ids)}]", indent=pad + "- "))
        lines.append(_wrap(_support_line(case, ticker), indent=pad))
        for challenge in case.challenges_received:
            lines.append(_wrap(f"Challenged by the {challenge.by}: {challenge.text}", indent=pad))
        if case.rebuttal is not None:
            verb = "Conceded" if case.rebuttal.concedes else "Answer"
            lines.append(_wrap(f"{verb}: {case.rebuttal.text}", indent=pad))
    elif case.eliminated_in_round in _ROUND_WORDS:
        lines.append(_wrap(f"This view was {_ROUND_WORDS[case.eliminated_in_round]}.", indent=pad))
    lines.append(_wrap("What would prove it wrong: " + "; ".join(case.falsifiers), indent=pad))
    return lines


def _cases(bundle: EvidenceBundle, result: DebateResult, lex: Lexicon) -> list[str]:
    lines = [_heading(case_heading(result))]
    if result.outcome == "clear":
        lines.append(_wrap(
            f"The {result.leaders[0].persona_label} made the best-supported case on this "
            "evidence. That is a judgement about how well it was argued, not a prediction."
        ))
    else:
        lines.append(_wrap(
            "No single case won. These views scored too close together to separate, so "
            "they are shown side by side."
        ))
    lines.append("")
    leader_ids = {c.persona for c in result.leaders}
    for case in result.leaders:
        lines.extend(_case_block(case, bundle.ticker, lex, full=True))
        lines.append("")
    others = [c for c in result.cases if c.persona not in leader_ids]
    if others:
        lines.append(_wrap("Other views:"))
        for case in others:
            lines.extend(_case_block(case, bundle.ticker, lex, full=False))
    return lines


def _watch_for(bundle: EvidenceBundle, result: DebateResult) -> list[str]:
    lines = [_heading("WHAT TO WATCH FOR")]
    for case in result.leaders:
        lines.append(_wrap(f"Signs the {case.persona_label} is wrong: " + "; ".join(case.falsifiers),
                           indent="  - "))
    labels = {s.scenario_id: s.label for s in bundle.scenarios}
    for item in bundle.scenarios:
        if item.watch_for:
            lines.append(_wrap(f"{item.label}: " + ", ".join(item.watch_for) + ".", indent="  - "))
    for proposal in result.proposed_scenarios:
        where = (f"measured above under '{labels[proposal.mapped_to]}'" if proposal.mapped_to in labels
                 else "can't be measured from price data; watch the news for it")
        lines.append(_wrap(f"{proposal.name} (raised by an agent): {where}.", indent="  - "))
    return lines


def _you_decide(bundle: EvidenceBundle) -> list[str]:
    lines = [_heading("YOU DECIDE")]
    lines.append(_wrap(
        f"Nobody can tell you where {bundle.ticker} goes next, and this report has not told "
        "you to do anything. It showed what history says and how the arguments held up. "
        "Whether any of it fits you depends on how long you can wait, what else you own, and "
        "how big a fall you could live with. The decision is yours."
    ))
    lines.append(_wrap("This is not financial advice."))
    lines.append(_wrap(f"Data sources: {'; '.join(bundle.data_sources) or 'none recorded'}."))
    if bundle.is_synthetic:
        lines.append("  WARNING: this report uses SYNTHETIC data. The numbers are not real.")
    return lines


def render_debate_report(bundle: EvidenceBundle, result: DebateResult, depth: str = "beginner") -> str:
    lex = Lexicon(depth)  # type: ignore[arg-type]
    lines = ["=" * _WIDTH, f"  THE DEBATE -- {bundle.ticker}", "=" * _WIDTH,
             _heading("QUESTION"), f'  "{bundle.query}"']
    for warning in bundle.warnings:
        lines.append(_wrap(f"Note: {warning}"))
    lines += _simple_picture(bundle, lex)
    lines += _cases(bundle, result, lex)
    lines.append(render_scenarios_section(bundle, depth))
    lines += _watch_for(bundle, result)
    lines += _you_decide(bundle)
    return "\n".join(lines + ["", "=" * _WIDTH]) + "\n"


@dataclass
class DebateReport:
    """What the caller shows. `used_debate` False means the fallback ran."""

    text: str
    validation: ValidationReport
    used_debate: bool
    reason: str = ""


def build_debate_report(bundle: EvidenceBundle, result: DebateResult, depth: str = "beginner") -> DebateReport:
    """The validated debate report, or the deterministic outlook report."""
    from .outlook_report import sections_for  # noqa: PLC0415

    def fallback(reason: str) -> DebateReport:
        text = render_outlook_report(bundle, depth)
        return DebateReport(text, validate_report(text, bundle, required_sections=sections_for("outlook")),
                            False, reason)

    if result.outcome == "fallback":
        return fallback("the debate did not produce enough checked arguments")
    text = render_debate_report(bundle, result, depth)
    validation = validate_report(text, bundle, required_sections=debate_sections(result),
                                 forbid_certainty=True)
    if not validation.ok:
        return fallback("the debate report failed the final evidence check")
    return DebateReport(text, validation, True)
