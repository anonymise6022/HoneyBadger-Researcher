"""Render the outlook bundle: the snapshot, plus "what if" scenarios from history.

No model. Every figure is read off a bundle field, so the validator can trace
it. The wording is fixed by one rule: a scenario number is always described
as how often something happened *before* and what followed *then*. Phrases
like "the chance it will" never appear, because nothing here measures the
future.
"""

from __future__ import annotations

from ..evidence.scenario_schema import ScenarioResult
from ..evidence.schema import EvidenceBundle
from ..explain import Lexicon
from .snapshot_report import SNAPSHOT_SECTIONS, render_snapshot_report

__all__ = [
    "OUTLOOK_SECTIONS",
    "SCENARIO_HEADING",
    "render_outlook_report",
    "render_scenarios_section",
    "sections_for",
]

SCENARIO_HEADING = "WHAT IF: SCENARIOS FROM HISTORY"
OUTLOOK_SECTIONS = SNAPSHOT_SECTIONS[:-2] + (SCENARIO_HEADING,) + SNAPSHOT_SECTIONS[-2:]
MAIN_HORIZON = 21

_WIDTH = 78


def sections_for(question_type: str) -> tuple[str, ...]:
    """The headings a report of this question type must contain."""
    from .template_report import REQUIRED_SECTIONS  # noqa: PLC0415

    return {
        "snapshot": SNAPSHOT_SECTIONS,
        "outlook": OUTLOOK_SECTIONS,
    }.get(question_type, REQUIRED_SECTIONS)


def _wrap(text: str, indent: str = "  ") -> str:
    continuation = " " * len(indent)
    lines: list[str] = []
    current = indent
    for word in text.split():
        if current.strip() and len(current) + len(word) > _WIDTH:
            lines.append(current.rstrip())
            current = f"{continuation}{word} "
        else:
            current = f"{current}{word} "
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def _signed(value: float | None, decimals: int = 1) -> str:
    return "not available" if value is None else f"{value:+.{decimals}f}%"


def _scenario_lines(item: ScenarioResult, index: int, ticker: str, lex: Lexicon) -> list[str]:
    title = item.plain_label if lex.depth == "kid" else item.label
    if item.trigger_description and item.quantified:
        title += f" -- {item.trigger_description}"
    lines = [_wrap(title, indent=f"  {index}. ")]
    pad = "     "
    if not item.quantified:
        lines.append(_wrap(
            "Can't be measured" + (f": {item.note}" if item.note else "") + ".", indent=pad,
        ))
    else:
        if item.frequency_pct is not None:
            lines.append(_wrap(
                f"How often it happened: in {item.months_triggered} of "
                f"{item.months_observed} months ({item.frequency_pct:.0f}%).", indent=pad,
            ))
        outcome = item.outcome(MAIN_HORIZON)
        if outcome is not None and outcome.median_pct is not None:
            lines.append(_wrap(
                f"What {ticker} did over the next {outcome.horizon_days} trading days: "
                f"typically {_signed(outcome.median_pct)}; most cases between "
                f"{_signed(outcome.p10_pct)} and {_signed(outcome.p90_pct)}; worst "
                f"{_signed(outcome.worst_pct)}; it ended lower in "
                f"{outcome.share_lower_pct:.0f}% of cases.", indent=pad,
            ))
            lines.append(_wrap(
                f"Based on {outcome.n_days} days across {item.episodes_detected} separate "
                f"episodes (confidence: {item.confidence}).", indent=pad,
            ))
        elif item.note:
            lines.append(_wrap(f"Note: {item.note}.", indent=pad))
        if item.proxy_is_synthetic:
            lines.append(_wrap("The data for this condition is SYNTHETIC.", indent=pad))
    examples = [
        f"{e.label} {_signed(e.symbol_return_pct)}"
        for e in item.examples if e.symbol_return_pct is not None
    ]
    if examples:
        lines.append(_wrap(f"Past examples, {ticker} across each: " + "; ".join(examples) + ".",
                           indent=pad))
    if item.watch_for:
        lines.append(_wrap("Watch for: " + ", ".join(item.watch_for) + ".", indent=pad))
    return lines


def render_scenarios_section(bundle: EvidenceBundle, depth: str = "beginner") -> str:
    """The scenario section on its own, for the outlook and debate reports."""
    lex = Lexicon(depth)  # type: ignore[arg-type]
    lines = [f"\n{SCENARIO_HEADING}\n{'-' * _WIDTH}"]
    if not bundle.scenarios:
        lines.append(_wrap("No historical scenarios could be measured for this symbol."))
        return "\n".join(lines)
    lines.append(_wrap(
        "These are not predictions. For each kind of market below, this counts how often "
        f"it showed up in the past and what {bundle.ticker} did afterwards. Situations "
        "that price data cannot measure get no number, only what to watch for."
    ))
    baseline = bundle.scenario_baseline
    base = baseline.outcome(MAIN_HORIZON) if baseline else None
    if base is not None and base.median_pct is not None:
        lines.append("")
        lines.append(_wrap(
            f"For comparison, across all {baseline.sessions} sessions from "
            f"{baseline.history_start.isoformat()} to {baseline.history_end.isoformat()}, "
            f"{bundle.ticker} was typically {_signed(base.median_pct)} after "
            f"{base.horizon_days} trading days and ended lower in "
            f"{base.share_lower_pct:.0f}% of cases. A scenario only tells you something "
            "when it differs from that."
        ))
    lines.append("")
    for index, item in enumerate(bundle.scenarios, start=1):
        lines.extend(_scenario_lines(item, index, bundle.ticker, lex))
        lines.append("")
    lines.append(f"  {'scenario':<26}{'often':>7}{'5d':>9}{'21d':>9}{'63d':>9}{'lower':>8}")
    for item in bundle.scenarios:
        cells = [
            _signed(o.median_pct) if (o := item.outcome(h)) else "n/a" for h in (5, 21, 63)
        ]
        cells = ["n/a" if c == "not available" else c for c in cells]
        often = f"{item.frequency_pct:.0f}%" if item.frequency_pct is not None else "n/a"
        main = item.outcome(MAIN_HORIZON)
        lower = f"{main.share_lower_pct:.0f}%" if main and main.share_lower_pct is not None else "n/a"
        lines.append(
            f"  {item.label[:25]:<26}{often:>7}{cells[0]:>9}{cells[1]:>9}{cells[2]:>9}{lower:>8}"
        )
    return "\n".join(lines).rstrip()


def render_outlook_report(bundle: EvidenceBundle, depth: str = "beginner") -> str:
    """The snapshot report with the scenario section before its closing."""
    if bundle.question_type != "outlook":
        raise ValueError(f"render_outlook_report handles outlook bundles; got {bundle.question_type!r}")
    return render_snapshot_report(
        bundle,
        scenario_section=render_scenarios_section(bundle, depth),
        title="OUTLOOK -- WHAT HISTORY CAN AND CANNOT SAY",
    )
