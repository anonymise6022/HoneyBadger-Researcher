"""Turn an EvidenceBundle into plain text. No model, no network, no invention.

Pure string templating. Every number printed is read directly off a bundle
field, which is what makes the output trivially verifiable -- and what makes
`validator` meaningful when Phase 2 swaps this for a language model. If the
validator cannot pass the template, it is the validator that is wrong.

The section order is deliberate and is the argument the report makes:

    QUESTION              what was asked, and what the tool assumed
    SUMMARY               what actually happened, and whether it was unusual
    KEY EVIDENCE          what else happened, ranked by historical record
    COMPETING EXPLANATIONS  what argues against the leading story
    CONFIDENCE & SAMPLE SIZE  how much any of this is worth
    RAW DATA              the numbers, so nothing is hidden

Putting "was this even unusual" in the summary, before any candidate
factors, is the single most important choice here. Most daily moves are
noise, and a report that leads with a list of explanations implicitly
asserts there is something to explain.

**Language rules, enforced rather than hoped for.** This module never emits
"caused", "because", "due to", "drove", "triggered" or "led to". Factors
"coincided with" a move or are "associated with" it. `validator.py` checks
the output against that list, so a careless edit here fails a test rather
than shipping a causal claim.
"""

from __future__ import annotations

from ..evidence.schema import CandidateFactor, EvidenceBundle
from ..explain import Lexicon

__all__ = ["REQUIRED_SECTIONS", "render_factor_line", "render_report"]

REQUIRED_SECTIONS = (
    "QUESTION",
    "SUMMARY",
    "KEY EVIDENCE",
    "COMPETING EXPLANATIONS",
    "CONFIDENCE & SAMPLE SIZE",
    "RAW DATA",
)

_WIDTH = 78
_BULLET = "  - "


def _rule(char: str = "=") -> str:
    return char * _WIDTH


def _heading(title: str) -> str:
    return f"\n{title}\n{_rule('-')}"


def _wrap(text: str, indent: str = "  ") -> str:
    """Wrap to the report width, preserving the indent on every line."""
    # A bullet marker belongs on the first line only. Reusing the full
    # indent for continuations prints "- " down the left margin of every
    # wrapped line, which reads as a list of fragments rather than one item.
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


def render_factor_line(
    factor: CandidateFactor, index: int, lexicon: Lexicon | None = None
) -> str:
    """One factor, with its record and an honest reading of that record."""
    lex = lexicon or Lexicon("beginner")
    lines = [f"  {index}. {factor.label}"]
    lines.append(_wrap(factor.what_happened, indent="     "))

    if factor.historical_hit_rate is None:
        lines.append(
            _wrap(
                "Track record: not enough comparable days to measure one"
                + (f" ({factor.note})" if factor.note else ""),
                indent="     ",
            )
        )
        return "\n".join(lines)

    verdict = (
        "better than the base rate"
        if factor.beats_base_rate and (factor.lift or 0) > 0
        else "worse than the base rate"
        if factor.beats_base_rate
        else "no better than the base rate -- this is within the margin of error"
    )
    lines.append(
        _wrap(
            f"Track record: on {factor.condition_description}, this factor "
            f"coincided with a move in this direction {factor.historical_hit_rate:.0%} "
            f"of the time{lex.gloss('hit_rate')}, across {factor.sample_size} days. "
            f"The base rate on all days{lex.gloss('base_rate')} "
            f"was {factor.base_rate:.0%}, so this is {verdict}."
            + (
                f" Lift {factor.lift:+.1%}, standard error "
                f"{factor.hit_rate_standard_error:.1%}."
                if lex.is_analyst and factor.lift is not None
                else ""
            ),
            indent="     ",
        )
    )
    if factor.note and factor.historical_hit_rate is not None:
        lines.append(_wrap(f"Note: {factor.note}", indent="     "))
    return "\n".join(lines)


def _render_question(bundle: EvidenceBundle) -> str:
    lines = [_heading("QUESTION"), f'  "{bundle.query}"', f"  Symbol: {bundle.ticker}"]
    lines.append(f"  Period: {bundle.period_description}")
    if bundle.observation and bundle.observation.session_differs_from_request:
        lines.append(
            f"  Note: {bundle.observation.requested_date.isoformat()} was not a trading day; "
            f"this covers {bundle.observation.session_date.isoformat()}."
        )
    for warning in bundle.warnings:
        lines.append(_wrap(f"Assumption: {warning}", indent="  "))
    return "\n".join(lines)


def _render_summary(bundle: EvidenceBundle, lex: Lexicon) -> str:
    observation = bundle.observation
    lines = [_heading("SUMMARY")]
    if observation is None:
        lines.append("  No price observation was available for this question.")
        return "\n".join(lines)

    lines.append(_wrap(observation.plain_summary))
    if lex.is_analyst:
        lines.append(
            _wrap(
                f"z = {observation.sigma_multiple:+.2f}, sigma_daily = "
                f"{observation.daily_vol_pct:.2f}%, sigma_annual = "
                f"{observation.annualized_vol_pct:.1f}%, n = "
                f"{observation.lookback_sessions}."
            )
        )
    lines.append("")
    if not observation.is_unusual:
        lines.append(
            _wrap(
                "This move sits within the symbol's normal range, so ordinary day-to-day "
                "fluctuation already accounts for it. The factors below are things that "
                "happened at the same time, not explanations of the move."
            )
        )
    else:
        supported = bundle.supported_factors()
        if supported:
            best = supported[0]
            lines.append(
                _wrap(
                    f"This was a large move. The factor with the strongest historical "
                    f"association is {best.label}, which has coincided with moves "
                    f"in this direction {best.historical_hit_rate:.0%} of the time on "
                    f"comparable days, against a {best.base_rate:.0%} base rate. That is an "
                    f"association only -- it does not show that one made the other happen."
                )
            )
        else:
            lines.append(
                _wrap(
                    "This was a large move, but none of the factors checked has a historical "
                    "track record better than the base rate. In other words, the evidence "
                    "available here does not single out an explanation."
                )
            )
    return "\n".join(lines)


def _render_evidence(bundle: EvidenceBundle, lex: Lexicon) -> str:
    lines = [_heading("KEY EVIDENCE")]
    external = bundle.external_factors()
    if not external:
        lines.append("  No outside factors could be measured for this period.")
    else:
        lines.append(
            _wrap(
                "Ranked by how often each has coincided with moves like this one in the "
                f"past {bundle.history_years:.0f} years -- not by how convincing the story "
                "sounds.",
            )
        )
        lines.append("")
        for index, factor in enumerate(external, start=1):
            lines.append(render_factor_line(factor, index, lex))
            lines.append("")

    mechanical = bundle.mechanical_factors()
    if mechanical:
        lines.append("  WHAT THE MOVE WAS MADE OF")
        lines.append(
            _wrap(
                "These are parts of the symbol itself, so they describe the shape of the "
                "move rather than explaining it. Their high match rates are arithmetic.",
                indent="     ",
            )
        )
        lines.append("")
        for index, factor in enumerate(mechanical, start=1):
            lines.append(render_factor_line(factor, index, lex))
            lines.append("")

    if bundle.macro_releases:
        lines.append("  ECONOMIC DATA RELEASED AROUND THIS DAY")
        for release in bundle.macro_releases:
            lines.append(_wrap(release.description, indent="     - "))
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_competing(bundle: EvidenceBundle) -> str:
    lines = [_heading("COMPETING EXPLANATIONS")]
    if not bundle.counterevidence:
        lines.append(
            _wrap(
                "Nothing in the data collected argues strongly against the evidence above. "
                "That is not the same as confirming it."
            )
        )
        return "\n".join(lines)
    for item in bundle.counterevidence:
        lines.append(f"{_BULLET}{item.label}")
        lines.append(_wrap(item.detail, indent="      "))
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_confidence(bundle: EvidenceBundle, lex: Lexicon) -> str:
    lines = [_heading("CONFIDENCE & SAMPLE SIZE")]
    scored = [f for f in bundle.factors if f.historical_hit_rate is not None]
    if scored:
        smallest = min(f.sample_size for f in scored)
        largest = max(f.sample_size for f in scored)
        lines.append(
            _wrap(
                f"Hit rates come from {bundle.history_years:.0f} years of daily history. "
                f"Each is based on between {smallest} and {largest} comparable days. "
                "A rate built on fewer than 10 days is not reported at all."
            )
        )
        supported = bundle.supported_factors()
        lines.append("")
        lines.append(
            _wrap(
                f"{len(supported)} of {len(bundle.external_factors())} outside factors have a "
                "record that clears two standard errors above the base rate. The rest are "
                "inside the margin of error and should be read as coincidence."
            )
        )
    else:
        lines.append(
            _wrap("No factor had enough history to measure a track record for this period.")
        )
    if bundle.limitations:
        lines.append("")
        lines.append("  Limitations:")
        for limitation in bundle.limitations:
            lines.append(_wrap(limitation, indent="     - "))
    return "\n".join(lines)


def _render_raw(bundle: EvidenceBundle) -> str:
    lines = [_heading("RAW DATA")]
    observation = bundle.observation
    if observation is not None:
        lines.extend(
            [
                f"  Session date          {observation.session_date.isoformat()}",
                f"  Close                 {observation.close:.2f}",
                f"  Previous close        {observation.prior_close:.2f}",
                f"  Move over period      {observation.period_return_pct:+.2f}%",
                f"  Typical daily swing   {observation.daily_vol_pct:.2f}%",
                f"  Annualised volatility {observation.annualized_vol_pct:.2f}%",
                f"  Standard deviations   {observation.sigma_multiple:+.2f}",
                f"  Size category         {observation.outlier_label}",
                f"  Baseline window       {observation.lookback_sessions} sessions",
            ]
        )
        if observation.volume_ratio is not None:
            lines.append(f"  Volume vs average     {observation.volume_ratio:.2f}x")
    if bundle.factors:
        lines.append("")
        lines.append(
            f"  {'factor':<24}{'value':>9}{'hit':>7}{'base':>7}{'lift':>8}{'days':>7}  confidence"
        )
        for factor in bundle.ranked_factors():
            hit = f"{factor.historical_hit_rate:.0%}" if factor.historical_hit_rate is not None else "n/a"
            base = f"{factor.base_rate:.0%}" if factor.base_rate is not None else "n/a"
            lift = f"{factor.lift:+.0%}" if factor.lift is not None else "n/a"
            lines.append(
                f"  {factor.label[:23]:<24}{factor.observed_value:>9.2f}{hit:>7}{base:>7}"
                f"{lift:>8}{factor.sample_size:>7}  {factor.confidence}"
            )
    lines.append("")
    lines.append(f"  Data sources: {'; '.join(bundle.data_sources) or 'none recorded'}")
    if bundle.is_synthetic:
        lines.append("  WARNING: this report uses SYNTHETIC data. The numbers are not real.")
    return "\n".join(lines)


def render_report(bundle: EvidenceBundle, depth: str = "beginner") -> str:
    """Render a full attribution report as plain text.

    `depth` selects how much terminology is explained in passing. It changes
    wording only -- every level renders from the same bundle and passes the
    same validator, so two readers at different depths cannot come away with
    different figures.

    Raises ValueError for a snapshot bundle -- that has its own renderer, and
    silently producing an attribution-shaped report for it would drop the
    fundamentals without saying so.
    """
    if bundle.question_type != "attribution":
        raise ValueError(
            f"render_report handles attribution bundles; got {bundle.question_type!r}. "
            "Use synthesis.snapshot_report for snapshot questions."
        )
    lex = Lexicon(depth)  # type: ignore[arg-type]

    body = "\n".join(
        [
            _rule(),
            f"  RESEARCH NOTE -- {bundle.ticker}",
            _rule(),
            _render_question(bundle),
            _render_summary(bundle, lex),
            _render_evidence(bundle, lex),
            _render_competing(bundle),
            _render_confidence(bundle, lex),
            _render_raw(bundle),
            "",
            _rule(),
            _wrap(
                "This is a summary of evidence, not investment advice. Nothing above "
                "identifies what made the move happen -- markets move for motives that "
                "are never observable -- and every factor listed is something that "
                "happened at the same time.",
            ),
            _rule(),
        ]
    )
    return body + "\n"
