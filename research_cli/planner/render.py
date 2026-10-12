"""Write a plan as plain text, and check it like every other report.

Every figure printed comes from the `Plan`, and the text is validated with
`extra_numbers=plan.numbers()` -- the one place in the package allowed to
widen the numeric trace, because these figures are computed by planner code,
not written by a model. The same verdict and certainty bans apply as
everywhere else: the options are described, never recommended, and nothing
here promises an outcome.

An optional model call can retell the options in simpler words. Its text is
checked against the same figures; if it fails twice it is simply not shown.
"""

from __future__ import annotations

from ..evidence.schema import EvidenceBundle
from ..synthesis.llm_report import LLMUnavailable
from ..synthesis.validator import ValidationReport, validate_report
from .plan import DollarRange, Plan, PlanOption
from .sizing import BAD_STRETCH_SIGMAS

__all__ = ["PLAN_SECTIONS", "explain_with_model", "render_plan", "validate_plan"]

PLAN_SECTIONS = ("YOUR DECISION", "WHAT HISTORY SAYS ABOUT YOUR TIME FRAME", "OPTION A",
                 "OPTION B", "OPTION C", "PRACTICAL POINTS", "YOU DECIDE")
_WIDTH = 78


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


def _usd(value: float, cents: bool = False) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}" if cents else f"{sign}${abs(value):,.0f}"


def _gain(value: float) -> str:
    return ("+" if value >= 0 else "") + _usd(value)


def _range_line(r: DollarRange) -> str:
    return (f"bad stretch {_gain(r.bad)}, typical {_gain(r.typical)}, good stretch "
            f"{_gain(r.good)}, worst ever {_gain(r.worst)}")


def _beyond_limit(option: PlanOption, max_loss: float) -> str:
    if -option.outcome.worst <= max_loss:
        return ""
    return (f" The worst past stretch would have lost {_usd(-option.outcome.worst)}, more than "
            "your limit: a bad stretch is not the worst one.")


def render_plan(plan: Plan, depth: str = "beginner") -> str:
    req, risk = plan.request, plan.risk
    a, b, c = plan.options
    staged = c.staged
    head = lambda title: f"\n{title}\n{'-' * _WIDTH}"  # noqa: E731
    out = ["=" * _WIDTH, f"  A PLAN FOR YOUR DECISION -- {req.display_symbol}", "=" * _WIDTH]

    out += [head("YOUR DECISION"), _wrap(
        f"You said you want to put up to {_usd(req.amount)} into {req.display_symbol} for "
        f"{req.horizon_label} ({req.horizon_days} trading days), and that the most you could "
        f"stand to lose is {_usd(req.max_loss)} ({req.max_loss_pct:.0f}% of it). Everything "
        "below is built around those three answers. This tool never places orders; if you go "
        "ahead, you do that yourself at your broker.")]

    out += [head("WHAT HISTORY SAYS ABOUT YOUR TIME FRAME"), _wrap(
        f"Every past stretch of {req.horizon_days} trading days in {plan.history_sessions} "
        f"sessions of history ({risk.n_windows} stretches): the worst 1 in "
        f"{100 // risk.low_percentile} ended {risk.p10_pct:+.1f}%, a typical one "
        f"{risk.median_pct:+.1f}%, the best 1 in {100 // (100 - risk.high_percentile)} "
        f"{risk.p90_pct:+.1f}%, and the worst of all {risk.worst_pct:+.1f}%."), _wrap(
        f"Lately the price moves about {risk.daily_vol_pct:.2f}% on a normal day, which works out "
        f"to about {risk.horizon_vol_pct:.1f}% over your time frame. This plan calls a fall of "
        f"{risk.bad_stretch_loss_pct:.1f}% a bad stretch: "
        + ("two of those swings would be more than that, so it is capped there. "
           if risk.bad_stretch_loss_pct < BAD_STRETCH_SIGMAS * risk.horizon_vol_pct - 1e-9
           else "two of those swings. ")
        + "These describe the past, not a forecast.")]
    if depth == "kid":
        out.append(_wrap("Think of it like packing for a trip: only bring what you could stand "
                         "to lose if the bag went missing."))

    out += [head(f"OPTION A: {a.title.upper()}"), _wrap(
        f"Put in {_usd(a.invest, True)} and keep {_usd(a.cash, True)} in cash. If a bad stretch "
        f"of {risk.bad_stretch_loss_pct:.1f}% happened, {_usd(a.invest, True)} would lose about "
        f"{_usd(req.max_loss)}, your limit. On past stretches: {_range_line(a.outcome)}."
        + _beyond_limit(a, req.max_loss)), _wrap(
        "Trade-off: far less at risk, and far less to gain if it goes well.")]

    if b.stop is not None:
        body = (
            f"Put in {_usd(b.invest)}, and decide now to sell if the price closes "
            f"{b.stop_pct:.1f}% below where you bought; at today's price that is "
            f"{_usd(b.stop_price, True)}. Replayed on every past stretch, that exit would have "
            f"been used in {b.stop.hit_share_pct:.0f}% of them. Outcomes: {_range_line(b.outcome)}."
            + _beyond_limit(b, req.max_loss))
        trade = ("Trade-off: you keep the full upside, but ordinary swings often set off the exit, "
                 "selling after a fall and possibly before a recovery. A price can also jump "
                 "straight past an exit overnight.")
    else:
        body = (f"Put in {_usd(b.invest)} with no exit rule, since your limit is the whole amount. "
                f"Outcomes: {_range_line(b.outcome)}.")
        trade = "Trade-off: nothing limits a fall except your own decision to sell."
    out += [head(f"OPTION B: {b.title.upper()}"), _wrap(body), _wrap(trade)]

    out += [head(f"OPTION C: {c.title.upper()}"), _wrap(
        f"Put in the same {_usd(c.invest, True)} as option A, in {staged.tranches} equal parts "
        f"{staged.spacing_days} trading days apart, and keep the rest in cash. Tested on "
        f"{staged.starts} past start dates with the backtest engine (each part bought at the next "
        "day's opening price, trading costs included) against putting it all in on day one: "
        f"spreading it out ended higher in {staged.staged_beat_share_pct:.0f}% of start dates. "
        f"Typical ending value {_usd(staged.staged_median, True)} spread out against "
        f"{_usd(staged.lump_median, True)} all at once; worst {_usd(staged.staged_worst, True)} "
        f"against {_usd(staged.lump_worst, True)}. Outcomes: {_range_line(c.outcome)}."), _wrap(
        "Trade-off: if the price falls early, later parts buy cheaper; if it rises, they buy "
        "dearer. Neither way is reliably better; spreading out mostly softens the regret of "
        "bad timing.")]

    out += [head("PRACTICAL POINTS")] + [_wrap(t, indent="  - ") for t in (
        "Costs: many brokers charge no commission on shares, but every trade crosses a small gap "
        "between the buying and selling price. The tests above include a small cost per trade.",
        "Taxes: selling at a gain can be taxed, and the rules depend on where you live and how "
        "long you held. This tool does not give tax advice; ask someone qualified.",
        "Concentration: one company is riskier than many. If this amount is a large part of "
        "your savings, that matters more than anything above.",
        "Orders: this tool never places orders and never connects to a broker.",
    )]
    out += [head("YOU DECIDE"), _wrap(
        "These options describe what happened before with amounts like yours. None is a "
        "recommendation, and none can promise a result; the future can land outside every "
        "range above. Which one, if any, fits you is your call. This is not financial advice.")]
    if plan.is_synthetic:
        out.append("  WARNING: this plan uses SYNTHETIC data. The numbers are not real.")
    return "\n".join(out + ["", "=" * _WIDTH]) + "\n"


def _anchor(plan: Plan, bundle: EvidenceBundle | None) -> EvidenceBundle:
    """A bare bundle for validation: the plan's own figures are the evidence.

    The research bundle is not reused, because its counterevidence rules
    belong to the research report; a plan is checked against its numbers.
    """
    return EvidenceBundle(query=bundle.query if bundle else "", question_type="outlook",
                          ticker=plan.request.display_symbol, period_description="plan")


def validate_plan(text: str, plan: Plan, bundle: EvidenceBundle | None = None,
                  sections: tuple[str, ...] = PLAN_SECTIONS) -> ValidationReport:
    return validate_report(text, _anchor(plan, bundle), required_sections=sections,
                           forbid_certainty=True, extra_numbers=plan.numbers())


def explain_with_model(plan_text: str, plan: Plan, provider, depth: str = "beginner",
                       bundle: EvidenceBundle | None = None) -> str | None:
    """A simpler retelling by a model, or None. At most two calls."""
    reader = "a 10-year-old" if depth == "kid" else "a beginner investor"
    prompt = (
        f"Retell these three plan options in simple words for {reader}, in under 200 words. "
        "Use only numbers that appear in the text. Do not recommend any option, do not say "
        "buy or sell as advice, do not promise any outcome, and do not say what caused anything.\n\n"
        + plan_text
    )
    for _ in range(2):
        try:
            text = provider.ask(prompt)
        except LLMUnavailable:
            return None
        report = validate_plan(text, plan, bundle, sections=())
        if report.ok:
            return text.strip()
        prompt += "\n\nYour last answer was rejected: " + "; ".join(str(i) for i in report.errors[:5])
    return None
