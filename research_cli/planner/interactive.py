"""The opt-in conversation that leads to a plan -- only if the user asks.

After a report the user is asked once, defaulting to no. Nothing else in the
tool brings up investing. Saying yes leads to three questions -- how much,
for how long, the most you could stand to lose -- and then to the plan.
`input_fn` and `output` are injectable so the whole flow runs in tests
without a terminal.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import date, timedelta

from ..data.price_ingest import PriceDataError, Source, UnknownSymbolError, fetch_history
from ..evidence.schema import EvidenceBundle
from .inputs import PlanInputError, PlanRequest, parse_amount, parse_horizon, parse_max_loss
from .plan import build_plan
from .render import explain_with_model, render_plan, validate_plan

__all__ = ["build_plan_report", "offer_plan"]

_HISTORY_YEARS = 20


def build_plan_report(
    symbol: str, display: str, amount: float, horizon_days: int, horizon_label: str,
    max_loss: float, source: Source = "auto", as_of: date | None = None, depth: str = "beginner",
) -> tuple[str, object, object]:
    """Fetch history, build the plan, render and validate it: (text, validation, plan)."""
    end = as_of or date.today()  # noqa: DTZ011
    frame, used, _ = fetch_history(symbol, end - timedelta(days=int(_HISTORY_YEARS * 365.25)), end, source)
    request = PlanRequest(symbol, display, amount, horizon_days, horizon_label, max_loss)
    plan = build_plan(request, frame, is_synthetic=used == "mock")
    text = render_plan(plan, depth)
    return text, validate_plan(text, plan), plan


def _ask(prompt: str, parse: Callable[[str], object], input_fn, output):
    for _ in range(2):
        try:
            return parse(input_fn(prompt))
        except PlanInputError as exc:
            output(f"  {exc}")
        except EOFError:
            return None
    return None


def offer_plan(
    bundle: EvidenceBundle,
    plain: bool = False,
    source: Source = "auto",
    provider=None,
    depth: str = "beginner",
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    interactive: bool | None = None,
    as_of: date | None = None,
) -> str | None:
    """Ask once; on yes, gather the three answers and show the plan.

    Returns the plan text that was shown, or None. Runs only in an
    interactive terminal unless `interactive` says otherwise.
    """
    if not (interactive if interactive is not None else sys.stdin.isatty()):
        return None
    try:
        answer = input_fn("\nWant help planning an investment in this? (y/N) ").strip().lower()
    except EOFError:
        return None
    if answer not in ("y", "yes"):
        output("OK. Nothing more to do here; the decision is yours.")
        return None

    amount = _ask("How much would you put in? (e.g. $1000) ", parse_amount, input_fn, output)
    if amount is None:
        return None
    horizon = _ask("How long could you leave it alone? (e.g. 6 months, 1 year) ",
                   parse_horizon, input_fn, output)
    if horizon is None:
        return None
    max_loss = _ask("What is the most you could stand to lose? (e.g. $200 or 20%) ",
                    lambda text: parse_max_loss(text, amount), input_fn, output)
    if max_loss is None:
        return None

    symbol = bundle.observation.symbol if bundle.observation else bundle.ticker
    try:
        text, validation, plan = build_plan_report(
            symbol, bundle.ticker, amount, horizon[0], horizon[1], max_loss, source,
            as_of=as_of, depth=depth)
    except (PriceDataError, UnknownSymbolError, ValueError) as exc:
        output(f"Could not build a plan: {exc}")
        return None
    if not validation.ok:
        output("The plan did not pass its own figure check, so it is not shown:")
        for issue in validation.errors[:4]:
            output(f"  {issue}")
        return None
    output(text)
    if provider is not None:
        simpler = explain_with_model(text, plan, provider, depth, bundle)
        if simpler:
            output("IN SIMPLER WORDS (written by a model, checked against the figures above)")
            output(simpler)
        budget = getattr(provider, "budget", None)
        if budget is not None:
            output(budget.summary())
    output(f"[plan check] {validation.summary()}")
    return text
