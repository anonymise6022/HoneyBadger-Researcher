"""The desktop bridge for the debate, scenarios and plans -- kept out of api.py.

`AppApi` is `DesktopApi` plus three methods the Debate tab calls. It is a
subclass rather than an edit so the original bridge does not grow, and so
everything the existing tabs rely on is untouched.

**What the window shows is the validated text.** The debate report and the
plan are returned as the exact strings the validator passed, and the
interface prints them as they are. Nothing on the Debate tab is computed in
JavaScript.

**Offline mode uses the demo agents.** When the app runs on synthetic data
(`--mock`), the debate uses the canned demo agents and makes no model calls,
so the layout can be explored without spending any of the user's allowance.

**Forward-looking questions on the Ask tab are handed over.** "Is Tesla
going to go up" used to get a report about yesterday's move. `research` now
answers such questions with a pointer the interface follows to this tab.
"""

from __future__ import annotations

import traceback
from typing import Any

from ..data.price_ingest import PriceDataError, UnknownSymbolError
from ..debate.runner import run_debate_flow
from ..evidence.outlook import build_outlook_bundle
from ..explain import resolve_depth
from ..planner.inputs import PlanInputError, parse_amount, parse_horizon, parse_max_loss
from ..planner.interactive import build_plan_report
from ..query_parser import QueryParseError, parse_query
from ..synthesis.claude_cli import ClaudeCLIProvider
from ..synthesis.llm_budget import DEFAULT_MAX_CALLS
from .api import DesktopApi, _fail, _ok

__all__ = ["AppApi"]


class AppApi(DesktopApi):
    """The bridge the window uses: everything before, plus the Debate tab."""

    def research(self, query: str, depth: str = "beginner") -> dict[str, Any]:
        """As before, except questions about the future go to the Debate tab."""
        try:
            parsed = parse_query(query)
        except QueryParseError:
            return super().research(query, depth)
        if parsed.question_type == "outlook":
            result = _fail(
                "That is a question about where the price goes next.",
                "Nobody can answer that, but the Debate tab shows what history says "
                "and lets several viewpoints argue it out.",
            )
            result["debate_query"] = query
            return result
        return super().research(query, depth)

    def debate_status(self) -> dict[str, Any]:
        """Whether the user's own Claude can be used. Spends no model calls."""
        if self.mock:
            return _ok(mode="demo", detail="Offline: canned demo agents, no model calls.")
        provider = ClaudeCLIProvider()
        if provider.available():
            return _ok(mode="claude", detail="Uses your Claude login. Up to "
                       f"{DEFAULT_MAX_CALLS} model calls per question.")
        return _ok(mode="none", detail=(
            "Claude Code is not set up, so the plain report is shown instead. To use "
            "it, install Claude Code and run `claude auth login` in Terminal."
        ))

    def debate(self, query: str, depth: str = "beginner") -> dict[str, Any]:
        """Run the debate and return the validated report text."""
        try:
            depth = resolve_depth(depth)
            parsed = parse_query(query)
        except (QueryParseError, ValueError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))
        if parsed.question_type == "attribution":
            return _fail(
                "The debate is for questions about a company or its future.",
                "Try “is TSLA going to go up”. Questions about why a price moved belong "
                "on the Ask tab.",
            )
        try:
            bundle = build_outlook_bundle(parsed, source=self._source)
            run = run_debate_flow(bundle, "mock" if self.mock else "auto",
                                  DEFAULT_MAX_CALLS, depth)
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))
        except Exception as exc:  # noqa: BLE001 - never surface a traceback
            traceback.print_exc()
            return _fail(f"Something went wrong ({type(exc).__name__}: {exc})")

        result = run.result
        outcome = (
            "standard" if not run.report.used_debate
            else "clear" if result and result.outcome == "clear" else "several"
        )
        return _ok(
            ticker=bundle.ticker,
            report=run.report.text,
            outcome=outcome,
            reason=run.report.reason,
            notes=run.notes,
            usage=run.usage,
            synthetic=bundle.is_synthetic,
            validation=run.report.validation.summary(),
            validation_ok=run.report.validation.ok,
        )

    def plan(self, ticker: str, amount: str, horizon: str, max_loss: str,
             depth: str = "beginner") -> dict[str, Any]:
        """Plan options for a decision the user has made. Places no orders."""
        try:
            dollars = parse_amount(amount)
            days, label = parse_horizon(horizon)
            limit = parse_max_loss(max_loss, dollars)
            depth = resolve_depth(depth)
        except (PlanInputError, ValueError) as exc:
            return _fail(str(exc))
        symbol = ticker.strip().upper()
        try:
            text, validation, _ = build_plan_report(
                symbol, symbol, dollars, days, label, limit, self._source, depth=depth)
        except (UnknownSymbolError, PriceDataError, ValueError) as exc:
            return _fail(f"Could not build a plan: {exc}")
        if not validation.ok:
            return _fail("The plan did not pass its own figure check, so it is not shown.")
        return _ok(report=text, validation=validation.summary())
