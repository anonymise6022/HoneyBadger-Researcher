"""One call from the CLI to a finished, validated debate report.

Resolves the provider, wraps it in the call budget and cache, runs the
debate, and builds the report -- falling back to the deterministic outlook
report whenever no model is available, the debate produced too little, or
the final check fails. No printing happens here; the caller decides how to
show what comes back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..evidence.schema import EvidenceBundle
from ..synthesis.debate_report import DebateReport, build_debate_report
from ..synthesis.llm_budget import DEFAULT_CACHE_DIR, BudgetedProvider, CallBudget
from ..synthesis.llm_provider import LLMProvider, resolve_provider
from ..synthesis.llm_report import LLMUnavailable
from ..synthesis.outlook_report import render_outlook_report, sections_for
from ..synthesis.validator import validate_report
from .models import DebateResult
from .protocol import run_debate

__all__ = ["DebateRun", "run_debate_flow"]


@dataclass
class DebateRun:
    report: DebateReport
    result: DebateResult | None
    usage: str
    notes: list[str] = field(default_factory=list)
    provider_name: str = ""
    #: The budgeted provider, so a follow-up (the plan's plain-words retelling)
    #: spends from the same per-question limit. None when no model was used.
    provider: BudgetedProvider | None = None


def run_debate_flow(
    bundle: EvidenceBundle,
    provider_name: str = "auto",
    max_calls: int = 12,
    depth: str = "beginner",
    provider: LLMProvider | None = None,
    use_cache: bool = True,
) -> DebateRun:
    """Run the debate end to end. Never raises for model trouble."""
    budget = CallBudget(max_calls)
    try:
        inner = provider or resolve_provider(provider_name)
    except LLMUnavailable as exc:
        text = render_outlook_report(bundle, depth)
        report = DebateReport(
            text, validate_report(text, bundle, required_sections=sections_for("outlook")),
            False, f"no model is available ({exc})",
        )
        return DebateRun(report, None, budget.summary(), [])

    caching = use_cache and inner.name != "mock"
    wrapped = BudgetedProvider(inner, budget, DEFAULT_CACHE_DIR if caching else None)
    result = run_debate(bundle, wrapped, depth)
    report = build_debate_report(bundle, result, depth)
    notes = list(result.notes)
    if inner.name == "mock":
        notes.insert(0, "demo agents: these are canned answers; no model was called")
    return DebateRun(report, result, budget.summary(), notes, inner.name,
                     wrapped if inner.name != "mock" else None)
