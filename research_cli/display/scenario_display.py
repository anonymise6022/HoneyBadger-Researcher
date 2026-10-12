"""Rich rendering of the scenario section. Layout only -- no figures computed here.

Every cell is a bundle field formatted for display, so the table can never
show a number the validated report does not also contain.
"""

from __future__ import annotations

from ..evidence.schema import EvidenceBundle

__all__ = ["print_scenarios"]

_MAIN_HORIZON = 21


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def print_scenarios(bundle: EvidenceBundle, console) -> None:
    """A table of scenarios, then what to watch for in the unmeasurable ones."""
    if not bundle.scenarios:
        return
    from rich.table import Table

    baseline = bundle.scenario_baseline
    base = baseline.outcome(_MAIN_HORIZON) if baseline else None
    caption = (
        f"Any day, for comparison: typically {_signed(base.median_pct)} after "
        f"{_MAIN_HORIZON} trading days, lower in {base.share_lower_pct:.0f}% of cases."
        if base is not None and base.share_lower_pct is not None
        else None
    )
    table = Table(
        title="What if: scenarios from history (not predictions)",
        caption=caption, expand=True, title_justify="left",
    )
    for header, justify in (
        ("Scenario", "left"), ("Often", "right"), ("21d", "right"),
        ("Most cases", "right"), ("Worst", "right"), ("Lower", "right"),
        ("Eps.", "right"), ("Conf.", "left"),
    ):
        table.add_column(header, justify=justify)

    for item in bundle.scenarios:
        outcome = item.outcome(_MAIN_HORIZON)
        if not item.quantified or outcome is None:
            table.add_row(
                item.label, "[dim]can't measure[/dim]" if not item.quantified else
                (f"{item.frequency_pct:.0f}%" if item.frequency_pct is not None else "n/a"),
                "", "", "", "", str(item.episodes_detected) if item.quantified else "",
                "[dim]none[/dim]",
            )
            continue
        table.add_row(
            item.label + (" [red](synthetic)[/red]" if item.proxy_is_synthetic else ""),
            f"{item.frequency_pct:.0f}%" if item.frequency_pct is not None else "n/a",
            _signed(outcome.median_pct),
            f"{_signed(outcome.p10_pct)} to {_signed(outcome.p90_pct)}",
            _signed(outcome.worst_pct),
            f"{outcome.share_lower_pct:.0f}%" if outcome.share_lower_pct is not None else "n/a",
            str(item.episodes_detected),
            item.confidence,
        )
    console.print(table)

    for item in bundle.scenarios:
        if item.watch_for:
            console.print(f"  [bold]{item.label}[/bold] -- watch for: {', '.join(item.watch_for)}")
