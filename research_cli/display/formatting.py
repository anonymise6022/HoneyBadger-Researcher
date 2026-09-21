"""Terminal rendering with `rich`, degrading cleanly to plain text.

Two rules shape this module.

**Colour carries no information on its own.** Every coloured element is also
labelled in words, because a user may be piping to a file, reading in a
terminal without colour, or colour-blind. Green is never the only thing
saying "this factor is well supported" -- the words "better than the base
rate" are there too.

**Nothing is rendered that is not in the bundle.** The display layer
formats; it never computes a new figure. If a number is wanted on screen it
is added to the bundle first, so the validator still covers it.

If `rich` is missing, `render_bundle` falls back to the plain-text template.
The tool stays usable rather than failing on a presentation dependency.
"""

from __future__ import annotations

from ..evidence.schema import CandidateFactor, EvidenceBundle

__all__ = [
    "console_available",
    "print_backtest",
    "print_bundle",
    "print_error",
    "print_validation",
]


def console_available() -> bool:
    """Whether `rich` can be imported."""
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _confidence_style(factor: CandidateFactor) -> str:
    if factor.relationship == "mechanical":
        return "dim"
    if factor.beats_base_rate and (factor.lift or 0.0) > 0:
        return "green"
    if factor.beats_base_rate:
        return "yellow"
    return "dim"


def _factor_table(bundle: EvidenceBundle, mechanical: bool):
    from rich import box
    from rich.table import Table

    factors = bundle.mechanical_factors() if mechanical else bundle.external_factors()
    if not factors:
        return None

    title = (
        "What the move was made of (arithmetic, not explanation)"
        if mechanical
        else "What else happened, ranked by historical record"
    )
    table = Table(title=title, box=box.SIMPLE_HEAD, title_justify="left", expand=True)
    table.add_column("Factor", style="bold", no_wrap=False, ratio=3)
    table.add_column("Observed", justify="right", ratio=1)
    table.add_column("Match rate", justify="right", ratio=1)
    table.add_column("Base rate", justify="right", ratio=1)
    table.add_column("Days", justify="right", ratio=1)
    table.add_column("Reading", ratio=3)

    for factor in factors:
        if factor.historical_hit_rate is None:
            hit, base, reading = "n/a", "n/a", "too few comparable days"
        else:
            hit = f"{factor.historical_hit_rate:.0%}"
            base = f"{factor.base_rate:.0%}"
            if mechanical:
                reading = "part of the index"
            elif factor.beats_base_rate and (factor.lift or 0.0) > 0:
                reading = f"better than base rate ({factor.lift:+.0%})"
            elif factor.beats_base_rate:
                reading = f"leans the other way ({factor.lift:+.0%})"
            else:
                reading = "within the margin of error"
        table.add_row(
            factor.label,
            f"{factor.observed_value:+.2f}{factor.observed_units}",
            hit,
            base,
            str(factor.sample_size),
            reading,
            style=_confidence_style(factor),
        )
    return table


def _fundamentals_table(bundle: EvidenceBundle):
    """The snapshot's headline figures, each with what it means."""
    from rich import box
    from rich.table import Table

    fundamentals = bundle.fundamentals
    if fundamentals is None:
        return None

    def rate(value: float | None, decimals: int = 1) -> str:
        return "not available" if value is None else f"{value * 100:.{decimals}f}%"

    def ratio(value: float | None, decimals: int = 1) -> str:
        return "not available" if value is None else f"{value:.{decimals}f}"

    rows = [
        ("Price-to-earnings", ratio(fundamentals.trailing_pe),
         "what you pay per $1 of last year's profit"),
        ("Revenue growth", rate(fundamentals.revenue_growth),
         "how fast sales grew over the past year"),
        ("Profit margin", rate(fundamentals.profit_margin),
         "share of each sales dollar kept as profit"),
        ("Return on equity", rate(fundamentals.return_on_equity),
         "profit per $1 shareholders have put in"),
        ("Debt vs equity", ratio(fundamentals.debt_to_equity, 2),
         "above 1 means more debt than equity"),
        ("Dividend yield", rate(fundamentals.dividend_yield, 2),
         "annual cash paid out, as a share of the price"),
    ]
    table = Table(
        title="The numbers", box=box.SIMPLE_HEAD, title_justify="left", expand=True
    )
    table.add_column("Measure", style="bold", ratio=2)
    table.add_column("Value", justify="right", ratio=1)
    table.add_column("What it means", ratio=4)
    for label, value, meaning in rows:
        style = "dim" if value == "not available" else ""
        table.add_row(label, value, meaning, style=style)
    return table


def _peers_table(bundle: EvidenceBundle):
    """The subject beside its sector peers' medians."""
    from rich import box
    from rich.table import Table

    if not bundle.peers:
        return None

    rate_metrics = {"profit_margin", "revenue_growth", "dividend_yield", "return_on_equity"}
    table = Table(
        title="Compared with similar companies", box=box.SIMPLE_HEAD,
        title_justify="left", expand=True,
    )
    table.add_column("Measure", style="bold", ratio=2)
    table.add_column(bundle.ticker, justify="right", ratio=1)
    table.add_column("Typical peer", justify="right", ratio=1)
    table.add_column("Difference", justify="right", ratio=1)

    for peer in bundle.peers:
        if peer.subject_value is None:
            table.add_row(peer.label, "not available", "-", "-", style="dim")
            continue
        is_rate = peer.metric in rate_metrics
        subject = f"{peer.subject_value * 100:.1f}%" if is_rate else f"{peer.subject_value:.2f}"
        median = f"{peer.peer_median * 100:.1f}%" if is_rate else f"{peer.peer_median:.2f}"
        if peer.vs_median_pct is None:
            difference, style = "-", ""
        else:
            difference = f"{peer.vs_median_pct:+.0f}%"
            # Neutral colouring: "higher" is not "better" for a P/E, and the
            # tool does not rank companies. Only the magnitude is emphasised.
            style = "bold" if abs(peer.vs_median_pct) > 40 else ""
        table.add_row(peer.label, subject, median, difference, style=style)
    return table


def _print_snapshot(bundle: EvidenceBundle, console) -> None:
    """Render a company snapshot: identity, figures, peers, price, news."""
    from rich.panel import Panel

    fundamentals = bundle.fundamentals
    if fundamentals is not None:
        name = fundamentals.company_name or bundle.ticker
        descriptor = (
            f"{name} ({bundle.ticker})"
            + (f" -- {fundamentals.sector}" if fundamentals.sector else "")
        )
        if fundamentals.is_profitable is False:
            descriptor += "\n\n[yellow]This company currently loses money, which changes how "
            descriptor += "every figure below should be read.[/yellow]"
        console.print(
            Panel(descriptor, title="What you are looking at", border_style="cyan", expand=True)
        )

    for builder in (_fundamentals_table, _peers_table):
        table = builder(bundle)
        if table is not None:
            console.print(table)

    if bundle.trend is not None:
        console.print(
            Panel(bundle.trend.summary, title="Price history", border_style="blue", expand=True)
        )

    if bundle.news:
        console.print("\n[bold]Recent headlines[/bold] [dim](titles only, not read)[/dim]")
        for item in bundle.news:
            when = f" ({item.published.isoformat()})" if item.published else ""
            console.print(f"  • {item.title} [dim]-- {item.publisher}{when}[/dim]")


def print_bundle(bundle: EvidenceBundle, report: str, plain: bool = False) -> None:
    """Render a bundle and its report to the terminal.

    `report` is the already-validated prose. The plain path prints it
    verbatim, so what the user reads is exactly what the validator checked;
    the rich path lays the same bundle fields out as panels and tables and
    adds no figure the report does not contain.
    """
    if plain or not console_available():
        print(report)
        return

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    observation = bundle.observation

    is_snapshot = bundle.question_type == "snapshot"
    header = f"[bold]{bundle.ticker}[/bold]  ·  {bundle.period_description}"
    if bundle.is_synthetic:
        header += "  ·  [bold red]SYNTHETIC DATA -- not real prices[/bold red]"
    console.print(
        Panel(
            header,
            title="Company snapshot" if is_snapshot else "Research note",
            border_style="blue",
            expand=True,
        )
    )

    if is_snapshot:
        _print_snapshot(bundle, console)

    if observation is not None:
        tone = "yellow" if observation.is_unusual else "cyan"
        console.print(
            Panel(
                observation.plain_summary,
                title="What happened",
                border_style=tone,
                expand=True,
            )
        )

    if not is_snapshot:
        for mechanical in (False, True):
            table = _factor_table(bundle, mechanical)
            if table is not None:
                console.print(table)

    if bundle.macro_releases:
        console.print("\n[bold]Economic data released around this day[/bold]")
        for release in bundle.macro_releases:
            console.print(f"  • {release.description}")

    if bundle.counterevidence:
        body = "\n\n".join(
            f"[bold]{item.label}[/bold]\n{item.detail}" for item in bundle.counterevidence
        )
        console.print(
            Panel(
                body,
                title=(
                    "What this does not tell you" if is_snapshot else "Competing explanations"
                ),
                border_style="magenta",
                expand=True,
            )
        )

    if bundle.limitations:
        body = "\n".join(f"• {limitation}" for limitation in bundle.limitations)
        console.print(Panel(body, title="Limitations", border_style="dim", expand=True))

    if bundle.warnings:
        body = "\n".join(f"• {warning}" for warning in bundle.warnings)
        console.print(Panel(body, title="Assumptions made", border_style="yellow", expand=True))

    closing = (
        f"This has not said whether to own {bundle.ticker}. The same figures point "
        "different ways depending on how long you plan to hold, what else you own, and "
        "what you would do if it fell by half -- none of which appear above. The "
        "evidence is here; the decision is a separate thing, and it is yours."
        if is_snapshot
        else "This is a summary of evidence, not investment advice. Nothing above "
        "identifies what made the move happen; every factor listed is something that "
        "happened at the same time."
    )
    console.print(Panel(closing, border_style="dim", expand=True))
    console.print(f"[dim]Sources: {'; '.join(bundle.data_sources)}[/dim]")


def print_validation(validation, plain: bool = False) -> None:
    """Show the validator's verdict, loudly when it fails."""
    if plain or not console_available():
        print(f"\n[validator] {validation.summary()}")
        for issue in validation.issues:
            print(f"  {issue}")
        return

    from rich.console import Console

    console = Console()
    if validation.ok and not validation.warnings:
        console.print(f"[dim]Validator: {validation.summary()}[/dim]")
        return
    style = "red" if not validation.ok else "yellow"
    console.print(f"[{style}]Validator: {validation.summary()}[/{style}]")
    for issue in validation.issues:
        console.print(f"  [{style}]{issue}[/{style}]")


def print_error(message: str, suggestion: str = "", plain: bool = False) -> None:
    """Show a user-facing error. Never a traceback."""
    if plain or not console_available():
        print(f"Error: {message}")
        if suggestion:
            print(suggestion)
        return

    from rich.console import Console
    from rich.panel import Panel

    body = message if not suggestion else f"{message}\n\n{suggestion}"
    Console().print(Panel(body, title="Could not answer that", border_style="red", expand=True))


# --- backtests -------------------------------------------------------------

def _plain_backtest(outcome, symbol: str, walk_forward_mode: bool, chart_path) -> None:
    """Fixed-width fallback, used when rich is unavailable or --plain is set."""
    print("=" * 78)
    label = outcome.template if hasattr(outcome, "template") else outcome.strategy_name
    print(f"  BACKTEST -- {symbol} -- {label}".rstrip())
    print("=" * 78)
    strategy, benchmark = outcome.stats, outcome.benchmark_stats
    print(f"  {'':<24}{'strategy':>14}{'buy & hold':>14}")
    for (label, value), (_, benchmark_value) in zip(
        strategy.summary_rows(), benchmark.summary_rows()
    ):
        print(f"  {label:<24}{value:>14}{benchmark_value:>14}")
    print()
    print(f"  Excess return vs buying and holding: {outcome.excess_return_pct:+.2f} points")
    if chart_path is not None:
        print(f"  Chart: {chart_path}")


def print_backtest(
    outcome,
    symbol: str,
    walk_forward_mode: bool,
    chart_path=None,
    is_synthetic: bool = False,
    warnings: list[str] | None = None,
    plain: bool = False,
) -> None:
    """Render a backtest or walk-forward result.

    The strategy and the benchmark are shown side by side in one table, never
    the strategy alone. A return figure with nothing beside it invites the
    reader to judge it against zero, and the question that matters is whether
    the rule beat simply owning the thing.
    """
    if plain or not console_available():
        _plain_backtest(outcome, symbol, walk_forward_mode, chart_path)
        return

    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    # getattr's default argument is evaluated eagerly, so the obvious
    # one-liner would touch .strategy_name on a WalkForwardResult that has
    # no such attribute.
    label = outcome.template if hasattr(outcome, "template") else outcome.strategy_name
    header = f"[bold]{symbol}[/bold]  ·  {label}"
    header += (
        "  ·  walk-forward (out-of-sample)" if walk_forward_mode else "  ·  single pass"
    )
    if is_synthetic:
        header += "  ·  [bold red]SYNTHETIC PRICES -- not real[/bold red]"
    console.print(Panel(header, title="Backtest", border_style="blue", expand=True))

    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Measure", style="bold", ratio=2)
    table.add_column("This rule", justify="right", ratio=1)
    table.add_column("Buy and hold", justify="right", ratio=1)
    for (label, value), (_, benchmark_value) in zip(
        outcome.stats.summary_rows(), outcome.benchmark_stats.summary_rows()
    ):
        table.add_row(label, value, benchmark_value)
    console.print(table)

    excess = outcome.excess_return_pct
    verdict_style = "green" if excess > 0 else "yellow"
    direction = "ahead of" if excess > 0 else "behind"
    lines = [
        (
            f"[{verdict_style}]The rule finished {abs(excess):.2f} percentage points "
            f"{direction} simply buying and holding.[/{verdict_style}]"
        )
    ]
    if hasattr(outcome, "folds"):
        lines.append(
            f"Out of {len(outcome.folds)} out-of-sample periods, it beat buying and "
            f"holding in {outcome.folds_beating_benchmark}."
        )
        lines.append(f"Parameter choice was {outcome.parameter_stability}.")
    lines.append(
        f"Trading costs came to {outcome.total_costs:,.2f}, which is already subtracted "
        "from every figure above."
    )
    console.print(Panel("\n".join(lines), title="What this means", border_style=verdict_style, expand=True))

    for warning in warnings or []:
        console.print(f"[yellow]Note: {warning}[/yellow]")
    for note in getattr(outcome, "notes", [])[:5]:
        console.print(f"[dim]{note}[/dim]")

    caveat = (
        "A backtest shows what a rule would have returned on data that has already "
        "happened, which is not what it will return next. These rules are textbook "
        "examples published decades ago, not edges: if one of them looks profitable "
        "here, the most likely explanation is this particular history, not a "
        "discovery."
    )
    if not walk_forward_mode:
        caveat += (
            " This run used one fixed parameter set over the whole period, so the "
            "figures are in-sample and flatter the rule. Re-run with --walk-forward "
            "for an out-of-sample estimate."
        )
    console.print(Panel(caveat, border_style="dim", expand=True))
    if chart_path is not None:
        console.print(f"[dim]Chart written to {chart_path}[/dim]")
