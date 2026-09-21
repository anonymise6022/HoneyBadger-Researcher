"""Desktop application: the same research tool, in a window you can click.

A Textual app rather than a native-widget one, because the terminal
aesthetic is the point -- dense, monospaced, information-first -- while the
raw CLI is not something you can hand to somebody who has never used a
shell. This keeps the look and adds an input box, tabs, scrolling, a
progress indicator and clickable examples.

**Everything runs on a worker thread.** Fetching four years of history for
a dozen symbols takes seconds, and Textual's event loop drives the UI: any
blocking call inside a handler freezes the window, including the spinner
meant to reassure the user something is happening. `@work(thread=True)`
moves the work off the loop and `call_from_thread` marshals results back.

**No new formatting lives here.** The panels and tables come from the same
builders the CLI uses, so the window and the terminal cannot disagree about
what the evidence says -- and the validator still runs against the rendered
text before anything is displayed.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, ClassVar

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from .backtest.engine import BacktestConfig
from .backtest.plots import plot_backtest
from .backtest.strategies import STRATEGY_TEMPLATES
from .backtest.walk_forward import WalkForwardConfig, walk_forward
from .data.price_ingest import PriceDataError, UnknownSymbolError, fetch_history
from .display.formatting import _factor_table, _fundamentals_table, _peers_table
from .evidence.snapshot import build_snapshot_bundle
from .evidence.why_moved import build_why_moved_bundle
from .query_parser import QueryParseError, parse_query
from .settings import describe_credentials
from .synthesis.snapshot_report import SNAPSHOT_SECTIONS, render_snapshot_report
from .synthesis.template_report import REQUIRED_SECTIONS, render_report
from .synthesis.validator import validate_report

__all__ = ["ResearchApp", "main"]

EXAMPLE_QUERIES = [
    "why did SPY fall today",
    "why did TSLA drop yesterday",
    "is AAPL good to invest",
    "why did the stock market fall this week",
    "should i buy KO",
]


class ResearchApp(App):
    """The main window."""

    TITLE = "research"
    SUB_TITLE = "evidence, not advice"

    CSS = """
    Screen { background: #101014; }

    #query-row { height: auto; padding: 1 2 0 2; }
    #query { width: 1fr; }
    #run { width: 12; margin-left: 1; }

    #examples { height: auto; padding: 0 2 1 2; color: #898781; }

    #backtest-row { height: auto; padding: 1 2; }
    #ticker { width: 20; }
    #strategy { width: 34; margin-left: 1; }
    #run-backtest { width: 14; margin-left: 1; }

    #results { padding: 0 2 1 2; height: 1fr; }
    #status { height: auto; padding: 0 2; color: #898781; }

    LoadingIndicator { height: 3; }
    .hidden { display: none; }
    """

    BINDINGS: ClassVar = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+k", "focus_query", "Search"),
    ]

    def __init__(self, mock: bool = False) -> None:
        super().__init__()
        self.mock = mock
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="research-tab"):
            with TabPane("Research", id="research-tab"):
                with Horizontal(id="query-row"):
                    yield Input(
                        placeholder='e.g. why did SPY fall today',
                        id="query",
                    )
                    yield Button("Run", id="run", variant="primary")
                yield Label(
                    "Try:  " + "   ".join(f"[@click=app.use_example('{q}')]{q}[/]"
                                          for q in EXAMPLE_QUERIES[:3]),
                    id="examples",
                    markup=True,
                )
            with TabPane("Backtest", id="backtest-tab"), Horizontal(id="backtest-row"):
                yield Input(placeholder="ticker, e.g. SPY", id="ticker")
                yield Select(
                    [
                        (f"{name} -- {meta['description']}", name)
                        for name, meta in sorted(STRATEGY_TEMPLATES.items())
                    ],
                    id="strategy",
                    value="ma-crossover",
                    allow_blank=False,
                )
                yield Button("Run", id="run-backtest", variant="primary")
        yield Static("", id="status")
        yield LoadingIndicator(id="spinner", classes="hidden")
        yield VerticalScroll(Static(self._welcome(), id="output"), id="results")
        yield Footer()

    # --- helpers ----------------------------------------------------------

    def _welcome(self) -> RenderableType:
        credentials = describe_credentials()
        missing = [c for c in credentials if not c.present]
        lines = [
            Text("Ask why something moved, or whether a company is worth a look.", style="bold"),
            Text(""),
            Text("This tool never tells you a cause and never tells you what to buy.", style="#898781"),
            Text("It shows what happened at the same time, how often that has", style="#898781"),
            Text("coincided with moves like this before, and what argues the other way.", style="#898781"),
            Text(""),
        ]
        if missing:
            lines.append(Text("Optional keys not set:", style="yellow"))
            for status in missing:
                lines.append(Text(f"  {status.name} -- unlocks {status.unlocks}", style="#898781"))
            lines.append(Text("  Set them with:  research keys FRED_API_KEY", style="#898781"))
        if self.mock:
            lines.append(Text(""))
            lines.append(Text("OFFLINE MODE -- all figures are synthetic and not real.", style="bold red"))
        return Group(*lines)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        self.query_one("#spinner").set_class(not busy, "hidden")
        self.query_one("#status", Static).update(Text(message, style="#898781"))
        self.query_one("#run", Button).disabled = busy
        self.query_one("#run-backtest", Button).disabled = busy

    def _show(self, renderable: RenderableType) -> None:
        self.query_one("#output", Static).update(renderable)
        self.query_one("#results", VerticalScroll).scroll_home(animate=False)

    def _show_error(self, message: str, suggestion: str = "") -> None:
        body = Text(message, style="bold")
        if suggestion:
            body = Group(body, Text(""), Text(suggestion, style="#898781"))
        self._show(Panel(body, title="Could not answer that", border_style="red"))

    def action_use_example(self, query: str) -> None:
        field = self.query_one("#query", Input)
        field.value = query
        field.focus()
        self.run_research(query)

    def action_clear(self) -> None:
        self.query_one("#query", Input).value = ""
        self._show(self._welcome())

    def action_focus_query(self) -> None:
        self.query_one("#query", Input).focus()

    # --- events -----------------------------------------------------------

    @on(Input.Submitted, "#query")
    def _on_query_submitted(self, event: Input.Submitted) -> None:
        self.run_research(event.value)

    @on(Button.Pressed, "#run")
    def _on_run_pressed(self) -> None:
        self.run_research(self.query_one("#query", Input).value)

    @on(Input.Submitted, "#ticker")
    @on(Button.Pressed, "#run-backtest")
    def _on_backtest_pressed(self) -> None:
        ticker = self.query_one("#ticker", Input).value.strip().upper()
        strategy = self.query_one("#strategy", Select).value
        if not ticker:
            self._show_error("Type a ticker first.", "For example: SPY, AAPL, KO")
            return
        self.run_backtest_job(ticker, str(strategy))

    # --- work -------------------------------------------------------------

    def run_research(self, query: str) -> None:
        if self._busy:
            return
        if not query.strip():
            self._show_error("Type a question first.", f"For example: {EXAMPLE_QUERIES[0]}")
            return
        self._set_busy(True, "Gathering evidence...")
        self._research_worker(query.strip())

    @work(thread=True, exclusive=True)
    def _research_worker(self, query: str) -> None:
        """Blocking data work, off the event loop so the window stays alive."""
        try:
            parsed = parse_query(query)
        except QueryParseError as exc:
            self.call_from_thread(self._finish_error, str(exc), exc.suggestion)
            return

        source = "mock" if self.mock else "auto"
        try:
            if parsed.question_type == "snapshot":
                bundle = build_snapshot_bundle(parsed, source=source)
                report = render_snapshot_report(bundle)
                sections = SNAPSHOT_SECTIONS
            else:
                bundle = build_why_moved_bundle(parsed, source=source)
                report = render_report(bundle)
                sections = REQUIRED_SECTIONS
        except UnknownSymbolError as exc:
            self.call_from_thread(self._finish_error, str(exc), exc.suggestion)
            return
        except PriceDataError as exc:
            self.call_from_thread(self._finish_error, str(exc), exc.suggestion)
            return
        except Exception as exc:  # noqa: BLE001 - a window must not show a traceback
            self.call_from_thread(
                self._finish_error,
                f"Something went wrong ({type(exc).__name__}: {exc})",
                "Try again, or run with offline mode to check the tool itself.",
            )
            return

        validation = validate_report(report, bundle, required_sections=sections)
        self.call_from_thread(self._finish_research, bundle, validation)

    def _finish_error(self, message: str, suggestion: str) -> None:
        self._set_busy(False)
        self._show_error(message, suggestion)

    def _finish_research(self, bundle: Any, validation: Any) -> None:
        self._set_busy(
            False,
            f"{validation.numbers_matched}/{validation.numbers_checked} figures traced "
            f"to the evidence"
            + ("" if validation.ok else "  --  VALIDATION FAILED"),
        )
        self._show(self._render_bundle(bundle))

    def _render_bundle(self, bundle: Any) -> RenderableType:
        """Lay a bundle out as panels and tables, reusing the CLI's builders."""
        blocks: list[RenderableType] = []
        header = Text(f"{bundle.ticker}  ·  {bundle.period_description}", style="bold")
        if bundle.is_synthetic:
            header.append("   SYNTHETIC DATA -- not real", style="bold red")
        blocks.append(
            Panel(
                header,
                title="Company snapshot" if bundle.question_type == "snapshot" else "Research note",
                border_style="blue",
            )
        )

        if bundle.observation is not None:
            blocks.append(
                Panel(
                    bundle.observation.plain_summary,
                    title="What happened",
                    border_style="yellow" if bundle.observation.is_unusual else "cyan",
                )
            )

        if bundle.question_type == "snapshot":
            fundamentals = bundle.fundamentals
            if fundamentals is not None:
                name = fundamentals.company_name or bundle.ticker
                descriptor = Text(f"{name} ({bundle.ticker})")
                if fundamentals.sector:
                    descriptor.append(f" -- {fundamentals.sector}", style="#898781")
                if fundamentals.is_profitable is False:
                    descriptor.append(
                        "\n\nThis company currently loses money, which changes how every "
                        "figure below should be read.",
                        style="yellow",
                    )
                blocks.append(Panel(descriptor, title="What you are looking at", border_style="cyan"))
            for builder in (_fundamentals_table, _peers_table):
                table = builder(bundle)
                if table is not None:
                    blocks.append(table)
            if bundle.trend is not None:
                blocks.append(Panel(bundle.trend.summary, title="Price history", border_style="blue"))
            if bundle.news:
                news = Table.grid(padding=(0, 1))
                news.add_column()
                for item in bundle.news:
                    when = f" ({item.published.isoformat()})" if item.published else ""
                    news.add_row(Text(f"• {item.title}").append(
                        f"  {item.publisher}{when}", style="#898781"
                    ))
                blocks.append(Panel(news, title="Recent headlines (titles only, not read)",
                                    border_style="#898781"))
        else:
            for mechanical in (False, True):
                table = _factor_table(bundle, mechanical)
                if table is not None:
                    blocks.append(table)
            if bundle.macro_releases:
                releases = Table.grid(padding=(0, 1))
                releases.add_column()
                for release in bundle.macro_releases:
                    releases.add_row(Text(f"• {release.description}"))
                blocks.append(Panel(releases, title="Economic data released around this day",
                                    border_style="#898781"))

        if bundle.counterevidence:
            body = Group(
                *[
                    Group(Text(item.label, style="bold"), Text(item.detail), Text(""))
                    for item in bundle.counterevidence
                ]
            )
            blocks.append(
                Panel(
                    body,
                    title=(
                        "What this does not tell you"
                        if bundle.question_type == "snapshot"
                        else "Competing explanations"
                    ),
                    border_style="magenta",
                )
            )

        if bundle.limitations:
            blocks.append(
                Panel(
                    Group(*[Text(f"• {text}") for text in bundle.limitations]),
                    title="Limitations",
                    border_style="#898781",
                )
            )
        if bundle.warnings:
            blocks.append(
                Panel(
                    Group(*[Text(f"• {text}") for text in bundle.warnings]),
                    title="Assumptions made",
                    border_style="yellow",
                )
            )

        closing = (
            f"This has not said whether to own {bundle.ticker}. How long you plan to hold, "
            "what else you own, and what you would do if it fell by half all change the "
            "answer, and none of them appear above. The evidence is here; the decision is "
            "yours."
            if bundle.question_type == "snapshot"
            else "This is a summary of evidence, not investment advice. Nothing above "
            "identifies what made the move happen; every factor listed is something that "
            "happened at the same time."
        )
        blocks.append(Panel(Text(closing, style="#898781"), border_style="#898781"))
        return Group(*blocks)

    # --- backtest ---------------------------------------------------------

    def run_backtest_job(self, ticker: str, strategy: str) -> None:
        if self._busy:
            return
        self._set_busy(True, f"Backtesting {strategy} on {ticker}...")
        self._backtest_worker(ticker, strategy)

    @work(thread=True, exclusive=True)
    def _backtest_worker(self, ticker: str, strategy: str) -> None:
        source = "mock" if self.mock else "auto"
        end = date.today()  # noqa: DTZ011
        start = end - timedelta(days=int(12 * 365.25))
        try:
            frame, price_source, _ = fetch_history(ticker, start, end, source)
            outcome = walk_forward(frame, strategy, WalkForwardConfig(), BacktestConfig())
        except UnknownSymbolError as exc:
            self.call_from_thread(self._finish_error, str(exc), exc.suggestion)
            return
        except (PriceDataError, ValueError) as exc:
            self.call_from_thread(
                self._finish_error, str(exc), "Try a longer history or another symbol."
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(
                self._finish_error, f"Backtest failed ({type(exc).__name__}: {exc})", ""
            )
            return

        from .cli import RESULTS_DIR

        chart = plot_backtest(
            outcome.equity,
            outcome.benchmark_equity,
            RESULTS_DIR / f"backtest_{ticker}_{strategy}.png",
            title=f"{ticker} -- {strategy}, walk-forward out-of-sample",
            subtitle=f"{len(outcome.folds)} folds",
        )
        self.call_from_thread(
            self._finish_backtest, outcome, ticker, chart, price_source == "mock"
        )

    def _finish_backtest(self, outcome: Any, ticker: str, chart: Any, synthetic: bool) -> None:
        self._set_busy(False, f"Chart written to {chart}" if chart else "")
        header = Text(f"{ticker}  ·  {outcome.template}  ·  walk-forward", style="bold")
        if synthetic:
            header.append("   SYNTHETIC PRICES -- not real", style="bold red")

        table = Table(expand=True)
        table.add_column("Measure", style="bold", ratio=2)
        table.add_column("This rule", justify="right", ratio=1)
        table.add_column("Buy and hold", justify="right", ratio=1)
        for (label, value), (_, benchmark) in zip(
            outcome.stats.summary_rows(), outcome.benchmark_stats.summary_rows()
        ):
            table.add_row(label, value, benchmark)

        excess = outcome.excess_return_pct
        verdict = Text(
            f"The rule finished {abs(excess):.2f} percentage points "
            f"{'ahead of' if excess > 0 else 'behind'} simply buying and holding.",
            style="green" if excess > 0 else "yellow",
        )
        detail = Text(
            f"\nIt beat buying and holding in {outcome.folds_beating_benchmark} of "
            f"{len(outcome.folds)} out-of-sample periods.\n"
            f"Parameter choice was {outcome.parameter_stability}.\n"
            f"Trading costs of {outcome.total_costs:,.2f} are already subtracted."
        )
        caveat = Text(
            "A backtest shows what a rule would have returned on data that has already "
            "happened, which is not what it will return next. These are textbook rules "
            "published decades ago, not edges.",
            style="#898781",
        )
        self._show(
            Group(
                Panel(header, title="Backtest", border_style="blue"),
                table,
                Panel(Group(verdict, detail), title="What this means",
                      border_style="green" if excess > 0 else "yellow"),
                Panel(caveat, border_style="#898781"),
            )
        )


def main() -> None:
    """Launch the application."""
    import sys

    ResearchApp(mock="--mock" in sys.argv).run()


if __name__ == "__main__":
    main()
