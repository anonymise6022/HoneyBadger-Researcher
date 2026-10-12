"""Command-line entry point.

    python -m research_cli "why did SPY fall today"
    python -m research_cli "is AAPL good to invest"

Errors here are the user's first experience of the tool, so no path in this
module is allowed to surface a traceback. Every expected failure -- a
mistyped ticker, a date before the symbol listed, a dead network -- is
caught and turned into a sentence plus a suggestion.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer

from .backtest.engine import BacktestConfig, run_backtest
from .backtest.plots import plot_backtest
from .backtest.strategies import STRATEGY_TEMPLATES, build_strategy
from .backtest.walk_forward import WalkForwardConfig, walk_forward
from .data.price_ingest import (
    PriceDataError,
    Source,
    UnknownSymbolError,
    fetch_history,
)
from .display.formatting import (
    console_available,
    print_backtest,
    print_bundle,
    print_error,
    print_validation,
)
from .evidence.outlook import build_outlook_bundle
from .evidence.snapshot import build_snapshot_bundle
from .evidence.why_moved import DEFAULT_HISTORY_YEARS, build_why_moved_bundle
from .debate.runner import run_debate_flow
from .display.debate_display import print_debate_report
from .explain import CLI_DEPTH_LABELS, DEPTH_ALIASES, resolve_depth
from .planner.command import plan_command
from .planner.interactive import offer_plan
from .synthesis.llm_budget import DEFAULT_MAX_CALLS
from .synthesis.llm_provider import PROVIDER_NAMES
from .query_parser import QueryParseError, parse_query
from .settings import CONFIG_PATH, KNOWN_KEYS, clear_key, describe_credentials, set_key
from .synthesis.llm_report import (
    LLMUnavailable,
    generate_report,
    llm_available,
    resolve_backend,
)
from .synthesis.outlook_report import render_outlook_report, sections_for
from .synthesis.snapshot_report import render_snapshot_report
from .synthesis.template_report import render_report
from .synthesis.validator import validate_report

_EXAMPLES = """
Examples (you can type a question directly -- no sub-command needed):

  python -m research_cli "why did SPY fall today"
  python -m research_cli "what happened to TSLA yesterday"
  python -m research_cli "why did NVDA drop on 2026-09-15"
  python -m research_cli "why did the stock market fall this week"
  python -m research_cli "why did EUR/USD rally today"
  python -m research_cli "is AAPL good to invest"
  python -m research_cli "should i buy TSLA"

Testing a trading rule against buying and holding:

  python -m research_cli backtest SPY --strategy ma-crossover
  python -m research_cli backtest AAPL --strategy rsi --years 8
  python -m research_cli backtest QQQ -s buy-and-hold --no-walk-forward

Windowed version (same tool, with tabs and an input box):

  python -m research_cli app

Useful flags:

  --depth    beginner | intermediate | analyst -- how much is explained
  --mock     work offline on synthetic data (no network, no keys)
  --plain    no colour or tables, easy to pipe or paste
  --json f   also write the evidence bundle to a file
  --llm      write the note with Claude (needs ANTHROPIC_API_KEY)

Tip: alias it to something shorter --
  alias research='python -m research_cli'
"""


#: Charts and exports land here regardless of where the command is run
#: from. A relative default would scatter `research_cli/results/` folders
#: through whatever directory the user happened to be standing in.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Research a stock move or a company, using evidence you can check.",
    # The examples go on the top-level help too, not only on `ask`. A first
    # run is almost always `--help`, and a help screen that lists two
    # sub-command names teaches nothing about what to type.
    epilog=_EXAMPLES,
)

def _progress(message: str, plain: bool):
    """A spinner when rich is available, otherwise a plain status line."""
    if plain or not console_available():
        class _Null:
            def __enter__(self):
                print(f"{message} ...", file=sys.stderr)
                return self

            def __exit__(self, *exc):
                return False

        return _Null()

    from rich.console import Console

    return Console(stderr=True).status(f"[dim]{message}...[/dim]", spinner="dots")


@app.command(epilog=_EXAMPLES)
def ask(
    query: Annotated[
        list[str], typer.Argument(help='Your question, e.g. "why did SPY fall today".')
    ],
    mock: Annotated[
        bool, typer.Option("--mock", help="Use synthetic data only. No network, no API keys.")
    ] = False,
    live: Annotated[
        bool,
        typer.Option("--live", help="Fail loudly instead of falling back to synthetic data."),
    ] = False,
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain text; no colour or tables.")
    ] = False,
    years: Annotated[
        float,
        typer.Option("--years", min=1.0, max=20.0, help="Years of history used for hit rates."),
    ] = DEFAULT_HISTORY_YEARS,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help='Treat this date as "today" (YYYY-MM-DD).')
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the evidence bundle to this file.")
    ] = None,
    show_validation: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Check every figure traces to the evidence.",
        ),
    ] = True,
    llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Write the note with Claude in plainer language. Needs ANTHROPIC_API_KEY.",
        ),
    ] = False,
    compare: Annotated[
        bool,
        typer.Option("--compare", help="Show the template and Claude versions side by side."),
    ] = False,
    depth: Annotated[
        str,
        typer.Option(
            "--depth", "-d",
            help="How much to explain: " + ", ".join([*CLI_DEPTH_LABELS, *DEPTH_ALIASES]) + ".",
        ),
    ] = "beginner",
    debate: Annotated[
        bool,
        typer.Option(
            "--debate",
            help="Let several AI agents argue over the evidence and show the strongest "
            "case or cases, with 'what if' scenarios. Uses your own Claude.",
        ),
    ] = False,
    provider: Annotated[
        str,
        typer.Option("--provider", help="Model for --debate: " + ", ".join(PROVIDER_NAMES) + "."),
    ] = "auto",
    max_calls: Annotated[
        int,
        typer.Option("--max-calls", min=1, max=40, help="Most model calls --debate may make."),
    ] = DEFAULT_MAX_CALLS,
) -> None:
    """Answer a research question about a ticker."""
    if mock and live:
        print_error(
            "--mock and --live contradict each other",
            "--mock forces synthetic data; --live forbids it. Pick one.",
            plain,
        )
        raise typer.Exit(code=2)

    try:
        depth = resolve_depth(depth)
    except ValueError as exc:
        print_error(str(exc), "", plain)
        raise typer.Exit(code=2) from None
    if provider not in PROVIDER_NAMES:
        print_error(f"{provider!r} is not a provider", "Choose one of: " + ", ".join(PROVIDER_NAMES), plain)
        raise typer.Exit(code=2)

    source: Source = "mock" if mock else "live" if live else "auto"
    text = " ".join(query).strip()

    today = date.today()  # noqa: DTZ011
    if as_of:
        try:
            today = date.fromisoformat(as_of)
        except ValueError:
            print_error(
                f"{as_of!r} is not a valid date",
                "Use the form YYYY-MM-DD, for example --as-of 2026-09-15.",
                plain,
            )
            raise typer.Exit(code=2) from None

    try:
        parsed = parse_query(text, today=today)
    except QueryParseError as exc:
        print_error(str(exc), exc.suggestion, plain)
        raise typer.Exit(code=2) from None

    question = parsed.question_type
    if debate and question == "snapshot":
        question = "outlook"  # a debate needs the scenarios an outlook bundle carries
    elif debate and question == "attribution":
        print_error(
            "--debate answers questions about a company or its future, not why a price moved",
            'Try: research "is TSLA going to go up" --debate. Showing the normal report.',
            plain,
        )
        debate = False
    try:
        with _progress(f"Gathering evidence for {parsed.ticker}", plain):
            if question == "snapshot":
                bundle = build_snapshot_bundle(parsed, source=source)
            elif question == "outlook":
                bundle = build_outlook_bundle(parsed, source=source)
            else:
                bundle = build_why_moved_bundle(parsed, source=source, history_years=years)
    except UnknownSymbolError as exc:
        print_error(str(exc), exc.suggestion, plain)
        raise typer.Exit(code=4) from None
    except PriceDataError as exc:
        print_error(str(exc), exc.suggestion, plain)
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 - never show a traceback to a user
        print_error(
            f"something went wrong while gathering data ({type(exc).__name__}: {exc})",
            "Try --mock to check the tool itself is working, or report this.",
            plain,
        )
        raise typer.Exit(code=1) from None

    sections = sections_for(question)
    if question == "snapshot":
        template = render_snapshot_report(bundle)
    elif question == "outlook":
        template = render_outlook_report(bundle, depth)
    else:
        template = render_report(bundle, depth)
    report = template
    validation = validate_report(template, bundle, required_sections=sections)

    if debate:
        with _progress(f"Agents are debating {parsed.ticker} (at most {max_calls} model calls)", plain):
            run = run_debate_flow(bundle, provider, max_calls, depth)
        print_debate_report(run.report.text, run.notes, run.usage, run.report.reason, plain)
        if show_validation:
            print_validation(run.report.validation, plain=plain)
        _write_json(bundle, json_out, plain)
        offer_plan(bundle, plain, source=source, provider=run.provider, depth=depth, as_of=today)
        return

    if llm or compare:
        if not llm_available():
            print_error(
                "no model is configured, so the plain template is shown instead",
                "Either download the local model (about 4GB, no key, nothing "
                "leaves this machine) with `research model --download`, or "
                "set ANTHROPIC_API_KEY. Everything else works without either.",
                plain,
            )
        else:
            backend = resolve_backend()
            writer = "a local model" if backend == "local" else "Claude"
            try:
                with _progress(f"Writing the note with {writer}", plain):
                    result = generate_report(bundle)
            except LLMUnavailable as exc:
                print_error(
                    f"could not reach the model ({exc}); showing the plain "
                    "template instead",
                    "",
                    plain,
                )
            else:
                if result.usable:
                    if compare:
                        print("=" * 78)
                        print("  TEMPLATE VERSION")
                        print("=" * 78)
                        print(template)
                        print("=" * 78)
                        print(
                            f"  WRITTEN VERSION ({result.model}, "
                            f"{result.attempts} attempt(s))"
                        )
                        print("=" * 78)
                    report, validation = result.report, result.validation
                else:
                    # Never show prose that failed the numeric trace: a
                    # confident, wrong number is the failure this whole
                    # pipeline exists to prevent.
                    print_error(
                        "Claude's draft did not pass the evidence check, so the plain "
                        "template is shown instead",
                        "\n".join(f"  {note}" for note in result.repair_notes[:4]),
                        plain,
                    )

    print_bundle(bundle, report, plain=plain)

    if show_validation:
        print_validation(validation, plain=plain)
    _write_json(bundle, json_out, plain)


def _write_json(bundle, json_out: Path | None, plain: bool) -> None:
    if json_out is None:
        return
    try:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(bundle.model_dump_json(indent=2))
        print(f"Evidence bundle written to {json_out}")
    except OSError as exc:
        print_error(f"could not write {json_out} ({exc})", "", plain)


@app.command(name="backtest")
def backtest_command(
    ticker: Annotated[str, typer.Argument(help="Symbol to test, e.g. SPY.")],
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy", "-s",
            help="Rule to test: " + ", ".join(sorted(STRATEGY_TEMPLATES)) + ".",
        ),
    ] = "ma-crossover",
    years: Annotated[
        float, typer.Option("--years", min=2.0, max=30.0, help="Years of history to test.")
    ] = 12.0,
    walk_forward_mode: Annotated[
        bool,
        typer.Option(
            "--walk-forward/--no-walk-forward",
            help="Re-choose parameters on each training window and score on the next "
            "period. Without it, one fixed setting runs over the whole history, which "
            "flatters the rule.",
        ),
    ] = True,
    train_bars: Annotated[
        int, typer.Option("--train-bars", min=60, help="Bars per training window.")
    ] = 504,
    test_bars: Annotated[
        int, typer.Option("--test-bars", min=20, help="Bars per out-of-sample window.")
    ] = 126,
    cash: Annotated[
        float, typer.Option("--cash", min=100.0, help="Starting portfolio value.")
    ] = 10_000.0,
    slippage_bps: Annotated[
        float,
        typer.Option("--slippage-bps", min=0.0, help="Assumed slippage, in basis points."),
    ] = 5.0,
    mock: Annotated[bool, typer.Option("--mock", help="Use synthetic prices.")] = False,
    plain: Annotated[bool, typer.Option("--plain", help="Plain text output.")] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Where to write the chart. Defaults to the "
                     "package's own results folder, so it works from any directory."),
    ] = None,
) -> None:
    """Test a trading rule against buying and holding."""
    if strategy not in STRATEGY_TEMPLATES:
        print_error(
            f"unknown strategy {strategy!r}",
            "Available: " + ", ".join(sorted(STRATEGY_TEMPLATES)),
            plain,
        )
        raise typer.Exit(code=2)

    symbol = ticker.strip().upper()
    source: Source = "mock" if mock else "auto"
    end = date.today()  # noqa: DTZ011
    start = end - timedelta(days=int(years * 365.25) + 40)

    try:
        with _progress(f"Fetching {symbol} price history", plain):
            frame, price_source, fetch_warnings = fetch_history(symbol, start, end, source)
    except UnknownSymbolError as exc:
        print_error(str(exc), exc.suggestion, plain)
        raise typer.Exit(code=4) from None
    except PriceDataError as exc:
        print_error(str(exc), exc.suggestion, plain)
        raise typer.Exit(code=1) from None

    execution = BacktestConfig(initial_cash=cash, slippage_bps=slippage_bps)
    try:
        with _progress("Running the backtest", plain):
            if walk_forward_mode:
                outcome = walk_forward(
                    frame,
                    strategy,
                    WalkForwardConfig(train_bars=train_bars, test_bars=test_bars),
                    execution,
                )
            else:
                outcome = run_backtest(frame, build_strategy(strategy), execution)
    except ValueError as exc:
        print_error(
            f"could not run the backtest: {exc}",
            "Try --years with a larger number, or smaller --train-bars/--test-bars.",
            plain,
        )
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 - never show a traceback to a user
        print_error(
            f"something went wrong while backtesting ({type(exc).__name__}: {exc})",
            "Try --mock to check the tool itself is working.",
            plain,
        )
        raise typer.Exit(code=1) from None

    is_synthetic = price_source == "mock"
    destination = out_dir if out_dir is not None else RESULTS_DIR
    chart = plot_backtest(
        outcome.equity,
        outcome.benchmark_equity,
        destination / f"backtest_{symbol}_{strategy}.png",
        title=f"{symbol} -- {strategy}"
        + (", walk-forward out-of-sample" if walk_forward_mode else ", single pass"),
        subtitle=(
            f"{len(outcome.folds)} folds, parameters re-chosen on each training window"
            if walk_forward_mode
            else "one fixed parameter set over the whole history"
        ),
    )
    print_backtest(
        outcome,
        symbol=symbol,
        walk_forward_mode=walk_forward_mode,
        chart_path=chart,
        is_synthetic=is_synthetic,
        warnings=list(fetch_warnings),
        plain=plain,
    )


@app.command(name="app")
def app_command(
    mock: Annotated[
        bool, typer.Option("--mock", help="Offline mode: synthetic data only.")
    ] = False,
    terminal: Annotated[
        bool,
        typer.Option("--terminal", help="Use the older in-terminal interface instead."),
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable the web inspector.")
    ] = False,
) -> None:
    """Open the windowed version of this tool."""
    if terminal:
        from .app import ResearchApp

        ResearchApp(mock=mock).run()
        return

    from .desktop import launch

    try:
        launch(mock=mock, debug=debug)
    except RuntimeError as exc:
        print_error(
            str(exc),
            "Falling back to the terminal interface: research app --terminal",
            plain=False,
        )
        raise typer.Exit(code=1) from None


@app.command(name="quant")
def quant_command(
    ticker: Annotated[str, typer.Argument(help="Symbol to analyse, e.g. SPY.")],
    mock: Annotated[bool, typer.Option("--mock", help="Use synthetic prices.")] = False,
    plain: Annotated[bool, typer.Option("--plain", help="Plain text output.")] = False,
) -> None:
    """Experimental quantitative diagnostics for a symbol."""
    from .quant import ALPHA_NOTICE, analyse

    symbol = ticker.strip().upper()
    end = date.today()  # noqa: DTZ011
    try:
        with _progress(f"Fetching {symbol}", plain):
            frame, _, _ = fetch_history(
                symbol, end - timedelta(days=1500), end,
                "mock" if mock else "auto",
            )
    except UnknownSymbolError as exc:
        print_error(str(exc), exc.suggestion, plain)
        raise typer.Exit(code=4) from None
    except PriceDataError as exc:
        print_error(str(exc), exc.suggestion, plain)
        raise typer.Exit(code=1) from None

    with _progress("Fitting models", plain):
        result = analyse(frame, symbol)

    typer.echo(f"\n  ALPHA -- {ALPHA_NOTICE}\n")
    if result.volatility is not None:
        v = result.volatility
        typer.echo(f"  Volatility ({v.regime})")
        typer.echo(f"    EWMA                {v.ewma_pct:.2f}%")
        typer.echo(f"    Yang-Zhang          {v.yang_zhang_pct:.2f}%")
        typer.echo(f"    Close-to-close      {v.close_to_close_pct:.2f}%")
        typer.echo(f"    Forecast ({v.forecast_horizon_days}d)      {v.forecast_pct:.2f}%  [{v.forecast_model}]")
        if v.long_run_pct is not None:
            typer.echo(f"    Long-run            {v.long_run_pct:.2f}%")
        for note in v.notes:
            typer.echo(f"    note: {note}")
    if result.regime is not None:
        typer.echo(f"\n  Regime: {result.regime.classification}")
        typer.echo(f"    {result.regime.describe()}")
    if result.mean_reversion is not None:
        typer.echo("\n  Mean reversion")
        typer.echo(f"    {result.mean_reversion.describe()}")
    for failure in result.failures:
        typer.echo(f"\n  could not run -- {failure}")


@app.command(name="keys")
def keys_command(
    name: Annotated[
        str | None,
        typer.Argument(help="Key to set, e.g. FRED_API_KEY. Omit to list what is set."),
    ] = None,
    value: Annotated[
        str | None,
        typer.Argument(help="The key itself. Omit to be prompted without it echoing."),
    ] = None,
    remove: Annotated[bool, typer.Option("--remove", help="Delete this key.")] = False,
) -> None:
    """Show or store API keys.

    Keys are saved to a file rather than an environment variable because a
    double-clicked application does not inherit your shell's environment on
    macOS or Windows. Setting one here makes it work everywhere.
    """
    if name is None:
        typer.echo(f"Credentials file: {CONFIG_PATH}")
        typer.echo("(environment variables, when set, take priority over this file)\n")
        for status in describe_credentials():
            typer.echo(f"  {status.describe()}")
        typer.echo("\nTo add one:  research keys FRED_API_KEY")
        typer.echo("Free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    key_name = name.strip().upper()
    if key_name not in KNOWN_KEYS:
        typer.echo(f"Unknown key {key_name!r}. Known keys: {', '.join(sorted(KNOWN_KEYS))}")
        raise typer.Exit(code=2)

    if remove:
        typer.echo(
            f"{key_name} removed." if clear_key(key_name) else f"{key_name} was not set."
        )
        return

    # hide_input keeps the key off the screen and out of shell history.
    secret = value if value is not None else typer.prompt(
        f"Paste your {key_name}", hide_input=True
    )
    try:
        path = set_key(key_name, secret)
    except ValueError as exc:
        typer.echo(f"Could not save it: {exc}")
        raise typer.Exit(code=2) from None
    typer.echo(f"Saved {key_name} to {path} (readable only by you).")
    typer.echo(f"This unlocks: {KNOWN_KEYS[key_name]}")


@app.command(name="model")
def model_command(
    download: Annotated[
        bool, typer.Option("--download", help="Fetch the weights (about 4GB, once).")
    ] = False,
) -> None:
    """Show or fetch the local model that writes `--llm` reports.

    Separate from `ask` on purpose. Four gigabytes is not something to
    download as a side effect of asking a question, so `ask --llm` reports
    the model as unavailable and falls back to the template until this has
    been run.
    """
    from .synthesis.local_llm import ensure_weights, LocalUnavailable, describe

    status = describe()
    typer.echo(f"Model:  {status.model}")
    typer.echo(f"Status: {status.detail}")

    if not download:
        if not status.ready and status.supported and status.installed:
            typer.echo("\nTo fetch it:  research model --download")
        typer.echo(
            "\nReports are written on this machine. No key, no network, and the "
            "evidence never leaves the laptop."
        )
        return

    if status.ready:
        typer.echo("\nAlready downloaded; nothing to do.")
        return
    try:
        path = ensure_weights()
    except LocalUnavailable as exc:
        typer.echo(f"\n{exc}")
        raise typer.Exit(code=2) from None
    typer.echo(f"\nDownloaded to {path}")
    typer.echo("`research ask \"why did SPY fall today\" --llm` will now use it.")


#: Command names the shim below must not swallow.
app.command(name="plan")(plan_command)


_COMMANDS = {"app", "ask", "backtest", "keys", "model", "plan", "quant"}


def main() -> None:
    """Entry point, with a shim that keeps bare questions working.

    Typer requires an explicit subcommand once an app has more than one. That
    would turn `research "why did SPY fall today"` into
    `research ask "why did SPY fall today"`, which is worse for the audience
    this tool is for. So a first argument that is not a known command and not
    a flag is treated as the start of a question.
    """
    argv = sys.argv[1:]
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        sys.argv.insert(1, "ask")
    app()


if __name__ == "__main__":
    main()
