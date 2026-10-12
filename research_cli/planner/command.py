"""The `plan` command: options for an investment the user has already decided on.

Registered on the CLI app by `cli.py`; kept here so the command module stays
small. It places no orders and connects to no broker.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

import typer

from ..data.price_ingest import PriceDataError, UnknownSymbolError
from ..display.formatting import print_error
from ..explain import resolve_depth
from ..query_parser import resolve_symbol
from .inputs import PlanInputError, parse_amount, parse_horizon, parse_max_loss
from .interactive import build_plan_report

__all__ = ["plan_command"]


def plan_command(
    ticker: Annotated[str, typer.Argument(help="The symbol you have decided to invest in.")],
    amount: Annotated[str, typer.Option("--amount", help="How much, e.g. $1000.")],
    horizon: Annotated[str, typer.Option("--horizon", help="How long, e.g. '1 year'.")] = "1 year",
    max_loss: Annotated[
        str, typer.Option("--max-loss", help="Most you could stand to lose, e.g. $200 or 20%.")
    ] = "20%",
    mock: Annotated[bool, typer.Option("--mock", help="Synthetic data only.")] = False,
    depth: Annotated[str, typer.Option("--depth", "-d", help="kid, beginner, ...")] = "beginner",
    as_of: Annotated[str | None, typer.Option("--as-of", help="History ends here.")] = None,
) -> None:
    """Plan options for an investment you have already decided to make. Places no orders."""
    try:
        dollars = parse_amount(amount)
        days, label = parse_horizon(horizon)
        limit = parse_max_loss(max_loss, dollars)
        depth = resolve_depth(depth)
        end = date.fromisoformat(as_of) if as_of else None
    except (PlanInputError, ValueError) as exc:
        print_error(str(exc), "", True)
        raise typer.Exit(code=2) from None
    symbol, _ = resolve_symbol(ticker)
    try:
        text, validation, _ = build_plan_report(
            symbol, symbol, dollars, days, label, limit, "mock" if mock else "auto",
            as_of=end, depth=depth)
    except (PriceDataError, UnknownSymbolError) as exc:
        print_error(str(exc), getattr(exc, "suggestion", ""), True)
        raise typer.Exit(code=4) from None
    except ValueError as exc:
        print_error(f"could not build a plan: {exc}", "", True)
        raise typer.Exit(code=1) from None
    if not validation.ok:
        print_error("the plan did not pass its own figure check, so it is not shown", "", True)
        raise typer.Exit(code=1)
    print(text)
    print(f"[plan check] {validation.summary()}")
