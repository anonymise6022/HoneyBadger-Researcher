"""Print a validated debate report. Styling only; the text is printed as checked.

The rich path bolds the section headings and nothing else, so what the user
reads is character-for-character the report the validator passed.
"""

from __future__ import annotations

__all__ = ["print_debate_report"]

_HEADINGS = (
    "QUESTION", "THE SIMPLE PICTURE", "THE STRONGEST CASE", "SEVERAL STRONG VIEWS",
    "WHAT IF: SCENARIOS FROM HISTORY", "WHAT TO WATCH FOR", "YOU DECIDE",
)


def print_debate_report(text: str, notes: list[str], usage: str, fallback_reason: str = "",
                        plain: bool = False) -> None:
    from .formatting import console_available  # noqa: PLC0415

    if plain or not console_available():
        if fallback_reason:
            print(f"Note: showing the standard report because {fallback_reason}.")
        print(text)
        for note in notes:
            print(f"  debate note: {note}")
        print(usage)
        return

    from rich.console import Console
    from rich.markup import escape

    console = Console()
    if fallback_reason:
        console.print(f"[yellow]Showing the standard report because {escape(fallback_reason)}.[/yellow]")
    for line in text.splitlines():
        if line.strip() in _HEADINGS or line.startswith("  THE DEBATE --"):
            console.print(f"[bold]{escape(line)}[/bold]")
        else:
            console.print(escape(line), highlight=False)
    for note in notes:
        console.print(f"[dim]  debate note: {escape(note)}[/dim]")
    console.print(f"[dim]{escape(usage)}[/dim]")
