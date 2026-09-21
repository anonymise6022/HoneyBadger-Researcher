"""Frozen-application entry point.

Separate from `__main__.py` because the two are executed differently.
`python -m research_cli.desktop` runs inside the package, so relative
imports resolve. PyInstaller runs its entry script as a *top-level* module
with no parent package, and a relative import there fails at startup with
"attempted relative import with no known parent package" -- which surfaces
to the user as an app that bounces in the Dock and vanishes.

So this file uses absolute imports and is the target the spec points at.
"""

from __future__ import annotations

import sys


def main() -> None:
    from research_cli.desktop.shell import launch

    launch(mock="--mock" in sys.argv, debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
