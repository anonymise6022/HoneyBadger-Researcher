"""Bundle entry point: `python -m research_cli.desktop`, and the .app's target."""

from __future__ import annotations

import sys

from .shell import launch


def main() -> None:
    launch(mock="--mock" in sys.argv, debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
