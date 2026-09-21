"""Desktop application: a native window over the same analysis code.

    api.py     the JavaScript bridge -- marshals arguments and results only
    shell.py   the pywebview window
    web/       the interface itself (HTML, CSS, JavaScript)

No analysis happens here. Reports, hit rates and statistics come from the
same functions the command line calls, and the validator still runs before
any text reaches the window, so the two front ends cannot disagree about
what the evidence says.
"""

from __future__ import annotations

from .api import DesktopApi
from .shell import launch

__all__ = ["DesktopApi", "launch"]
