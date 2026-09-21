"""The window: a native WebKit view hosting the interface.

pywebview rather than Electron or Tauri. On macOS it wraps the system's
WKWebView, so the bundle carries no browser engine -- the app is a few
megabytes of Python and assets rather than a couple of hundred megabytes of
Chromium -- and text is rendered by the same engine as Safari, properly
hinted and subpixel-antialiased. That directly serves the brief: the
interface should look like a terminal, not like a picture of one.

The window is frameless with a transparent title bar, so the traffic lights
float over the interface's own header rather than sitting in a grey strip
above it. `app.css` reserves the left padding they need.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .api import DesktopApi

__all__ = ["WINDOW_TITLE", "launch"]

WINDOW_TITLE = "Research"
WEB_ROOT = Path(__file__).resolve().parent / "web"

_MIN_WIDTH = 900
_MIN_HEIGHT = 620


def _entry_point() -> str:
    """Absolute path to index.html, resolved for both source and bundle.

    PyInstaller unpacks data files into a temporary directory recorded on
    `sys._MEIPASS`. Without this branch the app finds its assets when run
    from a checkout and shows a blank window when run from the .app, which
    is a confusing failure to diagnose after the fact.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    root = Path(bundled) / "web" if bundled else WEB_ROOT
    index = root / "index.html"
    if not index.is_file():
        raise FileNotFoundError(
            f"interface assets are missing from {root}; the application bundle is "
            "incomplete"
        )
    return str(index)


def launch(mock: bool = False, debug: bool = False, width: int = 1240, height: int = 840) -> None:
    """Open the window and block until it is closed.

    Parameters
    ----------
    mock : run entirely on synthetic data, with no network access.
    debug : enable the WebKit inspector (right-click -> Inspect Element).
    width, height : initial window size, in points.
    """
    try:
        import webview
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "pywebview is not installed; run "
            "pip install -r research_cli/requirements.txt"
        ) from exc

    api = DesktopApi(mock=mock)
    webview.create_window(
        WINDOW_TITLE,
        _entry_point(),
        js_api=api,
        width=width,
        height=height,
        min_size=(_MIN_WIDTH, _MIN_HEIGHT),
        # The interface paints its own dark ground; matching it here stops a
        # white flash between the window appearing and the CSS loading.
        background_color="#0d0f12",
        frameless=False,
        easy_drag=False,
    )
    webview.start(debug=debug)
