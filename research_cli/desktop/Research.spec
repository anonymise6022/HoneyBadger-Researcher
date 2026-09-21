# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for Research.app.

Three things here are not defaults and matter:

`datas` ships the `web/` folder into the bundle. `shell._entry_point` looks
for it under `sys._MEIPASS` at runtime; without this the app builds fine and
opens a blank window, which is a miserable thing to debug afterwards.

`hiddenimports` names packages PyInstaller's static analysis cannot see.
`arch` reaches its models through string lookups and `yfinance` imports
`curl_cffi` lazily, so neither is detected by following import statements.

`excludes` drops the heavy scientific stack that is installed in this
environment for the older options pipeline but unused here. Without it the
bundle carries torch and xgboost and grows by hundreds of megabytes.

Build with:  pyinstaller research_cli/desktop/Research.spec --noconfirm
"""

from pathlib import Path

# SPECPATH is .../research_cli/desktop, so the project root is two levels
# up: parents[0] is research_cli, parents[1] is the repository root.
PROJECT = Path(SPECPATH).resolve().parents[1]

a = Analysis(
    # launcher.py, not __main__.py: PyInstaller runs its entry script as a
    # top-level module, where the relative imports in __main__.py cannot
    # resolve.
    [str(PROJECT / "research_cli" / "desktop" / "launcher.py")],
    pathex=[str(PROJECT)],
    binaries=[],
    datas=[(str(PROJECT / "research_cli" / "desktop" / "web"), "web")],
    hiddenimports=[
        "research_cli",
        "research_cli.desktop.api",
        "arch",
        "arch.univariate",
        "curl_cffi",
        "yfinance",
        "webview.platforms.cocoa",
        "statsmodels.tsa.statespace._filters",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "xgboost", "sklearn", "tkinter", "PyQt5", "PyQt6", "PySide6", "IPython"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Research",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Research",
)

app = BUNDLE(
    coll,
    name="Research.app",
    icon=str(PROJECT / "research_cli" / "desktop" / "Research.icns"),
    bundle_identifier="local.research-cli.desktop",
    version="0.2.0",
    info_plist={
        "CFBundleName": "Research",
        "CFBundleDisplayName": "Research",
        "CFBundleShortVersionString": "0.2.0",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        # The app reads market data over HTTPS only; no plaintext exception
        # is needed, and asking for one would be a gratuitous weakening.
        "NSAppTransportSecurity": {"NSAllowsArbitraryLoads": False},
    },
)
