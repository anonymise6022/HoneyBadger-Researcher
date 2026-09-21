"""Loader that exposes `options_pricing/refined/*.py` as package modules.

The refined pricing code predates this package and is written to run as
flat scripts: its modules import each other by bare name
(`from black_scholes import call_price`), which only resolves when that
directory is on `sys.path`. Copying the files into `quant_pipeline` would
create a second copy of the Black-Scholes and SVI implementations that
would silently drift from the tested originals, so instead they are loaded
from their original location and re-exported by the thin modules beside
this one.

The bare-name aliases are registered in `sys.modules` only for the duration
of the load, and `sys.path` is never modified, so importing this package
cannot change how unrelated code resolves the name `black_scholes`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["REFINED_DIR", "load_refined"]

REFINED_DIR = Path(__file__).resolve().parents[2] / "options_pricing" / "refined"

_CACHE: dict[str, ModuleType] = {}


def load_refined(name: str) -> ModuleType:
    """Import `options_pricing/refined/<name>.py` and cache the module.

    Raises ImportError if the file is missing, which means the repository
    layout changed and `REFINED_DIR` needs updating.
    """
    if name in _CACHE:
        return _CACHE[name]

    path = REFINED_DIR / f"{name}.py"
    if not path.is_file():
        raise ImportError(
            f"cannot find {path}; quant_pipeline.quant_model re-exports the pricing code "
            f"from options_pricing/refined/, so that directory must exist alongside it"
        )

    spec = importlib.util.spec_from_file_location(f"_refined_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)

    # The refined modules import one another by bare name. Expose the
    # already-loaded ones under those names just while this one executes.
    shadowed = {alias: sys.modules.get(alias) for alias in _CACHE}
    sys.modules.update(_CACHE)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for alias, previous in shadowed.items():
            if previous is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = previous

    _CACHE[name] = module
    return module
