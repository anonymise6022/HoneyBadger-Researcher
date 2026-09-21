"""Synthetic data generators: every module's offline fallback lives here.

The pipeline must run end to end with no API keys and no network, both so
that `demo_results.py` works on a clean checkout and so that tests can
assert against data whose true parameters are known. Each generator takes
an explicit `seed` and returns a reproducible pandas object.

    fbm.py      fractional Brownian motion, fractional OU log-vol paths
    regimes.py  multi-asset return panels with injected regime shifts
    macro.py    CPI / unemployment / 2y / 10y / HY-OAS panels with consensus
    chains.py   option chains carrying a realistic volatility smile
    world.py    all of the above wired into one coherent market

Nothing in here is calibrated to a specific market. The numbers are chosen
to be *plausible* -- 2-4% inflation, 4% unemployment, 300-600bp high-yield
spreads, H around 0.14 in the vol path -- so that downstream code meets
realistic magnitudes, not so that backtests on them mean anything.
"""

from __future__ import annotations

from .chains import mock_chain_series, mock_option_chain
from .fbm import fbm_vol_series, fractional_brownian_motion, fractional_gaussian_noise
from .macro import mock_macro_panel
from .regimes import REGIME_LABELS, regime_shift_return_panel
from .world import SimulatedWorld, simulate_world

__all__ = [
    "REGIME_LABELS",
    "SimulatedWorld",
    "fbm_vol_series",
    "fractional_brownian_motion",
    "fractional_gaussian_noise",
    "mock_chain_series",
    "mock_macro_panel",
    "mock_option_chain",
    "regime_shift_return_panel",
    "simulate_world",
]
