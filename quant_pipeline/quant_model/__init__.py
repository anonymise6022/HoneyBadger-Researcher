"""The market's view: option chain -> SVI smile -> risk-neutral density.

This half of the pipeline never looks at macro data. It reports only what
the option market is currently pricing for the underlying's terminal
distribution, expressed in the same units as the macro forecast so the two
can be differenced.

    black_scholes.py     pricing and Greeks
    svi_model.py         arbitrage-constrained smile calibration
    implied_density.py   Breeden-Litzenberger curvature extraction
    market_view.py       the composition of the three, plus the strike ->
                         return change of variables

The first three re-export `options_pricing/refined/` rather than copying it;
see `_refined.py`.
"""

from __future__ import annotations

from .implied_density import ImpliedDistribution, extract_density
from .market_view import MarketView, implied_return_moments, market_view_from_chain
from .svi_model import SVIParams, fit_svi, svi_total_variance

__all__ = [
    "ImpliedDistribution",
    "MarketView",
    "SVIParams",
    "extract_density",
    "fit_svi",
    "implied_return_moments",
    "market_view_from_chain",
    "svi_total_variance",
]
