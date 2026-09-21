"""End-to-end options research pipeline: macro view vs. market-implied view.

The pipeline answers one question each day: does a forecast of the return
distribution built from macroeconomic conditions disagree with the
distribution the options market is currently pricing, and if so, is the
disagreement worth trading?

    macro_model/       macro data -> features -> forecast return distribution
    quant_model/       option chain -> SVI smile -> risk-neutral density
    regime_detection/  TDA + rough-volatility regime diagnostics
    ensemble/          divergence score, regime modulation, trade decision
    mock_data/         synthetic generators so every module runs offline

Design rule for the signal chain: `macro_model` and `quant_model` produce
the two distributions whose disagreement *is* the signal. `regime_detection`
never produces a trade on its own -- it only scales conviction up or down.
That separation is deliberate; see `ensemble.divergence_signal`.

Nothing here requires an API key. Every data-fetching function has a
synthetic fallback, so `python demo_results.py` runs on a clean checkout.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
