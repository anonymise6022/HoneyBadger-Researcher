"""Raw SVI smile calibration, re-exported from `options_pricing/refined`.

w(k) = a + b(rho(k - m) + sqrt((k - m)^2 + sigma^2)), fitted under the
Gatheral & Jacquier (2014) no-arbitrage constraints. See that module for
the parametrization and the constraint derivations.
"""

from __future__ import annotations

from ._refined import load_refined

_svi = load_refined("svi_model")

SVIParams = _svi.SVIParams
SVIFit = _svi.SVIFit
svi_total_variance = _svi.svi_total_variance
fit_svi = _svi.fit_svi

__all__ = ["SVIFit", "SVIParams", "fit_svi", "svi_total_variance"]
