"""Breeden-Litzenberger density extraction, re-exported from `options_pricing/refined`.

q(K) = exp(rT) d2C/dK2. See that module for the derivation, the non-uniform
finite-difference scheme, and why the moments it returns are moments of the
*truncated* distribution.

`ImpliedDistribution` carries moments in strike space. To compare them with
a macro forecast of returns, convert with
`quant_pipeline.quant_model.market_view.implied_return_moments`.
"""

from __future__ import annotations

from ._refined import load_refined

_density = load_refined("implied_density")

ImpliedDistribution = _density.ImpliedDistribution
extract_density = _density.extract_density

__all__ = ["ImpliedDistribution", "extract_density"]
