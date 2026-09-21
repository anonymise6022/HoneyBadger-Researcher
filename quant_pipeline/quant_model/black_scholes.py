"""Black-Scholes pricing and Greeks, re-exported from `options_pricing/refined`.

See that module for the derivations. This file exists so callers can write
`from quant_pipeline.quant_model.black_scholes import call_price` without
depending on the older flat layout; the implementation is not duplicated.
"""

from __future__ import annotations

from ._refined import load_refined

_bs = load_refined("black_scholes")

d1_d2 = _bs.d1_d2
call_price = _bs.call_price
put_price = _bs.put_price
delta = _bs.delta
gamma = _bs.gamma
theta = _bs.theta
vega = _bs.vega

__all__ = ["call_price", "d1_d2", "delta", "gamma", "put_price", "theta", "vega"]
