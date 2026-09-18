# Implied Distribution Pipeline

Recovers the market's risk-neutral distribution of an underlying from its
options chain. Closed-form pricing, constrained least squares, and finite
differences — no ML, no backtesting.

## Data flow

```
data_scraper.py       →  black_scholes.py   →  svi_model.py      →  implied_density.py
chain DataFrame          w = iv² · T           fit w(k) smile       q(K) = e^(rT) ∂²C/∂K²
[strike, call_price,     k = ln(K/F)           (a, b, ρ, m, σ)      → density + moments
 put_price, iv,          + Greeks              no-arb constrained
 expiry, timestamp]
```

| Module | Role | Output |
| --- | --- | --- |
| `data_scraper.py` | Polygon.io equity chains (needs `POLYGON_API_KEY`); synthetic equity/FX chains offline. FX has no free public source, so `synthetic_fx_chain` stands in. | `DataFrame[strike, call_price, put_price, iv, expiry, timestamp]` |
| `black_scholes.py` | Vectorized BSM prices and Greeks (delta, gamma, theta, vega). | `ndarray` of prices/Greeks |
| `svi_model.py` | Gatheral raw SVI `w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))`, SLSQP fit under no-arbitrage bounds. | `SVIFit(params, residuals, rmse)` |
| `implied_density.py` | Breeden-Litzenberger on the (ideally SVI-smoothed) call curve. | `ImpliedDistribution(strikes, density, mean, variance, skew)` |

Fit SVI before extracting density: the second derivative amplifies quote
noise by `ε/h²`, so raw quotes give a ragged, often negative density.

## Usage

```bash
pytest test_math.py    # 30 tests: parity, Greeks vs. finite differences, SVI bounds, lognormal recovery
```

Requires `numpy`, `scipy`, `pandas`, `requests` (plus `pytest` to test).

## References

Black & Scholes (1973); Breeden & Litzenberger (1978); Gatheral (2006),
*The Volatility Surface*, ch. 3; Gatheral & Jacquier (2014), *Arbitrage-free
SVI volatility surfaces*; Lee (2004), moment formula for wing bounds.
