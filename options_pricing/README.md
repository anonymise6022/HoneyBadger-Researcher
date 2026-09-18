# Options Pricing & Implied Distribution — Math Pipeline

A minimal, textbook-verifiable pipeline from an options chain to the market's
implied probability distribution of the underlying. Pure calculus and
numerical methods — no ML, no backtesting.

## Pipeline

```
[data_scraper.py]        [black_scholes.py]         [svi_model.py]           [implied_density.py]
  fetch/mock chain   ->   invert prices to IV,   ->   fit SVI smile      ->   Breeden-Litzenberger
  (K, C, P, IV, T)        compute Greeks               w(k) = a + b(...)      q(K) = e^(rT) d²C/dK²
                                                        per expiry
        |                        |                          |                        |
        v                        v                          v                        v
   options chain            price/Greeks              smooth, arbitrage-      risk-neutral density,
   DataFrame                sanity checks             constrained vol           mean/variance/skew
                                                        curve, no-arb            of S_T
                                                        bounds enforced
```

1. **`data_scraper.py`** — pulls a live stock options chain from Polygon.io
   (`strike, call_price, put_price, IV, expiry, timestamp`; requires a free
   API key, falls back to a Black-Scholes mock chain without one). FX
   options have no free public data source, so `generate_fx_mock_chain`
   produces a synthetic Black-Scholes chain for a currency pair instead —
   swap in a paid vol provider behind the same schema if needed.
2. **`black_scholes.py`** — closed-form Black-Scholes price and Greeks
   (delta, gamma, theta, vega), used both to sanity-check quoted prices
   against a model and to generate the mock chain.
3. **`svi_model.py`** — fits Gatheral's raw SVI parametrization
   `w(k) = a + b(ρ(k-m) + sqrt((k-m)² + σ²))` to one expiry's total-variance
   smile via constrained least squares (`scipy.optimize.minimize`, SLSQP),
   enforcing no-static-arbitrage bounds on `[a, b, ρ, m, σ]`.
4. **`implied_density.py`** — takes a call-price curve (either raw market
   quotes or SVI-smoothed prices) and applies the Breeden-Litzenberger
   identity `q(K) = e^(rT) · d²C/dK²`, computed via central finite
   differences, to extract the market-implied risk-neutral density and its
   mean, variance, and skew.

## Running

Each file is standalone and runnable via `python <file>.py`, using mock
data where relevant so nothing requires API keys or network access to
demonstrate the math.

```bash
cd options_pricing
python black_scholes.py       # price + Greeks for a sample contract
python svi_model.py           # fit SVI to a synthetic noisy smile
python implied_density.py     # extract density from a BS-generated call curve
python data_scraper.py        # fetch live AAPL chain from Polygon.io (or mock), plus FX mock
```

Set `POLYGON_API_KEY` (env var, or pass `api_key=` directly) to pull live
stock options data; get a free key at https://polygon.io/. Without one,
`fetch_stock_options_chain` falls back to mock data automatically.

## Dependencies

`numpy`, `scipy`, `pandas`, `requests` only.
