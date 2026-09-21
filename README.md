Algorithmic Derivatives Trading System

A research-oriented algorithmic trading system focused on derivatives, particularly options.

Hmph! It's not like I'm impressed by this project or anything, okay? I mean, sure, it combines market data and all that boring macroeconomic stuff, along with... ugh, machine learning—whatever! And don't even get me started on volatility modeling and options pricing, it’s just... it's just a way to analyze derivatives and find those so-called 'favorable trading opportunities'. But whatever, it’s not like I care about that... I just hope it works out, that's all! Hmph!

Hmph! It's not like I care about your boring system or anything, but I guess it’s kinda smart to not just depend on one stupid prediction, you know? I mean, who would actually be so reckless? Instead, this whole options strategy thing takes into account all those things like direction, volatility, pricing, market conditions, and risk... all at once! Ugh, why do I even bother explaining this to you, it’s not like I’m impressed or anything! Just try to understand, okay?!**.

Status: In active development. Models and strategy components are still being tested and refined.


## Quick Start

Run the whole pipeline end to end on synthetic data. No API keys, no network,
no configuration:

```bash
pip install -r requirements.txt
python demo_results.py
```

It takes about 25 seconds and produces:

| Output | Where | What it is |
|---|---|---|
| Daily table | console | One row per day: macro forecast moments, option-implied moments, divergence score, TDA z-score, Hurst exponent, regime flag, conviction, trade decision |
| Summary block | console | Trade counts, regime breakdown, mean conviction, and **NaN counts and module failures per module**, so a broken component is visible immediately |
| Results table | `results/pipeline_output.csv` | Every intermediate quantity, one row per day, 39 columns -- open it in pandas or Excel |
| Chart | `results/conviction_over_time.png` | Conviction over time, with greyscale regime bands and up/down markers for long and short signals |

Useful flags:

```bash
python demo_results.py --days 180          # longer reported window
python demo_results.py --seed 42           # a different simulated world
python demo_results.py --refit-every 1     # refit the macro model daily (slower)
python demo_results.py --table-rows 20     # print only the last 20 table rows
python demo_results.py --no-plot           # skip the PNG
python demo_results.py --help              # everything else
```

Tests for the pipeline's mathematical properties:

```bash
pytest tests/unit/test_quant_pipeline.py -q
```

### Using real data

Every data-fetching function falls back to synthetic data and upgrades silently
when credentials are present:

```bash
export FRED_API_KEY=...      # macro series (free: fred.stlouisfed.org)
export POLYGON_API_KEY=...   # equity option chains (free tier: polygon.io)
```

With `FRED_API_KEY` set, `load_macro_panel()` fetches the live series instead of
generating them. Note that no free source provides *survey consensus* forecasts,
so on live data the surprise features fall back to a random-walk proxy and become
change features; this is flagged on `MacroFeatures.consensus_is_modelled`.

### What the demo does and does not show

It shows that the components fit together and produce numbers of sensible
magnitude on data whose true parameters are known -- the Hurst estimate should
land near the 0.14 the volatility path was generated at, the implied density
should integrate to nearly 1 with its mean at the forward, and the regime flags
should respond to the injected crisis.

It is **not a backtest**. The simulated world deliberately plants a relationship
between a macro feature and the underlying's drift so the pipeline has something
to find, at an effect size far beyond anything real macro data offers, and the
evaluation window is positioned to contain a regime change. Any apparent
profitability is circular by construction.

## Research Pipeline (`quant_pipeline/`)

A self-contained implementation of one strategy idea: compare the return
distribution implied by macroeconomic conditions against the one the options
market is pricing, and trade the disagreement.

```
quant_pipeline/
  macro_model/         macro data -> features -> forecast return distribution
    data_ingest.py       FRED retrieval, publication lags, mock fallback
    feature_engineering.py  surprise-vs-consensus, z-scores, curve slope, lags
    forecast_model.py    L1-regularized quantile regression -> mean/variance/skew
  quant_model/         option chain -> SVI smile -> risk-neutral density
    market_view.py       the composition, plus strike -> return change of variables
  regime_detection/    diagnostics that modulate conviction, never trigger trades
    tda_signal.py        persistent homology, persistence-landscape norms
    rough_vol.py         Hurst estimation and the RFSV forecast
  ensemble/
    divergence_signal.py moment gaps, premium baseline, conviction, decision
  mock_data/           synthetic generators so every module runs offline
```

Three design decisions worth knowing before reading the code:

**The two distributions are not directly comparable.** One is risk-neutral, the
other real-world. They differ by the equity and variance risk premia even when
nobody is wrong, so the raw gap would signal "long" every day. The signal is
today's gap relative to its own trailing median; see `ensemble/divergence_signal.py`.

**Regime detection only modulates.** The topological and rough-volatility signals
enter as multipliers in `[0, 1]` on conviction. Neither can create a position,
flip its sign, or increase its size -- `PipelineConfig` enforces the bound. The
reasoning, including where the published topological result fails to replicate,
is in `regime_detection/tda_signal.py`.

**Publication lag is applied before anything else.** FRED dates observations by
reference period, not release date; January CPI is dated 1 January and published
in mid-February. Joining on the reference date hands the model six weeks of
information it could not have had. `data_ingest.align_to_daily` shifts each
series by its release lag first.


Architecture

Market Data ───────┐
Options Data ──────┤
Open Interest ─────┤
Macro Data ────────┤
                   ↓
              Data Pipeline
                   ↓
           Feature Processing
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
   Direction    Volatility   Options
     Model        Model       Pricing
       └───────────┼───────────┘
                   ↓
             Risk Management
                   ↓
              Backtesting
                   ↓
             Trading Signal
1. Data Pipeline

Hmph! It's, like, not that I care or anything, but the data pipeline, um, it collects and processes all that market information that those various system components need. Not that I'm impressed or anything! So don't get the wrong idea, okay?!

W-Whatever! I guess, um, data sources might, like, include... Ugh, not that I care about it or anything! It's just... there are a few options, okay? Not that you need to know! Hmph!:

Underlying asset price data
Options chains
Implied volatility
Open Interest
Trading volume
Historical volatility
Order/positioning data
Macroeconomic data
The pipeline converts raw data into structured datasets and features that can be used for modeling, analysis, and backtesting.

The system was initially experimented with using BTC/USDT data, but the architecture is now being developed specifically around derivatives and options rather than a single underlying asset.

2. Macro Data Processor

The macro data processor collects, organizes, and processes economic and financial information that may affect derivatives markets.

Instead of forcing all inputs into a rigid predefined template, the processor is designed to handle relatively flexible datasets and transform them into structured features that can be consumed by the rest of the system.

Potential inputs include:

Inflation
Interest rates
Central-bank data
Economic indicators
FX data
Financial conditions
Company data
Other macroeconomic variables
The goal is to turn otherwise messy economic information into quantifiable inputs for the trading system.

3. Direction Model

The direction model estimates the potential movement of the underlying asset.

The project has experimented with:

Linear models
SGD-based models
XGBoost
Neural networks using PyTorch
Early experiments showed that relying heavily on conventional technical indicators produced weak predictive performance. As a result, the project is moving toward features that incorporate market structure, positioning, volatility, and macroeconomic conditions.

The direction model is treated as one input to the derivatives strategy, rather than the entire strategy itself.

4. Volatility Model

Volatility is a central component of the system because options prices depend heavily on volatility.

The volatility layer is intended to estimate and analyze:

Realized volatility
Implied volatility
Volatility regimes
Volatility term structure
Volatility dynamics
GARCH-style volatility estimates
The model's volatility estimates can then be compared with the volatility implied by current option prices.

This allows the system to evaluate not only where the underlying might move, but also how much it may move and how that compares with the market's expectations.

5. Options Pricing

The options layer evaluates derivatives using quantitative pricing methods.

Current/planned components include:

Black-Scholes-based pricing
Implied volatility
Greeks
Option valuation
Volatility comparisons
Mispricing analysis
The long-term objective is to compare theoretical or model-derived values against observed market prices to identify potentially favorable opportunities.

For example:

Market Option Price
        ↓
Implied Volatility
        ↓
Compare with Model Volatility
        ↓
Evaluate Pricing Difference
        ↓
Risk / Reward Analysis
6. Risk Management

Risk management is a separate layer of the system.

A model prediction by itself does not determine whether a trade should be taken.

The risk layer is intended to handle:

Position sizing
Volatility-based exposure
Maximum position limits
Drawdown controls
Portfolio exposure
Options-specific risk
Transaction costs
Slippage
Risk/reward constraints
The objective is to determine whether a trade is worth taking and how much exposure it should receive, rather than simply maximizing prediction accuracy.

7. Inference & Trading Strategy

The inference layer brings the individual components together.

It is responsible for:

Loading trained models.
Retrieving recent market, options, and macro data.
Processing the data into model features.
Generating directional predictions.
Estimating volatility.
Evaluating option pricing and Greeks.
Applying risk-management rules.
Producing a final trading signal.
The intended decision process is therefore closer to:

Market Conditions
       +
Macro Conditions
       +
Direction
       +
Volatility
       +
Option Pricing
       +
Risk
       ↓
Trading Decision
rather than simply:

Price goes up → Buy
8. Backtesting

The complete derivatives strategy will be evaluated through historical backtesting before any live deployment.

The backtesting system is intended to account for:

Option pricing
Entry and exit conditions
Position sizing
Implied vs. realized volatility
Transaction fees
Slippage
Portfolio exposure
Drawdowns
Risk limits
Model accuracy alone is not considered sufficient. The goal is to determine whether the complete strategy produces a robust risk-adjusted result out of sample.

Technology

The project is primarily built using:

Python
PyTorch
pandas
NumPy
scikit-learn
XGBoost
SciPy
Statistical and volatility-modeling libraries
Financial and market-data APIs
Current Status

The project is under active development.

Completed / Explored

Market-data pipeline
Open Interest integration
Feature generation
Multiple ML model experiments
PyTorch model development
Initial model evaluation
Macro data processor
Initial derivatives/options architecture
In Development

Improved quantitative features
Macro feature processing
Neural-network refinement
Volatility modeling
Options pricing
Greeks
Risk management
Full derivatives backtesting
Future

More comprehensive options-data pipeline
Multi-model strategy integration
Paper trading
Broker/exchange integration
Automated execution
Production-grade monitoring and risk controls
Disclaimer

This project is for research and educational purposes. It is not financial advice and is not currently intended to guarantee or demonstrate profitable trading performance.
