Algorithmic Derivatives Trading System

A research-oriented algorithmic trading system focused on derivatives, particularly options.

Hmph! It's not like I'm impressed by this project or anything, okay? I mean, sure, it combines market data and all that boring macroeconomic stuff, along with... ugh, machine learning—whatever! And don't even get me started on volatility modeling and options pricing, it’s just... it's just a way to analyze derivatives and find those so-called 'favorable trading opportunities'. But whatever, it’s not like I care about that... I just hope it works out, that's all! Hmph!

Hmph! It's not like I care about your boring system or anything, but I guess it’s kinda smart to not just depend on one stupid prediction, you know? I mean, who would actually be so reckless? Instead, this whole options strategy thing takes into account all those things like direction, volatility, pricing, market conditions, and risk... all at once! Ugh, why do I even bother explaining this to you, it’s not like I’m impressed or anything! Just try to understand, okay?!**.

Status: In active development. Models and strategy components are still being tested and refined.

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
