"""Options paper trading, on synthetic contracts priced from the real underlying.

**What is real and what is not.** The underlying's price path is real market
data. The option contracts are not: there is no free source of historical
option chains, so each contract is *constructed* at the moment it is opened
and priced with Black-Scholes, using a volatility estimated from the
underlying's own recent history plus a variance risk premium. That is a
model, not a quote.

Three consequences, all of which flatter the results and all of which are
reported on every run rather than buried:

* **No bid-ask spread on the option.** Real option spreads are wide,
  especially away from the money, and they are frequently the difference
  between a profitable strategy and a losing one. A configurable spread is
  applied, but it is a guess.
* **Implied volatility is modelled, not observed.** Real IV moves with
  supply and demand and spikes before earnings; here it tracks trailing
  realized volatility plus a fixed premium. Strategies that are really bets
  on the *level* of IV -- which is most option strategies -- are therefore
  being tested against a caricature of the thing they trade.
* **Fractional contracts.** A real SPY contract covers 100 shares and needs
  tens of thousands of dollars of stock behind it; whole contracts would
  make every one of these strategies untestable on a normal account. Sizes
  here are continuous, which is a simplification in the user's favour.
* **European exercise, no dividends, no early assignment.** A short American
  option can be assigned before expiry, especially a put that goes deep in
  the money, and that risk is simply absent here.

So this is a tool for understanding the *shape* of an option strategy's
payoff -- that a covered call trades upside for income, that a protective put
costs a steady premium for a floor -- and not for estimating what one would
have earned.

Positions are held to expiry and rolled on a fixed schedule, which is both
the simplest honest choice and roughly what a retail seller actually does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestResult
from .risk_metrics import compute_stats

__all__ = [
    "OPTION_TEMPLATES",
    "OptionBacktestConfig",
    "black_scholes_price",
    "run_option_backtest",
]

_TRADING_DAYS = 252
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function -- exact, not tabulated."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    spot: float, strike: float, years: float, rate: float, volatility: float, kind: str
) -> float:
    """European option price under Black-Scholes.

        d1 = [ln(S/K) + (r + v^2/2)T] / (v sqrt(T))
        d2 = d1 - v sqrt(T)
        call = S N(d1) - K e^{-rT} N(d2)
        put  = K e^{-rT} N(-d2) - S N(-d1)

    At expiry (T <= 0) or with zero volatility the formula is undefined, and
    the correct value is the discounted intrinsic payoff -- returned directly
    rather than by taking a limit numerically.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")

    if years <= 0 or volatility <= 0:
        intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
        return float(intrinsic)

    vol_sqrt_t = volatility * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    discount = math.exp(-rate * years)

    if kind == "call":
        return float(spot * _normal_cdf(d1) - strike * discount * _normal_cdf(d2))
    return float(strike * discount * _normal_cdf(-d2) - spot * _normal_cdf(-d1))


@dataclass(frozen=True)
class OptionBacktestConfig:
    """Assumptions for an options run. Every one of these flatters or hurts.

    Attributes
    ----------
    initial_cash : starting portfolio value.
    days_to_expiry : tenor of each contract, in trading days. 21 is about a
        month, the tenor most retail option income strategies use.
    moneyness : strike as a multiple of spot at the moment of opening. 1.05
        on a call is 5% out of the money. `None` means "use whatever the
        strategy template considers typical", which is the usual case; an
        explicit value overrides it. An earlier version had the template
        always win, which made this parameter silently do nothing.
    rate : risk-free rate used for discounting.
    volatility_lookback : trailing bars used to estimate volatility.
    variance_risk_premium : multiplier on realized volatility to stand in for
        implied. 1.15 means options are priced about 15% above realized,
        which is roughly the long-run average for index options and is the
        single most consequential assumption here.
    option_spread_pct : half-spread charged on entry and exit, as a fraction
        of the option's price. Options are far wider than shares and this is
        the crudest part of the model.
    contract_size : shares per contract.
    """

    initial_cash: float = 10_000.0
    days_to_expiry: int = 21
    moneyness: float | None = None
    rate: float = 0.04
    volatility_lookback: int = 40
    variance_risk_premium: float = 1.15
    option_spread_pct: float = 0.02
    contract_size: int = 100

    def __post_init__(self) -> None:
        if self.days_to_expiry < 5:
            raise ValueError(
                f"days_to_expiry must be at least 5; shorter than that and the "
                f"daily-bar model has no resolution. Got {self.days_to_expiry}"
            )
        if self.moneyness is not None and not 0.5 <= self.moneyness <= 1.5:
            raise ValueError(
                f"moneyness must lie between 0.5 and 1.5 of spot, got {self.moneyness}"
            )
        if self.variance_risk_premium <= 0:
            raise ValueError("variance_risk_premium must be positive")
        if not 0.0 <= self.option_spread_pct < 0.5:
            raise ValueError(
                f"option_spread_pct is a fraction of the option price and must lie in "
                f"[0, 0.5), got {self.option_spread_pct}"
            )


@dataclass
class _Leg:
    """One open contract."""

    kind: str          # "call" or "put"
    strike: float
    expiry_index: int  # bar position at which it settles
    quantity: float    # positive long, negative short
    opened_at: float   # premium paid (negative) or received (positive), net of spread


def _realized_volatility(closes: np.ndarray, lookback: int) -> float:
    """Annualized volatility from the trailing window, floored for sanity."""
    if len(closes) < 3:
        return 0.2
    returns = np.diff(np.log(closes[-(lookback + 1):]))
    if len(returns) < 2:
        return 0.2
    value = float(returns.std(ddof=1) * math.sqrt(_TRADING_DAYS))
    # A dead window would price every option at intrinsic; floor it.
    return max(value, 0.05)


@dataclass
class OptionStrategy:
    """One option overlay, described by what it does at each roll.

    `share_exposure` is how much of the underlying is held alongside the
    option -- 1.0 for a covered call, 0.0 for a naked short put -- and
    `legs` says which contracts to open.
    """

    key: str
    name: str
    description: str
    share_exposure: float
    leg_kinds: tuple[tuple[str, float], ...]  # (kind, sign) per contract
    moneyness_override: float | None = None
    caveat: str = ""
    #: How many contracts to trade. This is not a detail -- getting it wrong
    #: silently turns one strategy into another.
    #:
    #:   "covered"        one contract per contract_size shares held. A
    #:                    covered call sized any other way is a naked short
    #:                    call with some stock next to it.
    #:   "cash-secured"   enough contracts that the cash covers assignment at
    #:                    the strike.
    #:   "premium-budget" enough contracts that the premium costs
    #:                    `premium_budget` of the account per cycle.
    sizing: str = "covered"
    premium_budget: float = 0.05


OPTION_TEMPLATES: dict[str, OptionStrategy] = {
    "covered-call": OptionStrategy(
        key="covered-call",
        name="Covered call",
        description="Own the shares and sell a call against them for income.",
        share_exposure=1.0,
        leg_kinds=(("call", -1.0),),
        moneyness_override=1.05,
        sizing="covered",
        caveat=(
            "Income now in exchange for your upside above the strike. In a strong "
            "rally this underperforms simply holding, and the shares can be called "
            "away."
        ),
    ),
    "protective-put": OptionStrategy(
        key="protective-put",
        name="Protective put",
        description="Own the shares and buy a put as insurance against a fall.",
        share_exposure=1.0,
        leg_kinds=(("put", 1.0),),
        moneyness_override=0.95,
        sizing="covered",
        caveat=(
            "A floor under your losses, paid for with a premium every month. In a "
            "flat or rising market that premium is a steady drag."
        ),
    ),
    "cash-secured-put": OptionStrategy(
        key="cash-secured-put",
        name="Cash-secured put",
        description="Hold cash and sell a put, collecting income unless it falls.",
        share_exposure=0.0,
        leg_kinds=(("put", -1.0),),
        moneyness_override=0.95,
        sizing="cash-secured",
        caveat=(
            "You keep the premium while the price holds up, and buy the shares at "
            "the strike if it does not. The payoff is close to a covered call's."
        ),
    ),
    "long-call": OptionStrategy(
        key="long-call",
        name="Long call",
        description="Buy a call outright: leveraged upside, with the premium at risk.",
        share_exposure=0.0,
        leg_kinds=(("call", 1.0),),
        moneyness_override=1.02,
        sizing="premium-budget",
        caveat=(
            "Every contract that expires out of the money is a total loss on that "
            "premium. This is the strategy most likely to end at zero."
        ),
    ),
    "long-straddle": OptionStrategy(
        key="long-straddle",
        name="Long straddle",
        description="Buy a call and a put at the same strike: a bet on a big move either way.",
        share_exposure=0.0,
        leg_kinds=(("call", 1.0), ("put", 1.0)),
        moneyness_override=1.0,
        sizing="premium-budget",
        caveat=(
            "Profitable only if the move is larger than both premiums combined. "
            "Since the model prices volatility above realized, this is expected to "
            "lose slowly -- which is also true in real markets."
        ),
    ),
}


def run_option_backtest(
    frame: pd.DataFrame,
    strategy_key: str,
    config: OptionBacktestConfig | None = None,
    execution: BacktestConfig | None = None,
) -> BacktestResult:
    """Roll an option overlay through history and mark it daily.

    Returns the same `BacktestResult` the equity engine produces, so the
    display layer needs no special case.

    Timing follows the same discipline as the equity engine: a contract
    opened on bar *i* is priced from information available at bar *i*, and
    every valuation afterwards uses only that day's spot and the volatility
    known by then.

    Raises ValueError for an unknown strategy or too little history.
    """
    if strategy_key not in OPTION_TEMPLATES:
        raise ValueError(
            f"unknown option strategy {strategy_key!r}; available: "
            f"{', '.join(sorted(OPTION_TEMPLATES))}"
        )
    settings = config or OptionBacktestConfig()
    costs = execution or BacktestConfig()
    strategy = OPTION_TEMPLATES[strategy_key]

    prices = frame.sort_index()
    closes = prices["Close"].to_numpy(dtype=float)
    index = prices.index
    warmup = settings.volatility_lookback + 2

    if len(closes) < warmup + settings.days_to_expiry * 2:
        raise ValueError(
            f"need at least {warmup + settings.days_to_expiry * 2} bars to roll this "
            f"strategy; the history has {len(closes)}"
        )

    cash = settings.initial_cash
    shares = 0.0
    legs: list[_Leg] = []
    trades = 0
    total_costs = 0.0
    notes: list[str] = []
    # Cash in and out of the option leg alone, so a result can be
    # attributed rather than guessed at.
    premium_collected = 0.0
    settlement_paid = 0.0

    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []
    # An explicit config value wins; otherwise the template's typical strike.
    moneyness = (
        settings.moneyness
        if settings.moneyness is not None
        else (strategy.moneyness_override or 1.0)
    )

    def option_value(leg: _Leg, position: int) -> float:
        """Mark one leg at the current bar."""
        remaining = max(leg.expiry_index - position, 0) / _TRADING_DAYS
        volatility = (
            _realized_volatility(closes[: position + 1], settings.volatility_lookback)
            * settings.variance_risk_premium
        )
        unit = black_scholes_price(
            closes[position], leg.strike, remaining, settings.rate, volatility, leg.kind
        )
        return leg.quantity * unit * settings.contract_size

    for position in range(warmup, len(closes)):
        spot = closes[position]

        # Settle anything expiring today, at intrinsic value.
        expiring = [leg for leg in legs if leg.expiry_index <= position]
        for leg in expiring:
            intrinsic = (
                max(spot - leg.strike, 0.0) if leg.kind == "call"
                else max(leg.strike - spot, 0.0)
            )
            cash += leg.quantity * intrinsic * settings.contract_size
            settlement_paid -= leg.quantity * intrinsic * settings.contract_size
            legs.remove(leg)

        # Open a new cycle whenever nothing is outstanding.
        if not legs:
            volatility = (
                _realized_volatility(closes[: position + 1], settings.volatility_lookback)
                * settings.variance_risk_premium
            )
            years = settings.days_to_expiry / _TRADING_DAYS
            expiry_index = position + settings.days_to_expiry
            if expiry_index >= len(closes):
                break  # not enough history left to see this cycle through

            # Rebalance the share leg first, so the option is written against
            # the position actually held.
            target_shares = strategy.share_exposure * (cash + shares * spot) / spot
            delta_shares = target_shares - shares
            if abs(delta_shares) > 1e-9:
                notional = abs(delta_shares) * spot
                share_cost = notional * costs.cost_rate
                cash -= delta_shares * spot + share_cost
                total_costs += share_cost
                shares = target_shares

            # One moneyness per strategy, applied to every leg. Right for
            # the templates here -- a covered call is one out-of-the-money
            # call, a straddle is both legs at the money -- and a strangle
            # would need this to move onto `leg_kinds`.
            strike = spot * moneyness

            # Size the position. Fractional contracts are allowed, which
            # real markets do not permit: one SPY contract covers 100 shares
            # and needs tens of thousands of dollars of stock behind it, so
            # a whole-contract model would refuse every account below that
            # and make this feature useless for the people it is for.
            equity_now = cash + shares * spot
            if strategy.sizing == "covered":
                contracts = shares / settings.contract_size
            elif strategy.sizing == "cash-secured":
                contracts = cash / (strike * settings.contract_size) if strike > 0 else 0.0
            else:
                unit_premium = sum(
                    black_scholes_price(
                        spot, strike, years, settings.rate, volatility, kind
                    )
                    for kind, _ in strategy.leg_kinds
                ) * settings.contract_size
                contracts = (
                    (strategy.premium_budget * equity_now) / unit_premium
                    if unit_premium > 1e-9
                    else 0.0
                )

            if contracts <= 1e-9:
                # Nothing can be written this cycle; carry the shares and
                # try again at the next roll rather than silently trading
                # a position the account cannot support.
                equity_values.append(equity_now)
                equity_index.append(index[position])
                continue

            for kind, sign in strategy.leg_kinds:
                quantity = sign * contracts
                fair = black_scholes_price(
                    spot, strike, years, settings.rate, volatility, kind
                )
                # The spread is always paid, whichever way the trade goes.
                spread = fair * settings.option_spread_pct
                premium = (fair - spread) if quantity < 0 else (fair + spread)
                cash -= quantity * premium * settings.contract_size
                premium_collected -= quantity * premium * settings.contract_size
                total_costs += abs(spread) * settings.contract_size * abs(quantity)
                legs.append(
                    _Leg(
                        kind=kind,
                        strike=strike,
                        expiry_index=expiry_index,
                        quantity=quantity,
                        opened_at=premium,
                    )
                )
                trades += 1

        equity = cash + shares * spot + sum(option_value(leg, position) for leg in legs)
        # A short-option account can be wiped out; record it and stop rather
        # than continuing with a negative portfolio, which would make every
        # subsequent return meaningless.
        if equity <= 0:
            notes.append(
                f"The account reached zero on {index[position].date()}. Short options "
                "can lose more than the premium collected."
            )
            equity_values.append(max(equity, 1e-6))
            equity_index.append(index[position])
            break

        equity_values.append(equity)
        equity_index.append(index[position])

    if len(equity_values) < 5:
        raise ValueError("the option backtest produced too few marks to measure")

    equity = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index), name="equity")

    # Buy and hold over the identical window, with the same entry cost.
    first = closes[warmup]
    units = settings.initial_cash * (1.0 - costs.cost_rate) / first
    benchmark = pd.Series(
        units * closes[warmup : warmup + len(equity)], index=equity.index, name="benchmark"
    )

    option_pnl = premium_collected - settlement_paid
    notes.append(
        f"Of the result above, the option leg contributed "
        f"{option_pnl:+,.0f} ({premium_collected:+,.0f} in premium, "
        f"{-settlement_paid:+,.0f} settled at expiry). The rest came from the "
        "shares."
    )
    notes.append(
        "Option prices here are modelled with Black-Scholes from the underlying's own "
        "volatility, not taken from real quotes. Treat the shape of the result as "
        "informative and the level as indicative only."
    )
    notes.append(
        "Black-Scholes assumes returns are lognormal. Real equity indices are "
        "negatively skewed -- big down days are more common than the model expects "
        "and big up days less so -- and real index options are priced with a "
        "volatility skew that compensates for it. This model has no skew, so an "
        "out-of-the-money call finishes in the money less often here than its own "
        "price implies. That systematically favours the option *seller*, and it is "
        "the main reason a covered call can look better here than the published "
        "buy-write indices have actually done."
    )
    if strategy.caveat:
        notes.append(strategy.caveat)

    return BacktestResult(
        equity=equity,
        exposure=pd.Series(strategy.share_exposure, index=equity.index, name="exposure"),
        benchmark_equity=benchmark,
        stats=compute_stats(equity, n_trades=trades),
        benchmark_stats=compute_stats(benchmark, n_trades=1),
        trades=trades,
        total_costs=total_costs,
        strategy_name=strategy.name,
        warmup_bars=warmup,
        notes=notes,
    )
