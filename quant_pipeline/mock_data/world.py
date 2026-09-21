"""One coherent synthetic market, wiring the separate generators together.

The individual generators in this package are independent by design, which
makes them good test fixtures and useless as a pipeline demonstration: a
macro forecast fitted against returns that have nothing to do with the
macro data produces a divergence signal that is pure noise, and the run
shows only that the plumbing does not crash.

`simulate_world` builds a single market in which the pieces are related the
way they are in reality:

* **Volatility has two components.** A slow regime schedule (calm, buildup,
  crash, recovery) sets the level; a rough fractional path with H = 0.14
  multiplies it. Both are needed. The schedule alone is piecewise linear
  and a Hurst estimator reads it as H near 1; the rough path alone has no
  crises for the topological signal to find.
* **All assets share the roughness multiplier**, so the point cloud the TDA
  layer sees contracts and expands as one, which is the co-movement that
  layer is built to detect.
* **Implied volatility exceeds realized volatility**, by a log premium that
  is itself larger in stressed regimes. This is the variance risk premium,
  and it is included specifically so the pipeline's `PremiumBaseline` has
  a real premium to remove rather than a zero to subtract.
* **The underlying's drift can be driven by a macro variable**, through the
  optional `drift_driver`.

**A warning about that last point.** When `drift_driver` is supplied, the
macro variable genuinely predicts returns *because this function put it
there*. A macro forecaster fitted on the result will find that relationship
and the divergence signal will look informative. That is circular: it
demonstrates the pipeline can recover a signal that exists, which is worth
knowing, and demonstrates nothing whatever about whether such a signal
exists in real markets. The default `drift_beta` is also far larger than
any real macro-return relationship, chosen so the demo's output has visible
structure rather than realistic effect sizes. Read the demo as a wiring
check, never as a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .chains import mock_chain_series
from .fbm import fbm_vol_series
from .regimes import regime_shift_return_panel

__all__ = ["SimulatedWorld", "simulate_world"]

_INTRADAY_OBSERVATIONS = 78  # 5-minute sampling over a 6.5-hour session


@dataclass(frozen=True)
class SimulatedWorld:
    """Everything the pipeline needs for one run, on a common date index.

    Attributes
    ----------
    returns : (T, n_assets) daily returns, the point cloud for the TDA
        layer. Column `asset_0` is the traded underlying.
    regime : the ground-truth regime label per day, for comparison against
        what `regime_detection` infers. The pipeline never sees this.
    prices : the underlying's price path, starting at `initial_price`.
    effective_vol : the true annualized conditional volatility per day,
        regime level times roughness multiplier. Latent; not observable.
    realized_vol : the observable daily realized-volatility estimate, i.e.
        `effective_vol` plus intraday estimation noise.
    implied_atm_vol : at-the-money implied volatility used to quote the
        chains -- realized volatility plus the variance risk premium.
    chains : one option chain per date, keyed by Timestamp.
    expiry_years : the expiry every chain is quoted at.
    """

    returns: pd.DataFrame
    regime: pd.Series
    prices: pd.Series
    effective_vol: pd.Series
    realized_vol: pd.Series
    implied_atm_vol: pd.Series
    chains: dict[pd.Timestamp, pd.DataFrame]
    expiry_years: float

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.returns.index  # type: ignore[return-value]

    @property
    def underlying_returns(self) -> pd.Series:
        return self.returns["asset_0"]


def simulate_world(
    n_days: int = 1500,
    n_assets: int = 4,
    start: str = "2019-01-02",
    initial_price: float = 100.0,
    hurst: float = 0.14,
    nu: float = 0.22,
    drift_driver: pd.Series | None = None,
    drift_beta: float = 0.25,
    variance_risk_premium: float = 0.12,
    stress_premium: float = 0.10,
    expiry_days: int = 21,
    rate: float = 0.03,
    quote_noise: float = 0.0025,
    seed: int | None = None,
) -> SimulatedWorld:
    """Build one coherent simulated market.

    Parameters
    ----------
    n_days : business days to simulate.
    n_assets : correlated series in the panel; `asset_0` is traded.
    start : first business day.
    initial_price : starting level of the underlying.
    hurst, nu : Hurst exponent and vol-of-log-vol of the roughness
        multiplier. `regime_detection.rough_vol` should recover `hurst`
        from `realized_vol`, biased a little downward by the measurement
        noise and a little upward by the regime schedule underneath it.
    drift_driver : optional standardized series, aligned to the simulated
        index, that drives the underlying's expected return. See the module
        docstring on why a signal planted here is circular.
    drift_beta : annualized drift per unit of `drift_driver`.
    variance_risk_premium : baseline log gap between implied and realized
        volatility. 0.12 is about a 13% relative premium, i.e. two to three
        volatility points on a 20-vol underlying.
    stress_premium : extra log premium at the peak of a crash, ramped in
        proportion to how far realized volatility sits above its median.
    expiry_days : option expiry, in business days. Must match the macro
        forecast horizon or `ensemble.moment_gaps` will refuse the pair.
    rate : continuously compounded risk-free rate used to quote the chains.
    quote_noise : per-quote implied-volatility noise.
    seed : base seed. Sub-generators derive distinct streams from it, so
        one seed reproduces the entire world.

    Raises ValueError for a non-positive price, an out-of-range premium, or
    a `drift_driver` that cannot be aligned to the simulated index.
    """
    if initial_price <= 0.0:
        raise ValueError(f"initial_price must be positive, got {initial_price}")
    if expiry_days < 1:
        raise ValueError(f"expiry_days must be at least 1, got {expiry_days}")
    if variance_risk_premium < 0.0 or stress_premium < 0.0:
        raise ValueError(
            f"premia are log spreads of implied over realized and should be non-negative; "
            f"got variance_risk_premium={variance_risk_premium}, stress_premium={stress_premium}"
        )

    base_seed = 0 if seed is None else int(seed)
    rng = np.random.default_rng(base_seed + 1000)

    panel = regime_shift_return_panel(n_days, n_assets, start=start, seed=base_seed)
    index = panel.index

    # Roughness multiplier: a rough path with unit long-run level, so it
    # modulates the regime volatility without changing its average.
    rough = fbm_vol_series(
        n_days, hurst=hurst, nu=nu, long_run_vol=1.0, mean_reversion=0.05,
        start=start, seed=base_seed + 1,
    )
    multiplier = np.exp(rough["log_vol"].to_numpy())

    asset_columns = [c for c in panel.columns if c.startswith("asset_")]
    returns = panel[asset_columns].to_numpy(dtype=np.float64) * multiplier[:, None]

    effective_vol = panel["true_vol"].to_numpy(dtype=np.float64) * multiplier

    if drift_driver is not None:
        aligned = drift_driver.reindex(index)
        if aligned.isna().all():
            raise ValueError(
                "drift_driver shares no dates with the simulated index; align it to "
                f"{index[0].date()}..{index[-1].date()} before passing it"
            )
        # A driver that is NaN during its own warm-up contributes no drift
        # there, which is the right behaviour: no information, no tilt.
        returns[:, 0] += drift_beta * aligned.fillna(0.0).to_numpy() / 252.0

    returns_frame = pd.DataFrame(returns, index=index, columns=asset_columns)
    prices = initial_price * (1.0 + returns_frame["asset_0"]).cumprod()

    measurement_noise = rng.standard_normal(n_days) * np.sqrt(0.5 / _INTRADAY_OBSERVATIONS)
    realized_vol = pd.Series(
        effective_vol * np.exp(measurement_noise), index=index, name="realized_vol"
    )

    # The variance risk premium widens in stress: implied volatility
    # overshoots realized by more when realized is already high, which is
    # the empirical pattern and gives the PremiumBaseline something
    # non-constant to track.
    stress = np.clip(np.log(effective_vol / np.median(effective_vol)), 0.0, None)
    log_premium = variance_risk_premium + stress_premium * stress
    implied_atm_vol = pd.Series(
        realized_vol.to_numpy() * np.exp(log_premium), index=index, name="implied_atm_vol"
    )

    expiry_years = expiry_days / 252.0
    chains = mock_chain_series(
        index,
        prices.to_numpy(),
        implied_atm_vol.to_numpy(),
        expiry_years=expiry_years,
        rate=rate,
        quote_noise=quote_noise,
        seed=base_seed + 2,
        n_strikes=41,
    )

    return SimulatedWorld(
        returns=returns_frame,
        regime=panel["regime"].rename("true_regime"),
        prices=prices.rename("price"),
        effective_vol=pd.Series(effective_vol, index=index, name="effective_vol"),
        realized_vol=realized_vol,
        implied_atm_vol=implied_atm_vol,
        chains=chains,
        expiry_years=expiry_years,
    )
