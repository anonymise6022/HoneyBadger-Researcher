"""Where the macro view and the market view are differenced into a decision.

The signal is a disagreement, not a prediction. `macro_model` says what the
return distribution should look like given economic conditions;
`quant_model` reports what the options market is pricing. When they differ
by more than they usually do, there is something to trade.

**The measure-mismatch problem, and what is done about it.**

The two distributions are not directly comparable and never will be. The
market-implied density is risk-neutral; the macro forecast is real-world.
They differ systematically even when nobody is wrong:

* The risk-neutral mean is the forward, i.e. the risk-free rate. The
  real-world mean includes the equity risk premium, historically several
  percent a year. A raw mean gap is therefore *persistently positive* and
  would signal "long" every single day.
* Implied variance exceeds realized variance by the variance risk premium,
  typically two to four volatility points. A raw variance gap would
  persistently say the market overprices volatility.
* Implied skew is more negative than realized skew, because index puts
  carry a crash-insurance premium.

All three are premia, not mispricings. So the raw gaps are not the signal;
the signal is *today's gap relative to its own recent history*. A
`PremiumBaseline` accumulates the trailing gaps and the comparison is made
against their median. This is the honest minimum. It does not decompose the
premium into its parts, it assumes the premium is slowly varying, and it
needs a burn-in period before it means anything -- all three are real
limitations and none of them go away by ignoring the problem.

**Why regime only modulates.**

`regime_detection` returns a flag and a roughness reading. They enter as
multipliers in [0, 1] on conviction and can never create a signal, flip its
sign, or increase its size. This is a structural choice, not a tuning one.
Both diagnostics are noisy and neither has out-of-sample evidence strong
enough to justify taking a position on its own -- the topological signal is
validated on three pre-2010 crashes and weak on everything since, and a
rolling Hurst estimate moves by several hundredths on data with a constant
true H. As a veto they can cost a good trade. As a trigger they would
generate positions from noise. `PipelineConfig` enforces the bound: both
scales must lie in [0, 1].

**What the score means.**

`divergence_score` is signed and expressed in standard deviations of the
market-implied distribution, so a value of 1.0 means the macro forecast's
mean sits one implied standard deviation away from where the market has it,
after removing the usual premium. `conviction` is a saturating transform of
its magnitude, times the regime multipliers, in [0, 1]. The trade decision
needs both: enough divergence to be worth acting on, and enough conviction
to be worth sizing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..config import TRADING_DAYS_PER_YEAR, PipelineConfig
from ..macro_model.forecast_model import MacroForecast
from ..quant_model.market_view import MarketView
from ..regime_detection import combine_regime_flags
from ..regime_detection.rough_vol import RoughVolConfig
from ..regime_detection.tda_signal import TDAConfig

__all__ = [
    "Decision",
    "DivergenceSignal",
    "MomentGap",
    "PremiumBaseline",
    "RegimeModulation",
    "divergence_signal",
    "moment_gaps",
    "regime_modulation",
]

Decision = Literal["long", "short", "no-trade"]

_HORIZON_TOLERANCE = 0.25  # the two horizons may differ by at most 25%
_SKEW_WEIGHT = 0.15  # contribution of the skew gap to the directional score
_CONVICTION_SCALE = 1.0  # divergence at which tanh-conviction reaches 0.76


@dataclass(frozen=True)
class MomentGap:
    """Macro forecast minus market view, moment by moment.

    Attributes
    ----------
    mean_gap : (macro mean - implied mean) divided by the implied standard
        deviation. Dimensionless, and in units a trader can reason about:
        1.0 is a one-sigma disagreement about the centre.
    variance_log_ratio : log(macro variance / implied variance). Logs rather
        than a difference so that "macro expects half the variance" and
        "macro expects twice the variance" are equal and opposite.
    skew_gap : macro skew minus implied skew. Both are standardized third
        moments, so this is already dimensionless.
    """

    mean_gap: float
    variance_log_ratio: float
    skew_gap: float

    def __sub__(self, other: MomentGap) -> MomentGap:
        """Element-wise difference, used to subtract a premium baseline."""
        if not isinstance(other, MomentGap):
            return NotImplemented
        return MomentGap(
            mean_gap=self.mean_gap - other.mean_gap,
            variance_log_ratio=self.variance_log_ratio - other.variance_log_ratio,
            skew_gap=self.skew_gap - other.skew_gap,
        )

    def is_finite(self) -> bool:
        """True when every component is finite -- the precondition for scoring."""
        return bool(
            np.all(np.isfinite([self.mean_gap, self.variance_log_ratio, self.skew_gap]))
        )


@dataclass(frozen=True)
class RegimeModulation:
    """The regime layer's verdict: a flag, a multiplier and its reasons.

    Attributes
    ----------
    regime_flag : the more severe of the topological and roughness flags.
    scale : the multiplier applied to conviction, in [0, 1].
    tda_flag, rough_flag : the two component flags, kept so a combined
        "unstable" can be attributed to whichever raised it.
    tda_z : landscape-norm z-score driving `tda_flag`. NaN before the
        z-score window fills.
    hurst, hurst_drop : the roughness reading driving `rough_flag`.
    reasons : short human-readable notes, for the demo's output and for
        debugging a surprising conviction.
    """

    regime_flag: str
    scale: float
    tda_flag: str
    rough_flag: str
    tda_z: float
    hurst: float
    hurst_drop: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DivergenceSignal:
    """The full decision for one day, with every intermediate kept.

    Attributes
    ----------
    raw_gaps : moment gaps before the premium baseline is removed.
    adjusted_gaps : gaps after removal. Equal to `raw_gaps` while the
        baseline is still warming up, in which case `baseline_ready` is
        False and the score should be treated as uninterpretable.
    divergence_score : signed, in implied standard deviations.
    base_conviction : conviction from the divergence magnitude alone.
    conviction : after regime modulation; this is what sizing would use.
    regime : the modulation that was applied.
    decision : "long", "short" or "no-trade".
    reason : why that decision, in one phrase.
    baseline_ready : whether the premium baseline had enough history.
    """

    raw_gaps: MomentGap
    adjusted_gaps: MomentGap
    divergence_score: float
    base_conviction: float
    conviction: float
    regime: RegimeModulation
    decision: Decision
    reason: str
    baseline_ready: bool


class PremiumBaseline:
    """Trailing median of the moment gaps, i.e. the standing risk premium.

    A rolling *median* rather than a mean: the gaps are heavy-tailed -- one
    day where the SVI fit is poor or the macro forecast is extreme moves a
    mean baseline enough to distort the next several weeks of signal -- and
    the median is what a premium estimate should be robust to.

    The window is a trade-off with no clean answer. Too short and the
    baseline tracks the signal itself, subtracting away the very
    disagreement being measured. Too long and it fails to follow the
    genuine slow variation in risk premia. The default of 60 observations
    is about a quarter.

    Note what this cannot do: if the premium shifts abruptly -- as the
    variance risk premium did in March 2020 -- the baseline lags by roughly
    half a window, and every signal in between is contaminated.
    """

    def __init__(self, window: int = 60, min_observations: int = 20) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        if min_observations < 2:
            raise ValueError(f"min_observations must be at least 2, got {min_observations}")
        if min_observations > window:
            raise ValueError(
                f"min_observations ({min_observations}) cannot exceed window ({window})"
            )
        self.window = window
        self.min_observations = min_observations
        self._history: deque[MomentGap] = deque(maxlen=window)

    def __len__(self) -> int:
        return len(self._history)

    @property
    def ready(self) -> bool:
        """True once enough gaps have accumulated for a usable median."""
        return len(self._history) >= self.min_observations

    def update(self, gap: MomentGap) -> None:
        """Add one day's gaps. Non-finite gaps are ignored, not stored."""
        if gap.is_finite():
            self._history.append(gap)

    def current(self) -> MomentGap:
        """Median gap over the window, or an all-zero gap before `ready`.

        Returning zeros rather than raising keeps the caller's code
        straight-line; `DivergenceSignal.baseline_ready` is what marks the
        resulting score as not yet meaningful.
        """
        if not self.ready:
            return MomentGap(0.0, 0.0, 0.0)
        stacked = np.array(
            [[g.mean_gap, g.variance_log_ratio, g.skew_gap] for g in self._history]
        )
        median = np.median(stacked, axis=0)
        return MomentGap(float(median[0]), float(median[1]), float(median[2]))


def moment_gaps(forecast: MacroForecast, view: MarketView) -> MomentGap:
    """Difference the two distributions, moment by moment.

    Raises ValueError when the forecast horizon and the option expiry
    disagree by more than 25%. Comparing a one-month macro forecast against
    a three-month implied density is not a divergence, it is a units error:
    variance grows with horizon, so the variance gap would be dominated by
    the mismatch and the signal would be a horizon difference wearing a
    mispricing's name.
    """
    expiry_days = view.expiry_years * TRADING_DAYS_PER_YEAR
    if expiry_days <= 0.0:
        raise ValueError(f"market view has a non-positive expiry: {view.expiry_years} years")
    relative_gap = abs(forecast.horizon_days - expiry_days) / expiry_days
    if relative_gap > _HORIZON_TOLERANCE:
        raise ValueError(
            f"horizon mismatch: the macro forecast covers {forecast.horizon_days} days but the "
            f"option expiry is {expiry_days:.1f} days ({relative_gap:.0%} apart, tolerance "
            f"{_HORIZON_TOLERANCE:.0%}). Re-fit the forecast at the option's horizon, or read "
            "the market view from an expiry that matches."
        )

    implied_std = view.std_return
    if implied_std <= 0.0:
        raise ValueError(f"market view has a non-positive implied std: {implied_std}")
    macro = forecast.distribution
    return MomentGap(
        mean_gap=float((macro.mean - view.mean_return) / implied_std),
        variance_log_ratio=float(np.log(macro.variance / view.variance_return)),
        skew_gap=float(macro.skew - view.skew_return),
    )


def regime_modulation(
    tda_z: float,
    hurst: float,
    hurst_drop: float,
    config: PipelineConfig | None = None,
    tda_config: TDAConfig | None = None,
    rough_config: RoughVolConfig | None = None,
) -> RegimeModulation:
    """Turn the two regime diagnostics into one flag and one multiplier.

    Each diagnostic is classified independently, the more severe flag wins,
    and the multiplier is read off `config`. A NaN from either -- which is
    what a window that has not filled yet produces -- classifies as
    "stable", so a cold start does not spend the run vetoing everything.
    That choice errs toward trading during warm-up, and the demo reports
    NaN counts per module so the warm-up period is visible rather than
    silently treated as calm.
    """
    cfg = config or PipelineConfig()
    tda_cfg = tda_config or TDAConfig(
        elevated_z=cfg.tda_elevated_z, unstable_z=cfg.tda_unstable_z
    )
    rough_cfg = rough_config or RoughVolConfig()

    reasons: list[str] = []
    if not np.isfinite(tda_z):
        tda_flag = "stable"
        reasons.append("tda z-score unavailable (window warming up)")
    elif tda_z >= tda_cfg.unstable_z:
        tda_flag = "unstable"
        reasons.append(f"landscape norm {tda_z:.1f}sd above its trailing mean")
    elif tda_z >= tda_cfg.elevated_z:
        tda_flag = "elevated"
        reasons.append(f"landscape norm {tda_z:.1f}sd above its trailing mean")
    else:
        tda_flag = "stable"

    if not np.isfinite(hurst_drop):
        rough_flag = "stable"
        reasons.append("hurst baseline unavailable (window warming up)")
    elif hurst_drop >= rough_cfg.unstable_drop:
        rough_flag = "unstable"
        reasons.append(f"H fell {hurst_drop:.3f} below its median (vol path turned rough)")
    elif hurst_drop >= rough_cfg.elevated_drop:
        rough_flag = "elevated"
        reasons.append(f"H fell {hurst_drop:.3f} below its median")
    else:
        rough_flag = "stable"

    regime_flag = combine_regime_flags(tda_flag, rough_flag)
    scale = {
        "stable": 1.0,
        "elevated": cfg.elevated_conviction_scale,
        "unstable": cfg.unstable_conviction_scale,
    }[regime_flag]
    if not reasons:
        reasons.append("both diagnostics within their normal ranges")

    return RegimeModulation(
        regime_flag=regime_flag,
        scale=float(scale),
        tda_flag=tda_flag,
        rough_flag=rough_flag,
        tda_z=float(tda_z),
        hurst=float(hurst),
        hurst_drop=float(hurst_drop),
        reasons=tuple(reasons),
    )


def _directional_score(gaps: MomentGap) -> float:
    """Combine the adjusted gaps into one signed score.

    The mean gap carries the direction: if macro expects a higher mean than
    the market prices, that is a reason to be long. The skew gap enters with
    a small positive weight -- when the market prices *more* negative skew
    than the macro forecast implies, `skew_gap` is positive and the market
    is paying up for downside protection the macro view does not justify,
    which leans long. The weight is deliberately small (0.15): the skew
    comparison is the least reliable of the three, since the implied skew
    depends on how far the SVI wings were extrapolated and the macro skew
    on a third moment reconstructed from seven quantiles.

    The variance gap is *not* in the directional score. A disagreement about
    variance is a reason to trade volatility -- a straddle, a strangle --
    not a reason to be long or short the underlying. It is reported in the
    signal for exactly that reason and left out of a direction it cannot
    determine.
    """
    return float(gaps.mean_gap + _SKEW_WEIGHT * gaps.skew_gap)


def divergence_signal(
    forecast: MacroForecast,
    view: MarketView,
    tda_z: float = float("nan"),
    hurst: float = float("nan"),
    hurst_drop: float = float("nan"),
    baseline: PremiumBaseline | None = None,
    config: PipelineConfig | None = None,
    update_baseline: bool = True,
) -> DivergenceSignal:
    """Produce one day's divergence score, conviction and trade decision.

    Parameters
    ----------
    forecast : the macro return-distribution forecast.
    view : the market-implied distribution, at a matching horizon.
    tda_z : today's landscape-norm z-score; NaN if unavailable.
    hurst, hurst_drop : today's roughness reading; NaN if unavailable.
    baseline : the premium baseline to difference against. When None, no
        premium is removed and `baseline_ready` is False -- the raw gaps
        are reported, but they are premia plus signal and should not be
        traded.
    config : thresholds and scales.
    update_baseline : whether to fold today's gaps into the baseline. True
        is correct for a sequential run. Set False to score a day without
        letting it influence its own baseline.

    Returns
    -------
    DivergenceSignal. Never raises for an ordinary bad day -- a non-finite
    gap yields a "no-trade" with the reason recorded -- but does raise
    ValueError for a horizon mismatch or a degenerate market view, which
    are wiring errors rather than market conditions.
    """
    cfg = config or PipelineConfig()
    raw = moment_gaps(forecast, view)
    regime = regime_modulation(tda_z, hurst, hurst_drop, cfg)

    # The baseline must be read *before* today's gap is folded in, or the
    # day partially cancels its own signal.
    baseline_ready = baseline is not None and baseline.ready
    adjusted = raw - baseline.current() if baseline is not None else raw
    if baseline is not None and update_baseline:
        baseline.update(raw)

    if not adjusted.is_finite():
        return DivergenceSignal(
            raw_gaps=raw,
            adjusted_gaps=adjusted,
            divergence_score=float("nan"),
            base_conviction=0.0,
            conviction=0.0,
            regime=regime,
            decision="no-trade",
            reason="divergence is not finite; a moment failed to compute",
            baseline_ready=baseline_ready,
        )

    score = _directional_score(adjusted)
    # tanh saturates, so a wildly large divergence -- which in practice
    # means a bad density extraction, not a spectacular opportunity --
    # cannot produce a conviction above 1 or a position sized off a tail.
    base_conviction = float(np.tanh(abs(score) / _CONVICTION_SCALE))
    conviction = float(base_conviction * regime.scale)

    if not baseline_ready:
        decision: Decision = "no-trade"
        reason = f"premium baseline still warming up ({len(baseline) if baseline else 0} observations)"
    elif abs(score) < cfg.divergence_trade_threshold:
        decision = "no-trade"
        reason = f"|divergence| {abs(score):.2f} below threshold {cfg.divergence_trade_threshold:.2f}"
    elif conviction < cfg.min_conviction:
        decision = "no-trade"
        reason = (
            f"conviction {conviction:.2f} below floor {cfg.min_conviction:.2f} "
            f"(regime '{regime.regime_flag}' scaled it by {regime.scale:.2f})"
        )
    else:
        decision = "long" if score > 0.0 else "short"
        reason = (
            f"macro mean {abs(adjusted.mean_gap):.2f}sd "
            f"{'above' if adjusted.mean_gap > 0 else 'below'} implied, regime "
            f"'{regime.regime_flag}'"
        )

    return DivergenceSignal(
        raw_gaps=raw,
        adjusted_gaps=adjusted,
        divergence_score=score,
        base_conviction=base_conviction,
        conviction=conviction,
        regime=regime,
        decision=decision,
        reason=reason,
        baseline_ready=baseline_ready,
    )
