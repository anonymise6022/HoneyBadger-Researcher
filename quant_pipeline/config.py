"""Tunable constants shared across the pipeline, in one place.

These are the numbers a researcher actually wants to sweep: window lengths,
regime thresholds, and the conviction cut that separates a trade from a
no-trade. They live here rather than as scattered defaults so that a
sensitivity study is a single object, not a grep.

Every value is a modelling choice, not a fact. The defaults are set so the
synthetic pipeline produces a readable mix of outcomes (roughly a third of
days trading); on real data they must be recalibrated against the observed
distribution of divergence scores, not carried over.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TRADING_DAYS_PER_YEAR", "PipelineConfig"]

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PipelineConfig:
    """Window lengths and decision thresholds for one pipeline run.

    Attributes
    ----------
    macro_lookback_days : rows of macro history used to fit the forecaster.
    macro_zscore_window : window for rolling z-scores of macro levels.
    macro_surprise_window : window for the surprise standard deviation that
        normalizes actual-minus-consensus.
    macro_lags : lag depths (in days) added for each macro feature.
    forecast_horizon_days : horizon of the forecast return distribution.
        Must match the option expiry the market view is read from, or the
        two distributions are not comparable.
    quantile_levels : quantiles the regression predicts and the
        distribution is reconstructed from.
    tda_window : points per rolling point cloud (Gidea & Katz use 50).
    tda_zscore_window : window for the rolling z-score of landscape norms.
    tda_elevated_z, tda_unstable_z : landscape-norm z-scores at which the
        regime flag steps from "stable" to "elevated" to "unstable".
    rough_vol_window : trailing realized-vol observations used to estimate
        the Hurst exponent.
    hurst_elevated_drop : how far H must fall below its trailing median
        before the rough-vol diagnostic counts as regime stress.
    divergence_trade_threshold : |divergence score| needed to consider a
        trade at all.
    min_conviction : conviction below which the decision is forced to
        no-trade regardless of divergence.
    unstable_conviction_scale, elevated_conviction_scale : multiplicative
        haircuts applied to conviction in each regime. Both are <= 1: the
        regime layer can only shrink conviction, never manufacture it.
    """

    macro_lookback_days: int = 750
    macro_zscore_window: int = 250
    macro_surprise_window: int = 24
    macro_lags: tuple[int, ...] = (1, 5, 21)
    forecast_horizon_days: int = 21

    quantile_levels: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

    tda_window: int = 50
    tda_zscore_window: int = 60
    tda_elevated_z: float = 1.0
    tda_unstable_z: float = 2.0

    rough_vol_window: int = 250
    hurst_elevated_drop: float = 0.05

    divergence_trade_threshold: float = 0.35
    min_conviction: float = 0.30
    elevated_conviction_scale: float = 0.70
    unstable_conviction_scale: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 < self.min_conviction < 1.0:
            raise ValueError(f"min_conviction must lie in (0, 1), got {self.min_conviction}")
        if self.divergence_trade_threshold < 0.0:
            raise ValueError(
                f"divergence_trade_threshold must be non-negative, got "
                f"{self.divergence_trade_threshold}"
            )
        if not self.tda_elevated_z < self.tda_unstable_z:
            raise ValueError(
                f"tda_elevated_z ({self.tda_elevated_z}) must be below tda_unstable_z "
                f"({self.tda_unstable_z}); the flags are ordered by severity"
            )
        for name in ("elevated_conviction_scale", "unstable_conviction_scale"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1] -- the regime layer modulates conviction "
                    f"downward only, never upward; got {value}"
                )
        if any(not 0.0 < q < 1.0 for q in self.quantile_levels):
            raise ValueError(f"quantile_levels must lie strictly in (0, 1), got {self.quantile_levels}")
        if list(self.quantile_levels) != sorted(self.quantile_levels):
            raise ValueError("quantile_levels must be sorted ascending")
