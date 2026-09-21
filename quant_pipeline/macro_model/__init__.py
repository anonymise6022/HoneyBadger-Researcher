"""The macro view: economic conditions -> a forecast return distribution.

    data_ingest.py          FRED retrieval, publication lags, mock fallback
    feature_engineering.py  surprises, z-scores, curve slope, lags
    forecast_model.py       quantile regression -> mean, variance, skew
    diagnostics.py          out-of-sample coverage and pinball skill

The output is a `ReturnDistribution` in the same units as the market-implied
one from `quant_model`, which is the only reason the two can be differenced.
Everything here runs offline: `load_macro_panel()` falls back to synthetic
data when no FRED_API_KEY is set.
"""

from __future__ import annotations

from .data_ingest import (
    FRED_SERIES,
    MacroDataError,
    MacroPanel,
    align_to_daily,
    load_macro_panel,
)
from .diagnostics import CoverageReport, coverage_report, pinball_loss
from .feature_engineering import MacroFeatureConfig, MacroFeatures, build_macro_features
from .forecast_model import (
    GaussianBaselineForecaster,
    MacroForecast,
    QuantileRegressionForecaster,
    ReturnDistribution,
    forward_return_target,
    moments_from_quantiles,
)

__all__ = [
    "FRED_SERIES",
    "CoverageReport",
    "GaussianBaselineForecaster",
    "MacroDataError",
    "MacroFeatureConfig",
    "MacroFeatures",
    "MacroForecast",
    "MacroPanel",
    "QuantileRegressionForecaster",
    "ReturnDistribution",
    "align_to_daily",
    "build_macro_features",
    "coverage_report",
    "forward_return_target",
    "load_macro_panel",
    "moments_from_quantiles",
    "pinball_loss",
]
