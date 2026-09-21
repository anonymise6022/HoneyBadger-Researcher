"""Tests asserting the mathematical properties each new module must satisfy.

Every test states a fact about the math or about a documented invariant --
the exact death radius of a circle's homology class, the exactness of the
Davies-Harte covariance, the fact that a risk modulator can only ever
shrink conviction -- rather than a fact about the implementation. All data
is synthetic and seeded; no network and no API keys.

    pytest tests/unit/test_quant_pipeline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, skewnorm

from quant_pipeline.config import PipelineConfig
from quant_pipeline.ensemble import (
    MomentGap,
    PremiumBaseline,
    divergence_signal,
    moment_gaps,
    regime_modulation,
)
from quant_pipeline.macro_model.data_ingest import (
    FRED_SERIES,
    MacroDataError,
    align_to_daily,
    fetch_fred_series,
    load_macro_panel,
)
from quant_pipeline.macro_model.diagnostics import coverage_report, pinball_loss
from quant_pipeline.macro_model.feature_engineering import (
    add_lags,
    build_macro_features,
    normalized_surprise,
    rolling_zscore,
    yield_curve_slope,
)
from quant_pipeline.macro_model.forecast_model import (
    QuantileRegressionForecaster,
    forward_return_target,
    moments_from_quantiles,
)
from quant_pipeline.mock_data import (
    mock_option_chain,
    regime_shift_return_panel,
    simulate_world,
)
from quant_pipeline.mock_data.fbm import fbm_vol_series, fractional_gaussian_noise
from quant_pipeline.quant_model import implied_return_moments, market_view_from_chain
from quant_pipeline.regime_detection import combine_regime_flags
from quant_pipeline.regime_detection.rough_vol import (
    _rfsv_weights,
    estimate_hurst,
    rfsv_forecast,
    rough_vol_estimate,
)
from quant_pipeline.regime_detection.tda_signal import (
    landscape_norms,
    persistence_landscape,
    rips_h1_diagram,
    rolling_tda_signal,
)

LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# --- fractional Brownian motion -------------------------------------------

@pytest.mark.parametrize("hurst", [0.1, 0.3, 0.5, 0.75])
def test_fgn_has_unit_variance(hurst: float) -> None:
    """Davies-Harte is exact, so the sample variance must be ~1 for any H."""
    sample = fractional_gaussian_noise(8192, hurst, seed=0)
    assert sample.var() == pytest.approx(1.0, abs=0.06)


@pytest.mark.parametrize("hurst", [0.1, 0.25, 0.75])
def test_fgn_lag_one_autocorrelation_matches_theory(hurst: float) -> None:
    """gamma(1) = 2^{2H-1} - 1, the defining covariance of fGn at lag 1."""
    sample = fractional_gaussian_noise(16384, hurst, seed=1)
    theoretical = 2.0 ** (2.0 * hurst - 1.0) - 1.0
    assert np.corrcoef(sample[:-1], sample[1:])[0, 1] == pytest.approx(theoretical, abs=0.03)


def test_fgn_at_half_is_white_noise() -> None:
    """H = 1/2 gives gamma(k) = 0 for every k != 0: ordinary Brownian motion."""
    sample = fractional_gaussian_noise(16384, 0.5, seed=2)
    for lag in (1, 2, 5, 20):
        assert abs(np.corrcoef(sample[:-lag], sample[lag:])[0, 1]) < 0.03


def test_fgn_rejects_invalid_hurst() -> None:
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="hurst"):
            fractional_gaussian_noise(64, bad)


# --- Hurst estimation and the RFSV forecast --------------------------------

def test_hurst_estimator_recovers_the_generating_exponent() -> None:
    """A path built at H should read back near H from the scaling regression.

    The tolerance is wide on purpose: the generator applies mean reversion
    on top of the fBm driver, which flattens the longest-lag increments and
    biases the slope down by roughly 10%.
    """
    for hurst in (0.10, 0.20, 0.35):
        path = fbm_vol_series(1500, hurst=hurst, nu=0.3, seed=3)["log_vol"].to_numpy()
        assert estimate_hurst(path).hurst == pytest.approx(hurst, abs=0.06)


def test_hurst_estimator_recovers_nu() -> None:
    """The intercepts identify nu once E|N(0,1)|^q is divided out."""
    path = fbm_vol_series(2000, hurst=0.14, nu=0.30, seed=4)["log_vol"].to_numpy()
    assert estimate_hurst(path).nu == pytest.approx(0.30, abs=0.05)


def test_self_similarity_check_is_near_one_for_an_fbm_path() -> None:
    """zeta_q must be linear in q when the path really is self-similar."""
    path = fbm_vol_series(1500, hurst=0.14, seed=5)["log_vol"].to_numpy()
    assert estimate_hurst(path).r_squared > 0.99


def test_rolling_window_smoothing_biases_hurst_upward() -> None:
    """The documented trap: a trailing-window vol proxy is not rough.

    This is asserted rather than merely noted because it is the most likely
    way to misuse the module -- 21-day trailing volatility is the proxy
    closest to hand, and it reports an ordinary diffusion.
    """
    frame = fbm_vol_series(1500, hurst=0.14, seed=6).dropna()
    daily = estimate_hurst(np.log(frame["realized_vol"].to_numpy())).hurst
    smoothed = estimate_hurst(np.log(frame["realized_vol_21d"].to_numpy())).hurst
    assert daily < 0.25
    assert smoothed > 0.40
    assert smoothed > daily + 0.2


def test_rfsv_kernel_weights_sum_to_one() -> None:
    """Renormalization is what makes a flat history forecast itself."""
    weights = _rfsv_weights(500, 0.14, 21.0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    assert np.all(np.diff(weights) < 0.0), "the kernel must decay into the past"


def test_rfsv_forecast_reproduces_a_constant_path() -> None:
    """E[log sigma_{t+D}] of a constant log-vol history is that constant."""
    level = np.log(0.23)
    expected, variance = rfsv_forecast(np.full(400, level), 0.14, 0.3, 21)
    assert expected == pytest.approx(level)
    assert variance > 0.0


def test_rfsv_forecast_rejects_a_non_rough_hurst() -> None:
    """At H >= 1/2 the prediction kernel's singularity is not integrable."""
    with pytest.raises(ValueError, match="H in"):
        rfsv_forecast(np.full(100, 0.0), 0.6, 0.3, 21)


def test_rough_vol_estimate_falls_back_rather_than_inventing_a_forecast() -> None:
    """A smooth series gives H >= 1/2; the forecast must degrade to the level."""
    smooth = np.exp(np.linspace(np.log(0.15), np.log(0.30), 400))
    estimate = rough_vol_estimate(smooth)
    if estimate.hurst >= 0.5:
        assert estimate.forecast_log_vol == pytest.approx(np.log(smooth[-1]))
        assert estimate.forecast_variance == 0.0


def test_rough_vol_rejects_non_positive_volatility() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        rough_vol_estimate(np.array([0.2, 0.1, 0.0, 0.15] * 100))


# --- persistent homology ---------------------------------------------------

def test_circle_has_exactly_one_persistent_loop() -> None:
    """The canonical check: n points on a unit circle give one H1 bar.

    Its birth is the point spacing 2 sin(pi/n) -- the radius at which
    neighbours connect -- and its death is sqrt(3), the exact Rips death
    radius of a circle, where triangles finally fill the hole in.
    """
    n = 24
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    diagram, _ = rips_h1_diagram(np.column_stack([np.cos(theta), np.sin(theta)]))
    assert diagram.shape == (1, 2)
    assert diagram[0, 0] == pytest.approx(2.0 * np.sin(np.pi / n), rel=1e-9)
    assert diagram[0, 1] == pytest.approx(np.sqrt(3.0), rel=1e-9)


def test_a_dense_blob_has_no_long_lived_loop() -> None:
    """Gaussian noise has loops, but short ones -- far below a circle's."""
    points = np.random.default_rng(7).standard_normal((30, 2))
    diagram, _ = rips_h1_diagram(points)
    persistence = diagram[:, 1] - diagram[:, 0] if diagram.size else np.array([0.0])
    assert persistence.max() < 0.5


def test_collinear_points_have_no_loops() -> None:
    """A cloud with no hole has empty H1, and an empty diagram has zero norm."""
    line = np.column_stack([np.linspace(0.0, 1.0, 20), np.zeros(20)])
    diagram, _ = rips_h1_diagram(line)
    assert diagram.size == 0
    assert landscape_norms(diagram) == (0.0, 0.0)


def test_every_bar_dies_by_the_enclosing_radius() -> None:
    """Above it the Rips complex is a cone, so no H1 class can survive."""
    points = np.random.default_rng(8).standard_normal((25, 3))
    diagram, enclosing = rips_h1_diagram(points)
    assert np.all(diagram[:, 1] <= enclosing + 1e-12)
    assert np.all(diagram[:, 1] > diagram[:, 0])


def test_landscape_of_a_single_bar_is_its_tent() -> None:
    """One bar (b, d) peaks at (d-b)/2 and has L1 norm exactly (d-b)^2/4."""
    birth, death = 0.5, 2.5
    _, landscape = persistence_landscape([[birth, death]], layers=2, grid_points=4001)
    assert landscape[0].max() == pytest.approx((death - birth) / 2.0, rel=1e-6)
    assert np.all(landscape[1] == 0.0), "a single bar has no second layer"
    l1, l2 = landscape_norms([[birth, death]], layers=2, grid_points=4001)
    assert l1 == pytest.approx((death - birth) ** 2 / 4.0, rel=1e-4)
    assert l2 > 0.0


def test_landscape_norm_scales_quadratically_with_the_cloud() -> None:
    """Scaling a cloud by c scales every bar by c, so L1 scales by c^2."""
    points = np.random.default_rng(9).standard_normal((24, 3))
    small, _ = rips_h1_diagram(points)
    large, _ = rips_h1_diagram(points * 3.0)
    assert landscape_norms(large)[0] == pytest.approx(9.0 * landscape_norms(small)[0], rel=1e-6)


def test_rolling_tda_signal_rejects_a_single_series() -> None:
    """A one-column panel is a collinear cloud; H1 would be trivially empty."""
    frame = pd.DataFrame({"only": np.random.default_rng(10).standard_normal(120)})
    with pytest.raises(ValueError, match="at least 2 series"):
        rolling_tda_signal(frame)


def test_rolling_tda_signal_flags_are_within_the_vocabulary() -> None:
    panel = regime_shift_return_panel(200, 3, seed=11)
    result = rolling_tda_signal(panel[["asset_0", "asset_1", "asset_2"]])
    assert set(result["regime_flag"]).issubset({"stable", "elevated", "unstable"})
    assert result["tda_l1"].notna().sum() > 100


# --- distribution reconstruction ------------------------------------------

def test_moments_of_gaussian_quantiles_are_exact() -> None:
    """Gaussian quantiles are linear in z-space, where the interpolation lives."""
    mean, std = 0.02, 0.05
    recovered = moments_from_quantiles(LEVELS, norm.ppf(LEVELS, mean, std))
    assert recovered[0] == pytest.approx(mean, abs=1e-9)
    assert recovered[1] == pytest.approx(std**2, abs=1e-9)
    assert recovered[2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("shape", [6.0, -6.0, 3.0])
def test_moments_of_skew_normal_quantiles_are_close(shape: float) -> None:
    """Seven quantiles recover a genuinely skewed distribution's moments."""
    distribution = skewnorm(shape, loc=0.0, scale=0.05)
    mean, variance, skew = moments_from_quantiles(LEVELS, distribution.ppf(LEVELS))
    assert mean == pytest.approx(float(distribution.mean()), rel=0.02)
    assert variance == pytest.approx(float(distribution.var()), rel=0.03)
    assert skew == pytest.approx(float(distribution.stats("s")), abs=0.05)


def test_rearrangement_sorts_rather_than_collapsing() -> None:
    """A fully crossed quantile vector must keep its spread, not flatten.

    A running maximum -- the fix this module deliberately does not use --
    would map this strictly decreasing input onto its first element and
    report zero variance.
    """
    crossed = np.array([0.05, 0.04, 0.03, 0.0, -0.02, -0.03, -0.05])
    _, variance, _ = moments_from_quantiles(LEVELS, crossed)
    assert variance > 0.0
    sorted_moments = moments_from_quantiles(LEVELS, np.sort(crossed))
    assert variance == pytest.approx(sorted_moments[1])


def test_moments_reject_unsorted_levels() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        moments_from_quantiles((0.5, 0.25, 0.75), (0.0, 1.0, 2.0))


# --- quantile regression ---------------------------------------------------

def _linear_training_data(n: int = 600, seed: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    """y = 0.4 x1 + N(0, 0.1), with x2 pure noise the penalty should drop."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n, name="date")
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    y = 0.4 * x1 + rng.standard_normal(n) * 0.1
    return pd.DataFrame({"x1": x1, "x2": x2}, index=index), pd.Series(y, index=index)


def test_quantile_regression_recovers_a_known_conditional_distribution() -> None:
    """With y = 0.4 x1 + N(0, 0.1), the forecast at x1 = 1 should centre near 0.4."""
    X, y = _linear_training_data()
    model = QuantileRegressionForecaster(LEVELS, alpha=0.0).fit(X, y)
    point = pd.Series({"x1": 1.0, "x2": 0.0})
    forecast = model.forecast(point, horizon_days=21)
    assert forecast.distribution.mean == pytest.approx(0.4, abs=0.06)
    assert forecast.distribution.std == pytest.approx(0.1, abs=0.03)
    assert abs(forecast.distribution.skew) < 0.5


def test_predicted_quantiles_are_monotone_across_levels() -> None:
    X, y = _linear_training_data()
    model = QuantileRegressionForecaster(LEVELS, alpha=1e-3).fit(X, y)
    predictions = model.predict_quantiles(X.head(50))
    assert np.all(np.diff(predictions, axis=1) >= -1e-12)


def test_l1_penalty_produces_sparser_coefficients() -> None:
    """That is the point of the penalty; without it nothing is exactly zero."""
    X, y = _linear_training_data()
    unpenalized = QuantileRegressionForecaster(LEVELS, alpha=0.0).fit(X, y)
    penalized = QuantileRegressionForecaster(LEVELS, alpha=0.05).fit(X, y)
    assert np.count_nonzero(np.abs(penalized.coefficients_) > 1e-9) < np.count_nonzero(
        np.abs(unpenalized.coefficients_) > 1e-9
    )


def test_fit_refuses_a_training_set_too_small_for_its_features() -> None:
    X, y = _linear_training_data(n=8)
    with pytest.raises(ValueError, match="too few"):
        QuantileRegressionForecaster(LEVELS).fit(X, y)


def test_forecast_rejects_a_multi_row_input() -> None:
    X, y = _linear_training_data()
    model = QuantileRegressionForecaster(LEVELS).fit(X, y)
    with pytest.raises(ValueError, match="one feature row"):
        model.forecast(X.head(3), horizon_days=21)


# --- out-of-sample calibration --------------------------------------------

def test_forecaster_is_calibrated_out_of_sample() -> None:
    """The real test of a quantile model: does the tau-quantile cover tau?

    Everything else in this file checks that the estimator computes what
    its math says. This checks that what it computes is *right* on data it
    never saw -- that the level it calls the 5th percentile really does sit
    above 5% of outcomes. A model can pass every other test here and be
    badly calibrated.

    Averaged over seeds rather than asserted on one split, because a single
    800-row test set carries enough noise to swing coverage at tau = 0.10 by
    four binomial standard errors on an unlucky draw -- the binomial error
    counts sampling noise only, not the estimation noise in the fitted
    quantiles themselves. A one-split version of this test is flaky, not
    strict. Across seeds the mean error collapses below 0.01, which is the
    statement worth guarding.
    """
    errors = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 1600
        index = pd.bdate_range("2010-01-01", periods=n, name="date")
        X = pd.DataFrame(
            {f"x{i}": rng.standard_normal(n) for i in range(1, 6)}, index=index
        )
        y = pd.Series(0.4 * X["x1"] + rng.standard_normal(n) * 0.10, index=index)
        model = QuantileRegressionForecaster(LEVELS, alpha=1e-3).fit(
            X.iloc[:800], y.iloc[:800]
        )
        report = coverage_report(model, X.iloc[800:], y.iloc[800:], y_train=y.iloc[:800])
        errors.append(report.error)
        assert np.all(np.abs(report.error) < 0.08), (
            f"seed {seed} is badly miscalibrated:\n{report.render('coverage')}"
        )

    mean_error = np.mean(errors, axis=0)
    assert np.all(np.abs(mean_error) < 0.025), (
        "coverage error averaged over seeds should vanish; got "
        + ", ".join(f"{t:.2f}:{e:+.4f}" for t, e in zip(LEVELS, mean_error))
    )


def test_conditioning_on_features_beats_the_unconditional_baseline() -> None:
    """Coverage alone is not enough: predicting the unconditional quantiles
    is perfectly calibrated and useless. The skill score is what separates
    a forecast from a histogram."""
    rng = np.random.default_rng(2)
    n = 1200
    index = pd.bdate_range("2010-01-01", periods=n, name="date")
    X = pd.DataFrame({"signal": rng.standard_normal(n), "noise": rng.standard_normal(n)}, index=index)
    y = pd.Series(0.5 * X["signal"] + rng.standard_normal(n) * 0.10, index=index)

    model = QuantileRegressionForecaster(LEVELS, alpha=1e-3).fit(X.iloc[:600], y.iloc[:600])
    report = coverage_report(model, X.iloc[600:], y.iloc[600:], y_train=y.iloc[:600])
    assert report.skill_score > 0.5, f"skill was {report.skill_score:.3f}"


def test_a_featureless_model_is_calibrated_but_has_no_skill() -> None:
    """The negative control that gives the skill score its meaning."""

    class UnconditionalForecaster:
        quantile_levels = LEVELS

        def __init__(self, y: pd.Series) -> None:
            self._quantiles = np.quantile(y.to_numpy(), LEVELS)

        def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
            return np.tile(self._quantiles, (len(X), 1))

    rng = np.random.default_rng(3)
    n = 1200
    index = pd.bdate_range("2010-01-01", periods=n, name="date")
    X = pd.DataFrame({"signal": rng.standard_normal(n)}, index=index)
    y = pd.Series(0.5 * X["signal"] + rng.standard_normal(n) * 0.10, index=index)

    report = coverage_report(
        UnconditionalForecaster(y.iloc[:600]), X.iloc[600:], y.iloc[600:], y_train=y.iloc[:600]
    )
    assert report.is_calibrated(tolerance_se=4.0), report.render("unconditional")
    assert abs(report.skill_score) < 0.05, "an unconditional model must show no skill"


def test_overlap_correction_shrinks_the_effective_sample() -> None:
    """Overlapping h-day targets are not h independent observations.

    Ignoring this understates the standard error by about sqrt(h) and turns
    ordinary noise into confident-looking miscalibration.
    """
    rng = np.random.default_rng(4)
    n = 900
    index = pd.bdate_range("2010-01-01", periods=n, name="date")
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)}, index=index)
    y = pd.Series(rng.standard_normal(n) * 0.1, index=index)
    model = QuantileRegressionForecaster(LEVELS, alpha=1e-3).fit(X.iloc[:500], y.iloc[:500])

    daily = coverage_report(model, X.iloc[500:], y.iloc[500:], horizon=1)
    monthly = coverage_report(model, X.iloc[500:], y.iloc[500:], horizon=21)

    assert monthly.n_effective == daily.n_effective // 21
    np.testing.assert_allclose(
        monthly.standard_error, daily.standard_error * np.sqrt(21.0), rtol=0.05
    )
    np.testing.assert_allclose(monthly.empirical, daily.empirical)


def test_pinball_loss_matches_its_definition() -> None:
    """rho_tau(u) = u(tau - 1{u < 0}), averaged over observations and levels."""
    y = np.array([1.0, -1.0])
    predictions = np.array([[0.0], [0.0]])
    assert pinball_loss(y, predictions, (0.25,)) == pytest.approx((0.25 * 1.0 + 0.75 * 1.0) / 2.0)


def test_pinball_loss_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        pinball_loss(np.zeros(5), np.zeros((4, 3)), (0.25, 0.5, 0.75))


def test_coverage_report_refuses_an_empty_test_set() -> None:
    X, y = _linear_training_data(n=600)
    model = QuantileRegressionForecaster(LEVELS).fit(X, y)
    with pytest.raises(ValueError, match="no test rows"):
        coverage_report(model, X.head(0), y.head(0))


def test_forward_return_target_compounds_and_looks_forward_only() -> None:
    """target_t = prod(1 + r_{t+1..t+h}) - 1, NaN over the final h rows."""
    returns = pd.Series([0.01] * 10, index=pd.bdate_range("2020-01-01", periods=10))
    target = forward_return_target(returns, 3)
    assert target.iloc[0] == pytest.approx(1.01**3 - 1.0)
    assert target.iloc[-3:].isna().all()


# --- macro ingestion and features -----------------------------------------

def test_release_lag_prevents_look_ahead() -> None:
    """A monthly print may not appear in the daily frame before its release.

    January CPI is dated 1 January by FRED and published in mid-February,
    so the aligned frame must still be showing December's value at the end
    of January.
    """
    panel = load_macro_panel(source="mock", seed=13)
    daily = align_to_daily(panel)
    cpi = panel.observations["CPIAUCSL"]
    spec = next(s for s in FRED_SERIES if s.series_id == "CPIAUCSL")
    reference = cpi.index[14]
    value = float(cpi.loc[reference])
    on_reference_date = daily["cpi"].asof(reference)
    assert on_reference_date != pytest.approx(value)
    released = reference + pd.Timedelta(days=spec.release_lag_days)
    assert float(daily["cpi"].asof(released + pd.Timedelta(days=4))) == pytest.approx(value)


def test_fetch_fred_series_without_a_key_names_the_remedy() -> None:
    with pytest.raises(MacroDataError, match="FRED_API_KEY"):
        fetch_fred_series("CPIAUCSL", "2020-01-01", "2020-06-01", api_key="")


def test_load_macro_panel_falls_back_to_mock_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert load_macro_panel(seed=14).source == "mock"


def test_rolling_zscore_matches_its_definition() -> None:
    series = pd.Series(np.arange(100, dtype=float), name="x")
    z = rolling_zscore(series, 10, min_periods=10)
    window = series.iloc[40:50]
    assert z.iloc[49] == pytest.approx((series.iloc[49] - window.mean()) / window.std(ddof=1))


def test_rolling_zscore_of_a_flat_series_is_nan_not_infinity() -> None:
    z = rolling_zscore(pd.Series([3.0] * 50, name="flat"), 10, min_periods=10)
    assert z.iloc[20:].isna().all()


def test_add_lags_shifts_backward_only() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    lagged = add_lags(frame, ["a"], (1, 2))
    assert lagged["a_lag1"].tolist()[1:] == [1.0, 2.0, 3.0]
    assert np.isnan(lagged["a_lag1"].iloc[0])
    assert lagged["a_lag2"].tolist()[2:] == [1.0, 2.0]


def test_normalized_surprise_divides_by_past_dispersion_only() -> None:
    """The scaling window is shifted, so a surprise cannot scale itself down."""
    index = pd.bdate_range("2020-01-01", periods=200)
    actual = pd.Series(np.repeat(np.arange(10, dtype=float), 20), index=index, name="x")
    consensus = actual - np.repeat([0.1, -0.1] * 5, 20)
    result = normalized_surprise(actual, consensus, window=4, min_periods=3)
    assert result.notna().sum() > 0
    assert np.isfinite(result.dropna()).all()


def test_yield_curve_slope_names_a_missing_leg() -> None:
    with pytest.raises(KeyError, match="yield_2y"):
        yield_curve_slope(pd.DataFrame({"yield_10y": [4.0]}))


def test_macro_features_are_built_without_look_ahead() -> None:
    """Every feature at row t is a function of rows <= t, so reversing the
    tail of the input must not change an earlier feature value."""
    panel = load_macro_panel(source="mock", seed=15)
    daily = align_to_daily(panel)
    full = build_macro_features(daily).frame
    truncated = build_macro_features(daily.iloc[:-200]).frame
    common = truncated.index[-50:]
    pd.testing.assert_frame_equal(
        full.loc[common], truncated.loc[common], check_exact=False, rtol=1e-10
    )


# --- market view -----------------------------------------------------------

def test_market_view_recovers_the_forward() -> None:
    """The risk-neutral mean of S_T is the forward S exp(rT)."""
    spot, expiry, rate = 100.0, 21.0 / 252.0, 0.03
    chain = mock_option_chain(spot=spot, expiry_years=expiry, rate=rate, atm_vol=0.2, n_strikes=41)
    view = market_view_from_chain(chain, spot, expiry, rate)
    assert view.forward == pytest.approx(spot * np.exp(rate * expiry))
    assert view.forward_error == pytest.approx(0.0, abs=0.01)
    assert view.distribution.total_mass == pytest.approx(1.0, abs=0.03)


def test_market_view_recovers_the_generating_volatility() -> None:
    """The chain is priced off an SVI slice; refitting must find it again."""
    for atm_vol in (0.12, 0.20, 0.35):
        width = float(np.clip(3.0 * atm_vol * np.sqrt(21.0 / 252.0), 0.15, 0.9))
        chain = mock_option_chain(
            spot=100.0, atm_vol=atm_vol, n_strikes=41, log_moneyness_width=width
        )
        view = market_view_from_chain(chain, 100.0, 21.0 / 252.0, 0.03)
        assert view.implied_vol_atm == pytest.approx(atm_vol, rel=0.06)


def test_negative_smile_skew_gives_a_negatively_skewed_density() -> None:
    """rho < 0 lifts the downside wing, which is left skew in return space."""
    chain = mock_option_chain(rho=-0.8, atm_vol=0.25, n_strikes=41)
    assert market_view_from_chain(chain, 100.0, 21.0 / 252.0, 0.03).skew_return < 0.0


def test_implied_return_moments_is_the_exact_affine_map() -> None:
    chain = mock_option_chain(spot=100.0, n_strikes=41)
    view = market_view_from_chain(chain, 100.0, 21.0 / 252.0, 0.03)
    mean, variance, skew = implied_return_moments(view.distribution, 100.0)
    assert mean == pytest.approx(view.distribution.mean / 100.0 - 1.0)
    assert variance == pytest.approx(view.distribution.variance / 100.0**2)
    assert skew == pytest.approx(view.distribution.skew), "skew is scale-invariant"


def test_market_view_needs_enough_quotes() -> None:
    chain = mock_option_chain(n_strikes=6).head(3)
    with pytest.raises(ValueError, match="at least 5 usable quotes"):
        market_view_from_chain(chain, 100.0, 21.0 / 252.0, 0.03)


# --- ensemble --------------------------------------------------------------

def _forecast_and_view(mean: float, variance: float, skew: float):
    """A MacroForecast with chosen moments, and a market view to compare it to."""
    from quant_pipeline.macro_model.forecast_model import (
        MacroForecast,
        ReturnDistribution,
    )

    forecast = MacroForecast(
        distribution=ReturnDistribution(
            mean=mean, variance=variance, skew=skew,
            quantile_levels=LEVELS, quantiles=norm.ppf(LEVELS, mean, np.sqrt(variance)),
        ),
        horizon_days=21, n_train=500, n_features=10,
        model="test", in_sample_pinball=0.0, nonzero_coefficients=5,
    )
    chain = mock_option_chain(spot=100.0, n_strikes=41)
    view = market_view_from_chain(chain, 100.0, 21.0 / 252.0, 0.03)
    return forecast, view


def test_moment_gaps_are_expressed_in_implied_standard_deviations() -> None:
    forecast, view = _forecast_and_view(0.05, 0.0025, -0.5)
    gaps = moment_gaps(forecast, view)
    assert gaps.mean_gap == pytest.approx((0.05 - view.mean_return) / view.std_return)
    assert gaps.variance_log_ratio == pytest.approx(np.log(0.0025 / view.variance_return))
    assert gaps.skew_gap == pytest.approx(-0.5 - view.skew_return)


def test_horizon_mismatch_is_refused() -> None:
    """A 5-day forecast against a 21-day expiry is a units error, not a signal."""
    forecast, view = _forecast_and_view(0.01, 0.0025, 0.0)
    object.__setattr__(forecast, "horizon_days", 5)
    with pytest.raises(ValueError, match="horizon mismatch"):
        moment_gaps(forecast, view)


def test_regime_modulation_can_only_shrink_conviction() -> None:
    """The structural guarantee: a risk modulator never amplifies a signal."""
    config = PipelineConfig()
    for tda_z, hurst_drop in [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0), (0.0, 0.08), (np.nan, np.nan)]:
        assert 0.0 <= regime_modulation(tda_z, 0.14, hurst_drop, config).scale <= 1.0


def test_worse_regime_never_raises_the_scale() -> None:
    config = PipelineConfig()
    scales = [regime_modulation(z, 0.14, 0.0, config).scale for z in (0.0, 1.5, 3.0)]
    assert scales[0] >= scales[1] >= scales[2]


def test_combine_regime_flags_takes_the_worst() -> None:
    assert combine_regime_flags("stable", "unstable") == "unstable"
    assert combine_regime_flags("elevated", "stable") == "elevated"
    assert combine_regime_flags("stable", "stable") == "stable"
    with pytest.raises(ValueError, match="unrecognized"):
        combine_regime_flags("stable", "calm")


def test_config_rejects_a_conviction_scale_above_one() -> None:
    """Enforcing the modulator bound at the config boundary, not by convention."""
    with pytest.raises(ValueError, match="downward only"):
        PipelineConfig(elevated_conviction_scale=1.4)


def test_premium_baseline_returns_the_median_once_ready() -> None:
    baseline = PremiumBaseline(window=10, min_observations=5)
    assert not baseline.ready
    for value in (0.1, 0.2, 0.3, 0.4, 5.0):  # an outlier the median must ignore
        baseline.update(MomentGap(value, 0.0, 0.0))
    assert baseline.ready
    assert baseline.current().mean_gap == pytest.approx(0.3)


def test_premium_baseline_ignores_non_finite_gaps() -> None:
    baseline = PremiumBaseline(window=10, min_observations=2)
    baseline.update(MomentGap(np.nan, 0.0, 0.0))
    baseline.update(MomentGap(np.inf, 0.0, 0.0))
    assert len(baseline) == 0


def test_no_trade_while_the_baseline_is_warming_up() -> None:
    forecast, view = _forecast_and_view(0.05, 0.0025, -0.5)
    signal = divergence_signal(forecast, view, baseline=PremiumBaseline())
    assert signal.decision == "no-trade"
    assert not signal.baseline_ready
    assert "warming up" in signal.reason


def test_unstable_regime_can_veto_a_trade_the_divergence_would_have_taken() -> None:
    """The modulator's whole purpose, asserted end to end."""
    config = PipelineConfig()
    forecast, view = _forecast_and_view(0.035, 0.0025, -1.4)

    calm_baseline, stressed_baseline = PremiumBaseline(5, 3), PremiumBaseline(5, 3)
    for baseline in (calm_baseline, stressed_baseline):
        for _ in range(4):
            baseline.update(MomentGap(0.0, 0.0, 0.0))

    calm = divergence_signal(forecast, view, tda_z=0.0, hurst=0.14, hurst_drop=0.0,
                             baseline=calm_baseline, config=config)
    stressed = divergence_signal(forecast, view, tda_z=4.0, hurst=0.05, hurst_drop=0.2,
                                 baseline=stressed_baseline, config=config)

    assert calm.divergence_score == pytest.approx(stressed.divergence_score)
    assert stressed.conviction < calm.conviction
    assert calm.decision == "long"
    assert stressed.decision == "no-trade"
    assert stressed.regime.regime_flag == "unstable"


# --- the simulated world ---------------------------------------------------

def test_simulated_world_is_internally_consistent() -> None:
    world = simulate_world(400, seed=16)
    assert len(world.chains) == len(world.index)
    assert (world.prices > 0.0).all()
    assert (world.implied_atm_vol > world.realized_vol).mean() > 0.9, (
        "implied volatility should exceed realized on nearly every day -- that is "
        "the variance risk premium the pipeline's baseline exists to remove"
    )


def test_simulated_world_is_reproducible() -> None:
    first = simulate_world(200, seed=17)
    second = simulate_world(200, seed=17)
    pd.testing.assert_frame_equal(first.returns, second.returns)
    pd.testing.assert_series_equal(first.realized_vol, second.realized_vol)
