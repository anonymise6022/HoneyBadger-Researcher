"""Tests for the quant models, against processes whose answers are known.

The package is labelled alpha because being correctly implemented is not the
same as being useful. These tests establish the first half of that: fed a
simulated random walk, a mean-reverting process or a trending one, each
estimator must return the value theory says it should.

The calibration tests matter most. An estimator that reports "trending" on
independent noise is worse than no estimator, and rescaled-range analysis
does exactly that unless its small-sample bias is corrected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_cli.quant import analyse
from research_cli.quant.mean_reversion import (
    estimate_mean_reversion,
    half_life,
    unit_root_pvalue,
)
from research_cli.quant.regime import (
    estimate_regime,
    expected_rescaled_range,
    hurst_exponent,
    variance_ratio,
)
from research_cli.quant.volatility import (
    close_to_close_volatility,
    ewma_volatility,
    forecast_volatility,
    yang_zhang_volatility,
)

TRADING_DAYS = 252


def _ohlc(closes: np.ndarray, spread: float = 0.004) -> pd.DataFrame:
    """Wrap a close series in plausible OHLC bars."""
    index = pd.bdate_range("2015-01-01", periods=len(closes))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes) * (1 + spread),
            "Low": np.minimum(opens, closes) * (1 - spread),
            "Close": closes,
            "Volume": np.full(len(closes), 1e6),
        },
        index=index,
    )


def _gbm(n: int, vol: float, seed: int = 0) -> pd.DataFrame:
    """Geometric Brownian motion with a known annualized volatility."""
    rng = np.random.default_rng(seed)
    daily = vol / np.sqrt(TRADING_DAYS)
    return _ohlc(100 * np.exp(np.cumsum(rng.normal(0, daily, n))))


# --- volatility ------------------------------------------------------------

@pytest.mark.parametrize("vol", [0.12, 0.25, 0.45])
def test_close_to_close_recovers_the_generating_volatility(vol: float) -> None:
    frame = _gbm(1500, vol, seed=1)
    estimate = close_to_close_volatility(frame, window=1000)
    assert estimate == pytest.approx(vol * 100, rel=0.12)


def test_yang_zhang_is_close_to_close_to_close_on_the_same_data() -> None:
    """They estimate the same quantity; Yang-Zhang just does it more efficiently."""
    frame = _gbm(1500, 0.25, seed=2)
    assert yang_zhang_volatility(frame, 500) == pytest.approx(
        close_to_close_volatility(frame, 500), rel=0.35
    )


def test_yang_zhang_refuses_incomplete_bars() -> None:
    frame = _gbm(300, 0.2)[["Close"]]
    with pytest.raises(ValueError, match="OHLC"):
        yang_zhang_volatility(frame)


def test_ewma_reacts_faster_than_a_long_window() -> None:
    """A volatility jump should show up in EWMA before a 250-day average."""
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.10 / np.sqrt(TRADING_DAYS), 500)
    wild = rng.normal(0, 0.45 / np.sqrt(TRADING_DAYS), 30)
    frame = _ohlc(100 * np.exp(np.cumsum(np.concatenate([calm, wild]))))
    assert ewma_volatility(frame) > close_to_close_volatility(frame, window=400)


def test_ewma_decay_must_be_a_proper_fraction() -> None:
    frame = _gbm(300, 0.2)
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="decay"):
            ewma_volatility(frame, decay=bad)


def test_forecast_reports_which_model_actually_ran() -> None:
    """A GARCH fit that fails must say so, not silently become EWMA."""
    estimate = forecast_volatility(_gbm(1200, 0.22, seed=4))
    assert estimate.forecast_model in {"GARCH(1,1)", "EWMA (GARCH unavailable)"}
    assert estimate.forecast_pct > 0


def test_forecast_needs_enough_history() -> None:
    with pytest.raises(ValueError, match="at least 60"):
        forecast_volatility(_gbm(40, 0.2))


def test_volatility_regime_labels_are_relative_to_the_long_run() -> None:
    estimate = forecast_volatility(_gbm(1200, 0.2, seed=5))
    assert estimate.regime in {"elevated", "normal", "subdued", "unknown"}


# --- regime ----------------------------------------------------------------

def test_expected_rescaled_range_grows_with_n() -> None:
    assert expected_rescaled_range(100) > expected_rescaled_range(20)


def test_hurst_of_independent_noise_is_near_one_half() -> None:
    """The calibration that the Anis-Lloyd correction exists to deliver.

    Uncorrected, rescaled range reports about 0.56 here and, with a
    regression error near 0.01, calls pure noise a five-sigma trend.
    """
    errors = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        h, _, _ = hurst_exponent(pd.Series(rng.standard_normal(4000)))
        errors.append(abs(h - 0.5))
    assert np.mean(errors) < 0.04, f"mean |H - 0.5| was {np.mean(errors):.3f}"


def test_hurst_separates_persistent_from_anti_persistent() -> None:
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(4000)
    persistent = np.zeros(4000)
    anti = np.zeros(4000)
    for i in range(1, 4000):
        persistent[i] = 0.5 * persistent[i - 1] + noise[i]
        anti[i] = -0.5 * anti[i - 1] + noise[i]
    h_persistent, _, _ = hurst_exponent(pd.Series(persistent))
    h_anti, _, _ = hurst_exponent(pd.Series(anti))
    assert 0.5 < h_persistent < 1.0
    assert h_anti < 0.5
    assert h_persistent > h_anti


def test_hurst_rejects_a_constant_series() -> None:
    with pytest.raises(ValueError, match="constant"):
        hurst_exponent(pd.Series(np.ones(500)))


def test_hurst_rejects_too_little_data() -> None:
    with pytest.raises(ValueError, match="at least"):
        hurst_exponent(pd.Series(np.random.default_rng(0).standard_normal(50)))


def test_variance_ratio_of_a_random_walk_is_one() -> None:
    rng = np.random.default_rng(8)
    ratio, z = variance_ratio(pd.Series(rng.standard_normal(4000)), lag=5)
    assert ratio == pytest.approx(1.0, abs=0.08)
    assert abs(z) < 2.0


def test_variance_ratio_detects_both_directions() -> None:
    rng = np.random.default_rng(9)
    noise = rng.standard_normal(4000)
    persistent = np.zeros(4000)
    anti = np.zeros(4000)
    for i in range(1, 4000):
        persistent[i] = 0.5 * persistent[i - 1] + noise[i]
        anti[i] = -0.5 * anti[i - 1] + noise[i]
    assert variance_ratio(pd.Series(persistent))[0] > 1.3
    assert variance_ratio(pd.Series(anti))[0] < 0.8


def test_regime_classifier_does_not_cry_wolf_on_random_walks() -> None:
    """Twelve independent random walks must all read as random walks."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        frame = _ohlc(100 * np.exp(np.cumsum(rng.standard_normal(3000) * 0.01)))
        assert estimate_regime(frame).classification == "random walk", f"seed {seed}"


def test_regime_describe_is_plain_language() -> None:
    frame = _gbm(2000, 0.2, seed=10)
    text = estimate_regime(frame).describe()
    assert "Hurst" in text
    for jargon in ("nan", "None", "classification="):
        assert jargon not in text


# --- mean reversion --------------------------------------------------------

def test_half_life_recovers_a_known_ou_process() -> None:
    """For P_t = b*P_{t-1} + e, the half-life is ln(2) / -ln(b)."""
    rng = np.random.default_rng(11)
    b = 0.95
    values = np.zeros(4000)
    for i in range(1, 4000):
        values[i] = b * values[i - 1] + rng.standard_normal()
    life, fitted_b, _ = half_life(pd.Series(values))
    assert fitted_b == pytest.approx(b, abs=0.03)
    assert life == pytest.approx(np.log(2) / -np.log(b), rel=0.25)


def test_adf_test_is_correctly_sized_on_random_walks() -> None:
    """A 5% test must reject about 5% of true random walks -- not none.

    Asserting on a single seed would be wrong in both directions: it fails
    whenever that seed lands in the rejection region (seed 12 gives 0.042),
    and it would pass for a broken test that never rejects anything. The
    property worth checking is the *rate*.
    """
    pvalues = [
        unit_root_pvalue(pd.Series(np.cumsum(np.random.default_rng(s).standard_normal(3000))))
        for s in range(20)
    ]
    assert all(p is not None for p in pvalues)
    rejections = sum(p < 0.05 for p in pvalues)
    assert rejections <= 4, f"{rejections}/20 is far above a 5% size; the test is too eager"
    assert np.median(pvalues) > 0.2, "most random walks should be nowhere near rejection"


def test_unit_root_is_rejected_for_a_genuine_ou_process() -> None:
    rng = np.random.default_rng(31)
    values = np.zeros(2000)
    for i in range(1, 2000):
        values[i] = 0.95 * values[i - 1] + rng.standard_normal()
    assert unit_root_pvalue(pd.Series(values)) < 0.01


def test_random_walks_are_rarely_called_mean_reverting() -> None:
    """Regression: the naive AR(1) t-test called *every* random walk reverting.

    Near a unit root the OLS estimate of b is biased downward and its
    t-statistic does not follow a normal distribution, so a 1-sigma bar
    fired on all eight of these seeds and reported half-lives of about 140
    days for pure noise. With a proper Dickey-Fuller test the false-positive
    rate falls to roughly the test's own 5% size.
    """
    false_positives = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        frame = _ohlc(100 * np.exp(np.cumsum(rng.standard_normal(2000) * 0.01)))
        estimate = estimate_mean_reversion(frame)
        if estimate.is_mean_reverting:
            false_positives += 1
        else:
            assert estimate.half_life_days is None, (
                "a half-life must not be reported when reversion was not established"
            )
    assert false_positives <= 2, (
        f"{false_positives}/8 random walks were called mean-reverting; a 5% test "
        "should seldom exceed one"
    )


def test_z_score_matches_its_definition() -> None:
    frame = _gbm(500, 0.2, seed=14)
    estimate = estimate_mean_reversion(frame, window=63)
    expected = (estimate.current_price - estimate.rolling_mean) / estimate.rolling_std
    assert estimate.z_score == pytest.approx(expected)


def test_mean_reversion_describe_warns_when_there_is_none() -> None:
    rng = np.random.default_rng(15)
    frame = _ohlc(100 * np.exp(np.cumsum(rng.standard_normal(2000) * 0.01)))
    estimate = estimate_mean_reversion(frame)
    if not estimate.is_mean_reverting:
        assert "no reliable tendency" in estimate.describe()


# --- the bundle ------------------------------------------------------------

def test_analyse_returns_every_model() -> None:
    result = analyse(_gbm(1500, 0.25, seed=16), "TEST")
    assert result.volatility is not None
    assert result.regime is not None
    assert result.mean_reversion is not None
    assert not result.is_empty


def test_one_failing_model_does_not_sink_the_others() -> None:
    """A model that cannot run must cost only itself.

    At 100 bars the regime estimator has too little history for rescaled
    range, while the volatility model (which needs 60) still fits. The point
    is that the failure is recorded and the other models still return.
    """
    result = analyse(_gbm(100, 0.2, seed=17), "TEST")
    assert result.failures, "expected at least one model to fail on 100 bars"
    assert result.regime is None, "rescaled range needs more than 100 bars"
    assert result.volatility is not None, "a failure elsewhere must not sink this"
    assert not result.is_empty
