"""Tests asserting the mathematical identities each module must satisfy.

Every test states a fact about the math, not about the implementation:
put-call parity, Greeks as finite-difference derivatives, SVI's no-arbitrage
bounds, and recovery of a known lognormal. All data is synthetic; no
network, no API keys. Run with `pytest test_math.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from black_scholes import call_price, d1_d2, delta, gamma, put_price, theta, vega
from data_scraper import (
    CHAIN_COLUMNS, OptionsDataError, build_chain, fetch_stock_options_chain,
    synthetic_equity_chain, synthetic_fx_chain,
)
from implied_density import extract_density
from svi_model import SVIParams, fit_svi, svi_total_variance

S, K, T, R, VOL = 100.0, 100.0, 0.5, 0.03, 0.25  # ATM, 6M, 3% rate, 25% vol


# --- black_scholes --------------------------------------------------------

def test_put_call_parity() -> None:
    """C - P = S - K exp(-rT), exactly, at every strike."""
    strikes = np.linspace(60.0, 140.0, 17)
    lhs = call_price(S, strikes, T, R, VOL) - put_price(S, strikes, T, R, VOL)
    np.testing.assert_allclose(lhs, S - strikes * np.exp(-R * T), rtol=1e-12, atol=1e-10)


def test_deep_itm_call_approaches_discounted_intrinsic() -> None:
    """As S/K -> inf the call tends to S - K exp(-rT)."""
    assert call_price(1000.0, K, T, R, VOL) == pytest.approx(1000.0 - K * np.exp(-R * T), rel=1e-9)


def test_delta_matches_numerical_derivative() -> None:
    """delta = dC/dS, against a central difference in spot."""
    h = 1e-4
    numerical = (call_price(S + h, K, T, R, VOL) - call_price(S - h, K, T, R, VOL)) / (2 * h)
    assert delta(S, K, T, R, VOL, "call") == pytest.approx(numerical, rel=1e-6)


def test_gamma_matches_second_derivative_and_is_parity_symmetric() -> None:
    """gamma = d2C/dS2, and call gamma == put gamma (parity kills the linear term)."""
    h = 1e-3
    def second_diff(price_fn) -> float:
        return (price_fn(S + h, K, T, R, VOL) - 2 * price_fn(S, K, T, R, VOL)
                + price_fn(S - h, K, T, R, VOL)) / h**2
    analytic = gamma(S, K, T, R, VOL)
    assert analytic == pytest.approx(second_diff(call_price), rel=1e-4)
    assert analytic == pytest.approx(second_diff(put_price), rel=1e-4)


def test_vega_matches_numerical_derivative_and_peaks_atm() -> None:
    """vega = dC/dsigma, and is maximized near the forward-at-the-money strike."""
    h = 1e-6
    numerical = (call_price(S, K, T, R, VOL + h) - call_price(S, K, T, R, VOL - h)) / (2 * h)
    assert vega(S, K, T, R, VOL) == pytest.approx(numerical, rel=1e-6)
    strikes = np.linspace(70.0, 130.0, 61)
    assert strikes[int(np.argmax(vega(S, strikes, T, R, VOL)))] == pytest.approx(S * np.exp(R * T), abs=2.0)


def test_theta_negative_and_delta_bounds() -> None:
    """ATM calls decay; call delta lies in (0,1) with delta_call - delta_put = 1."""
    assert theta(S, K, T, R, VOL, "call") < 0.0
    strikes = np.linspace(50.0, 150.0, 21)
    call_d = delta(S, strikes, T, R, VOL, "call")
    assert np.all((call_d > 0.0) & (call_d < 1.0))
    np.testing.assert_allclose(call_d - delta(S, strikes, T, R, VOL, "put"), 1.0, rtol=1e-12)


def test_d2_is_below_d1_by_vol_sqrt_time() -> None:
    d1, d2 = d1_d2(S, K, T, R, VOL)
    assert d1 - d2 == pytest.approx(VOL * np.sqrt(T), rel=1e-12)


@pytest.mark.parametrize("bad", [{"S": -1.0}, {"K": 0.0}, {"T": -0.5}, {"sigma": 0.0}, {"S": np.nan}])
def test_domain_violations_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        call_price(**({"S": S, "K": K, "T": T, "r": R, "sigma": VOL} | bad))


def test_mismatched_shapes_rejected() -> None:
    with pytest.raises(ValueError, match="broadcast-compatible"):
        call_price(np.array([100.0, 101.0]), np.array([90.0, 100.0, 110.0]), T, R, VOL)


# --- svi_model ------------------------------------------------------------

def _reference_smile() -> tuple[np.ndarray, np.ndarray, SVIParams]:
    truth = SVIParams(a=0.04, b=0.40, rho=-0.30, m=0.0, sigma=0.10)
    k = np.linspace(-0.5, 0.5, 25)
    return k, svi_total_variance(k, truth), truth


def test_svi_recovers_known_parameters() -> None:
    """Fitting noiseless SVI data returns the generating parameters."""
    k, w, truth = _reference_smile()
    fit = fit_svi(k, w)
    np.testing.assert_allclose(fit.params.as_array(), truth.as_array(), atol=1e-4)
    assert fit.rmse < 1e-6


def test_svi_robust_to_small_noise() -> None:
    """With 1e-4 quote noise the skew stays close and residuals stay small."""
    k, w, truth = _reference_smile()
    fit = fit_svi(k, w + np.random.default_rng(0).normal(0.0, 1e-4, size=w.shape))
    assert fit.rmse < 1e-3
    assert fit.params.rho == pytest.approx(truth.rho, abs=0.1)
    assert fit.residuals.shape == k.shape


def test_svi_fit_respects_no_arbitrage_bounds() -> None:
    """b >= 0, |rho| < 1, sigma > 0, and min_k w(k) >= 0."""
    p = fit_svi(*_reference_smile()[:2]).params
    assert p.b >= 0.0 and abs(p.rho) < 1.0 and p.sigma > 0.0
    assert p.min_total_variance() >= -1e-10


def test_svi_total_variance_positive_and_convex() -> None:
    """w(k) > 0 everywhere and curves upward, given b > 0 and sigma > 0."""
    truth = _reference_smile()[2]
    w = svi_total_variance(np.linspace(-2.0, 2.0, 401), truth)
    assert np.all(w > 0.0)
    assert np.all(np.diff(w, n=2) > -1e-12)


def test_svi_minimum_matches_analytic_vertex() -> None:
    """min_total_variance() equals the numerical minimum of w(k)."""
    truth = _reference_smile()[2]
    numerical = float(np.min(svi_total_variance(np.linspace(-5.0, 5.0, 200_001), truth)))
    assert truth.min_total_variance() == pytest.approx(numerical, abs=1e-6)


def test_svi_wing_slopes_match_asymptotes() -> None:
    """w(k)/k -> b(rho +/- 1) in the far wings."""
    truth = _reference_smile()[2]
    left, right = truth.wing_slopes()
    assert svi_total_variance(1e6, truth) / 1e6 == pytest.approx(right, rel=1e-5)
    assert svi_total_variance(-1e6, truth) / 1e6 == pytest.approx(-left, rel=1e-5)


def test_svi_rejects_degenerate_inputs() -> None:
    """Too few points, ragged lengths, and non-positive variance all raise."""
    k, w, _ = _reference_smile()
    with pytest.raises(ValueError, match="at least 5"):
        fit_svi(k[:4], w[:4])
    with pytest.raises(ValueError, match="same length"):
        fit_svi(k, w[:-1])
    bad = w.copy()
    bad[0] = -0.01
    with pytest.raises(ValueError, match="strictly positive"):
        fit_svi(k, bad)


# --- implied_density ------------------------------------------------------

def test_density_recovers_lognormal_moments() -> None:
    """BL on Black-Scholes prices returns the model's own lognormal:
    E[S_T] = S exp(rT), Var[S_T] = S^2 exp(2rT)(exp(sigma^2 T) - 1)."""
    strikes = np.linspace(20.0, 320.0, 3001)  # ~+/-5 sd, so truncation bias is small
    dist = extract_density(strikes, call_price(S, strikes, T, R, VOL), T, R)
    assert dist.mean == pytest.approx(S * np.exp(R * T), rel=1e-3)
    assert dist.variance == pytest.approx(
        S**2 * np.exp(2 * R * T) * (np.exp(VOL**2 * T) - 1.0), rel=1e-2
    )
    assert dist.total_mass == pytest.approx(1.0, abs=1e-3)


def test_density_nonnegative_and_integrates_to_one() -> None:
    strikes = np.linspace(40.0, 200.0, 801)
    dist = extract_density(strikes, call_price(S, strikes, T, R, VOL), T, R)
    assert np.all(dist.density >= 0.0)
    assert np.trapezoid(dist.density / dist.total_mass, dist.strikes) == pytest.approx(1.0, rel=1e-9)


def test_lognormal_density_is_right_skewed() -> None:
    """Lognormals are right-skewed in price space, more so at higher vol."""
    strikes = np.linspace(20.0, 320.0, 2001)
    low = extract_density(strikes, call_price(S, strikes, T, R, 0.20), T, R)
    high = extract_density(strikes, call_price(S, strikes, T, R, 0.40), T, R)
    assert 0.0 < low.skew < high.skew


def test_density_handles_non_uniform_grid() -> None:
    """The non-uniform second difference agrees with the uniform result."""
    uniform = np.linspace(40.0, 200.0, 401)
    clustered = np.unique(np.concatenate([
        np.linspace(40.0, 80.0, 60), np.linspace(80.0, 130.0, 260), np.linspace(130.0, 200.0, 90)
    ]))
    dense = extract_density(uniform, call_price(S, uniform, T, R, VOL), T, R)
    sparse = extract_density(clustered, call_price(S, clustered, T, R, VOL), T, R)
    assert sparse.mean == pytest.approx(dense.mean, rel=1e-3)


def test_density_rejects_bad_grids_and_flat_curves() -> None:
    """Short, unsorted, ragged, and zero-convexity inputs raise clearly."""
    strikes = np.linspace(60.0, 140.0, 51)
    prices = call_price(S, strikes, T, R, VOL)
    with pytest.raises(ValueError, match=">= 3 strikes"):
        extract_density(strikes[:2], prices[:2], T, R)
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_density(strikes[::-1], prices, T, R)
    with pytest.raises(ValueError, match="same length"):
        extract_density(strikes, prices[:-1], T, R)
    with pytest.raises(ValueError, match="T must be positive"):
        extract_density(strikes, prices, -1.0, R)
    with pytest.raises(ValueError, match="negligible mass"):
        extract_density(strikes, 100.0 - strikes, T, R)  # linear => no density


# --- data_scraper ---------------------------------------------------------

def test_synthetic_chains_match_schema() -> None:
    for chain in (synthetic_equity_chain(), synthetic_fx_chain()):
        assert list(chain.columns) == CHAIN_COLUMNS
        assert chain["strike"].is_monotonic_increasing
        assert (chain["call_price"] > 0.0).all() and (chain["put_price"] > 0.0).all()


def test_synthetic_chain_obeys_parity() -> None:
    """Chain prices are internally consistent, so SVI/BL tests are meaningful."""
    chain = synthetic_equity_chain(spot=100.0, expiry_years=0.5, rate=0.03)
    np.testing.assert_allclose(
        chain["call_price"] - chain["put_price"],
        100.0 - chain["strike"] * np.exp(-0.03 * 0.5),
        atol=1e-10,
    )


def test_fx_chain_is_tighter_than_equity_chain() -> None:
    """FX defaults span a narrower strike band, matching G10 vol levels."""
    def relative_width(chain) -> float:
        return (chain["strike"].max() - chain["strike"].min()) / chain["strike"].mean()
    assert relative_width(synthetic_fx_chain()) < relative_width(synthetic_equity_chain())


def test_build_chain_rejects_ragged_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        build_chain([100.0, 110.0], [5.0], [2.0, 1.0], 0.2, "TEST")
    with pytest.raises(ValueError, match="zero strikes"):
        build_chain([], [], [], 0.2, "TEST")


def test_fetch_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing credentials fail loudly rather than silently returning mocks."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(OptionsDataError, match="no Polygon.io API key"):
        fetch_stock_options_chain("AAPL")
