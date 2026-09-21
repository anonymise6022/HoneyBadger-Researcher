"""Forecast a full return distribution conditioned on macro features.

A point forecast of the mean is useless against an options market. The
market prices an entire distribution, so to disagree with it usefully the
macro side has to produce one too -- and specifically has to produce the
same three numbers the risk-neutral density gives up: mean, variance and
skew.

The estimator is **linear quantile regression** (Koenker & Bassett 1978).
For each level tau it solves

    min_beta  sum_i rho_tau(y_i - x_i'beta),    rho_tau(u) = u(tau - 1{u<0})

which is a linear program, exactly and without iteration. Fitting several
taus traces out the conditional quantile function, and the distribution is
reconstructed from it. Why this rather than fitting a parametric density:

* No distributional assumption. A Gaussian or mixture fit imposes shape on
  the tails; the quantile function reports what the data showed there.
* Each quantile has its own slope vector, so the model can say that credit
  spreads move the 5th percentile hard and the median barely at all --
  which is the entire content of a "macro says the left tail is fatter than
  the market thinks" signal, and is exactly what a mean-and-variance model
  cannot express.
* It is convex, so there is no local optimum to land in.

The cost is that quantile regression fitted independently per tau can
*cross* (an estimated 25th percentile above the 50th) in regions of feature
space with little data. Crossings are repaired by the monotone
rearrangement of Chernozhukov, Fernandez-Val & Galichon (2010), which on a
finite grid of levels is exactly sorting the predicted values. Sorting is
not the same as a running maximum, and the difference matters: a running
maximum over a badly crossed, decreasing set of predictions flattens every
level onto the first, producing a degenerate distribution with zero
variance. Sorting permutes the predictions instead, so it preserves their
spread and cannot manufacture a point mass. It also comes with the paper's
guarantee that the rearranged curve is no further from the true quantile
function than the unsorted one.

**Regularization.** Roughly a hundred macro features over a few hundred
training rows will fit noise. An L1 penalty on the slopes keeps the linear
program a linear program -- split beta into positive and negative parts --
while driving most coefficients to exactly zero, which is the right prior
for macro data where a handful of the hundred columns carry everything and
the rest are lagged copies of each other.

**Reconstructing moments.** Quantiles are mapped into Gaussian z-space,
z_i = Phi^{-1}(tau_i), and interpolated there with a monotone cubic
(PCHIP). A Gaussian conditional distribution is exactly linear in that
space, so interpolation error is zero in the null case and small nearby;
the tails extend linearly, which is the same as assuming Gaussian tails
beyond the outermost fitted quantile. Moments are then

    E[Y^k] = integral g(z)^k phi(z) dz

evaluated by Gauss-Hermite quadrature, which is exact for the linear tails
and near-exact for the piecewise-cubic interior. The alternative -- summing
over a uniform grid in tau -- truncates the tails, and truncation biases
variance down and skew toward zero, which would systematically understate
disagreement with the market.

References: Koenker & Bassett (1978), Econometrica 46(1) 33-50; Koenker
(2005), *Quantile Regression*; Chernozhukov, Fernandez-Val & Galichon
(2010), Econometrica 78(3) 1093-1125.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linprog
from scipy.stats import norm

__all__ = [
    "GaussianBaselineForecaster",
    "MacroForecast",
    "QuantileRegressionForecaster",
    "ReturnDistribution",
    "forward_return_target",
    "moments_from_quantiles",
]

_DEFAULT_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
_QUADRATURE_NODES = 96
_MIN_STD = 1e-12
_MIN_TRAIN_MULTIPLE = 3  # require n >= 3p before trusting a quantile fit


@dataclass(frozen=True)
class ReturnDistribution:
    """A forecast return distribution, summarized by moments and quantiles.

    The moments are the interface to `ensemble.divergence_signal`; the
    quantiles are kept so the reconstruction can be inspected or re-plotted
    without refitting.

    Attributes
    ----------
    mean, variance, skew : moments over the forecast horizon, in simple
        return units (0.01 = 1%). Skew is standardized (dimensionless).
    quantile_levels : the taus that were estimated.
    quantiles : the predicted values at those taus, monotone by construction.
    """

    mean: float
    variance: float
    skew: float
    quantile_levels: tuple[float, ...]
    quantiles: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.variance <= 0.0:
            raise ValueError(f"variance must be strictly positive, got {self.variance}")
        if len(self.quantile_levels) != len(self.quantiles):
            raise ValueError(
                f"quantile_levels and quantiles must have equal length; got "
                f"{len(self.quantile_levels)} and {len(self.quantiles)}"
            )

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def quantile(self, level: float) -> float:
        """Interpolate the quantile function at an arbitrary level in (0, 1).

        Uses the same Gaussian-z-space monotone interpolation the moments
        were computed from, so this and `mean`/`variance` describe one
        consistent distribution rather than two approximations of it.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie strictly in (0, 1), got {level}")
        return float(_quantile_function(np.asarray(self.quantile_levels), self.quantiles)(
            norm.ppf(level)
        ))


@dataclass(frozen=True)
class MacroForecast:
    """A distribution plus the diagnostics needed to judge whether to use it.

    Attributes
    ----------
    distribution : the forecast itself.
    horizon_days : business days the forecast covers. Must match the option
        expiry it is compared against.
    n_train : training rows the fit used.
    n_features : columns after dropping constant ones.
    model : "quantile_regression" or "gaussian_baseline".
    in_sample_pinball : mean pinball loss across levels on the training set.
        In-sample, so it measures fit, not skill -- a falling value with
        more features is overfitting, not improvement.
    nonzero_coefficients : slopes the L1 penalty left non-zero, summed over
        levels. Near the feature count means the penalty is doing nothing.
    consensus_is_modelled : carried through from the feature stage.
    """

    distribution: ReturnDistribution
    horizon_days: int
    n_train: int
    n_features: int
    model: str
    in_sample_pinball: float
    nonzero_coefficients: int
    consensus_is_modelled: bool = False


def forward_return_target(returns: pd.Series, horizon: int) -> pd.Series:
    """Cumulative simple return over the next `horizon` periods.

    target_t = prod_{j=1..h}(1 + r_{t+j}) - 1, so the value at t describes
    the future and is NaN for the final `horizon` rows. Those rows must be
    dropped before fitting; `QuantileRegressionForecaster.fit` does that
    automatically, which is the only thing standing between this function
    and a look-ahead bug.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 period, got {horizon}")
    if len(returns) <= horizon:
        raise ValueError(
            f"need more than {horizon} observations to form a {horizon}-period forward "
            f"return, got {len(returns)}"
        )
    gross = (1.0 + returns).cumprod()
    return (gross.shift(-horizon) / gross - 1.0).rename(f"fwd_return_{horizon}d")


def _quantile_function(
    levels: NDArray[np.float64], quantiles: NDArray[np.float64]
) -> PchipInterpolator:
    """Monotone interpolant of the quantile function in Gaussian z-space.

    Returns a callable g with g(Phi^{-1}(tau_i)) = q_i, extended linearly
    beyond the outermost node using the interpolant's own endpoint slope.
    `PchipInterpolator(extrapolate=True)` continues the end cubics instead,
    which can turn over and break monotonicity far out in the tail, so the
    extension is handled explicitly.
    """
    if levels.size < 2:
        raise ValueError(f"need at least 2 quantile levels to interpolate, got {levels.size}")
    z = norm.ppf(levels)
    # Monotone rearrangement = sorting on a finite grid; see the module
    # docstring on why this is not a running maximum.
    monotone = np.sort(np.asarray(quantiles, dtype=np.float64))
    interior = PchipInterpolator(z, monotone, extrapolate=False)
    slopes = interior.derivative()(np.array([z[0], z[-1]]))
    # A flat end slope would extrapolate to a constant tail (zero density
    # out there); floor it at a small positive value so the tail stays
    # strictly increasing and the distribution stays non-degenerate.
    spread = max(float(monotone[-1] - monotone[0]), _MIN_STD)
    slopes = np.maximum(slopes, 1e-3 * spread / max(float(z[-1] - z[0]), _MIN_STD))

    def evaluate(query: ArrayLike) -> NDArray[np.float64]:
        q = np.atleast_1d(np.asarray(query, dtype=np.float64))
        out = np.empty_like(q)
        below, above = q < z[0], q > z[-1]
        middle = ~(below | above)
        out[middle] = interior(q[middle])
        out[below] = monotone[0] + slopes[0] * (q[below] - z[0])
        out[above] = monotone[-1] + slopes[1] * (q[above] - z[-1])
        return out

    return evaluate  # type: ignore[return-value]


def moments_from_quantiles(
    levels: ArrayLike, quantiles: ArrayLike, n_nodes: int = _QUADRATURE_NODES
) -> tuple[float, float, float]:
    """Reconstruct (mean, variance, skew) from a set of predicted quantiles.

    Parameters
    ----------
    levels : quantile levels in (0, 1), strictly increasing.
    quantiles : predicted values; rearranged to be monotone if they cross.
    n_nodes : Gauss-Hermite nodes. 96 is far past convergence for a smooth
        quantile function and costs nothing at this size.

    Raises ValueError for unsorted or out-of-range levels, mismatched
    lengths, non-finite quantiles, or a reconstruction whose variance
    collapses to zero (every quantile identical).
    """
    tau = np.asarray(levels, dtype=np.float64).ravel()
    q = np.asarray(quantiles, dtype=np.float64).ravel()
    if tau.shape != q.shape:
        raise ValueError(f"levels and quantiles must match; got {tau.size} and {q.size}")
    if np.any(tau <= 0.0) or np.any(tau >= 1.0):
        raise ValueError(f"levels must lie strictly in (0, 1), got range [{tau.min()}, {tau.max()}]")
    if np.any(np.diff(tau) <= 0.0):
        raise ValueError("levels must be strictly increasing")
    if not np.all(np.isfinite(q)):
        raise ValueError("quantiles must all be finite")
    if n_nodes < 8:
        raise ValueError(f"n_nodes must be at least 8 for a usable quadrature, got {n_nodes}")

    g = _quantile_function(tau, q)
    # Gauss-Hermite integrates f(x) exp(-x^2); the substitution z = sqrt(2)x
    # turns that into an expectation under the standard normal density.
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    z = np.sqrt(2.0) * nodes
    probability = weights / np.sqrt(np.pi)

    values = g(z)
    mean = float(np.sum(probability * values))
    centered = values - mean
    variance = float(np.sum(probability * centered**2))
    if variance <= _MIN_STD:
        raise ValueError(
            f"reconstructed variance is degenerate ({variance:.3e}); the predicted "
            "quantiles are effectively identical, so no distribution can be formed"
        )
    skew = float(np.sum(probability * centered**3) / variance**1.5)
    return mean, variance, skew


def _pinball_loss(
    y: NDArray[np.float64], predictions: NDArray[np.float64], levels: NDArray[np.float64]
) -> float:
    """Mean pinball loss over samples and levels; the quantile-fit objective."""
    residual = y[:, None] - predictions
    return float(np.mean(np.maximum(levels * residual, (levels - 1.0) * residual)))


class QuantileRegressionForecaster:
    """L1-regularized linear quantile regression over macro features.

    Each level is fitted as a linear program in the variables
    (beta+, beta-, u, v) >= 0, with beta = beta+ - beta-:

        minimize   alpha * sum_j (beta+_j + beta-_j)            [j != intercept]
                 + (1/n) sum_i [tau * u_i + (1 - tau) * v_i]
        subject to Z beta+ - Z beta- + u - v = y

    At the optimum u_i and v_i are the positive and negative parts of the
    residual, so the second term is exactly the mean pinball loss. Dividing
    it by n keeps `alpha` meaning the same thing as the training window
    grows. The constraint matrix is sparse (the two identity blocks are
    most of it), which is what makes a hundred features and several hundred
    rows solve in milliseconds rather than seconds.

    Features are standardized before fitting -- an L1 penalty applied to
    raw columns would penalize a yield in percent and a z-score at wildly
    different effective strengths -- and the intercept is left unpenalized,
    since shrinking it toward zero would bias the whole distribution toward
    zero return.
    """

    def __init__(
        self,
        quantile_levels: tuple[float, ...] = _DEFAULT_LEVELS,
        alpha: float = 1e-3,
        max_features: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        quantile_levels : levels to fit, strictly increasing, in (0, 1).
            More levels resolve the shape better but each costs an LP;
            seven is enough to pin down mean, spread and asymmetry.
        alpha : L1 penalty strength on standardized slopes. Zero recovers
            unpenalized quantile regression, which will overfit here.
        max_features : optionally keep only the this-many features most
            correlated with the target, screened before fitting. A blunt
            instrument, but it bounds solve time on very wide inputs.
        """
        if len(quantile_levels) < 2:
            raise ValueError(
                f"need at least 2 levels to reconstruct a distribution, got {len(quantile_levels)}"
            )
        if any(not 0.0 < tau < 1.0 for tau in quantile_levels):
            raise ValueError(f"levels must lie strictly in (0, 1), got {quantile_levels}")
        if list(quantile_levels) != sorted(quantile_levels):
            raise ValueError("quantile_levels must be sorted ascending")
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if max_features is not None and max_features < 1:
            raise ValueError(f"max_features must be at least 1 when given, got {max_features}")

        self.quantile_levels = tuple(float(tau) for tau in quantile_levels)
        self.alpha = float(alpha)
        self.max_features = max_features
        self.feature_names_: tuple[str, ...] = ()
        self.coefficients_: NDArray[np.float64] | None = None
        self.intercepts_: NDArray[np.float64] | None = None
        self.center_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.n_train_: int = 0
        self.in_sample_pinball_: float = float("nan")

    def _screen(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """Drop constant columns, then optionally keep the top correlates."""
        varying = [c for c in X.columns if float(X[c].std(ddof=0)) > _MIN_STD]
        if not varying:
            raise ValueError(
                "every feature column is constant over the training window; there is "
                "nothing to condition on"
            )
        if self.max_features is None or len(varying) <= self.max_features:
            return varying
        correlation = X[varying].corrwith(y).abs().fillna(0.0)
        return list(correlation.nlargest(self.max_features).index)

    def _solve(self, Z: NDArray[np.float64], y: NDArray[np.float64], tau: float) -> NDArray[np.float64]:
        """Solve one level's linear program; returns beta including intercept."""
        n, p = Z.shape
        penalty = np.full(p, self.alpha)
        penalty[0] = 0.0  # the intercept column is not shrunk
        cost = np.concatenate([penalty, penalty, np.full(n, tau / n), np.full(n, (1.0 - tau) / n)])

        sparse_Z = sparse.csr_matrix(Z)
        identity = sparse.identity(n, format="csr")
        A_eq = sparse.hstack([sparse_Z, -sparse_Z, identity, -identity], format="csr")

        result = linprog(cost, A_eq=A_eq, b_eq=y, bounds=(0.0, None), method="highs")
        if not result.success:
            raise RuntimeError(
                f"quantile regression at tau={tau} failed to solve: {result.message}. "
                "This usually means the design matrix contains non-finite values."
            )
        return result.x[:p] - result.x[p : 2 * p]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileRegressionForecaster:
        """Fit every level on the rows where features and target are complete.

        Rows with any NaN in either are dropped -- which silently removes
        the last `horizon` rows, whose forward return does not exist yet,
        and that is precisely the point.

        Raises ValueError if fewer than `3 * n_features` usable rows remain,
        since a quantile fit on less than that is noise with a confidence
        interval.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame of named features, got {type(X).__name__}")
        aligned = X.join(y.rename("__target__"), how="inner").replace([np.inf, -np.inf], np.nan)
        aligned = aligned.dropna(how="any")
        if aligned.empty:
            raise ValueError(
                "no rows have both complete features and a realized forward return; "
                "check that the feature and return indexes overlap"
            )

        target = aligned["__target__"]
        design = aligned.drop(columns="__target__")
        kept = self._screen(design, target)
        design = design[kept]

        n = len(design)
        if n < _MIN_TRAIN_MULTIPLE * (len(kept) + 1):
            raise ValueError(
                f"{n} training rows for {len(kept)} features is too few; need at least "
                f"{_MIN_TRAIN_MULTIPLE} rows per parameter. Lengthen the lookback, or cap "
                f"max_features at {max(1, n // _MIN_TRAIN_MULTIPLE - 1)}."
            )

        values = design.to_numpy(dtype=np.float64)
        center = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale = np.where(scale > _MIN_STD, scale, 1.0)
        standardized = (values - center) / scale
        Z = np.column_stack([np.ones(n), standardized])
        y_array = target.to_numpy(dtype=np.float64)

        betas = np.vstack([self._solve(Z, y_array, tau) for tau in self.quantile_levels])

        self.feature_names_ = tuple(kept)
        self.center_, self.scale_ = center, scale
        self.intercepts_ = betas[:, 0]
        self.coefficients_ = betas[:, 1:]
        self.n_train_ = n
        self.in_sample_pinball_ = _pinball_loss(
            y_array, Z @ betas.T, np.asarray(self.quantile_levels)
        )
        return self

    def _require_fit(self) -> None:
        if self.coefficients_ is None or self.intercepts_ is None:
            raise RuntimeError("forecaster is not fitted; call fit(X, y) first")

    def predict_quantiles(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Predict every level for every row; shape (len(X), n_levels).

        Output is monotonically rearranged across levels -- sorted, per the
        Chernozhukov, Fernandez-Val & Galichon (2010) operator -- repairing
        any quantile crossing without collapsing the spread.
        """
        self._require_fit()
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise KeyError(f"input is missing features the model was fitted on: {missing}")
        values = X[list(self.feature_names_)].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "feature values contain NaN or infinity; a forecast cannot be made for a "
                "row with missing conditioning information"
            )
        standardized = (values - self.center_) / self.scale_
        predictions = standardized @ self.coefficients_.T + self.intercepts_
        return np.sort(predictions, axis=1)

    def forecast(
        self, x: pd.Series | pd.DataFrame, horizon_days: int, consensus_is_modelled: bool = False
    ) -> MacroForecast:
        """Forecast the distribution for a single feature row.

        Accepts a Series (one row) or a one-row DataFrame. Raises ValueError
        for a multi-row DataFrame -- use `predict_quantiles` there, since a
        distribution per row is a different object than a single forecast.
        """
        self._require_fit()
        frame = x.to_frame().T if isinstance(x, pd.Series) else x
        if len(frame) != 1:
            raise ValueError(
                f"forecast() summarizes one feature row into one distribution, got {len(frame)} "
                "rows; call predict_quantiles() for a batch"
            )
        quantiles = self.predict_quantiles(frame)[0]
        mean, variance, skew = moments_from_quantiles(self.quantile_levels, quantiles)
        nonzero = int(np.count_nonzero(np.abs(self.coefficients_) > 1e-12))
        return MacroForecast(
            distribution=ReturnDistribution(
                mean=mean,
                variance=variance,
                skew=skew,
                quantile_levels=self.quantile_levels,
                quantiles=quantiles,
            ),
            horizon_days=int(horizon_days),
            n_train=self.n_train_,
            n_features=len(self.feature_names_),
            model="quantile_regression",
            in_sample_pinball=self.in_sample_pinball_,
            nonzero_coefficients=nonzero,
            consensus_is_modelled=consensus_is_modelled,
        )


class GaussianBaselineForecaster:
    """Ridge mean, unconditional residual spread and skew -- the null model.

    Deliberately weaker than quantile regression: the conditional mean moves
    with the features, but the shape of the distribution around it does not.
    It exists for two reasons. It is a fallback when the training window is
    too short for a hundred-feature quantile fit to be meaningful, and it is
    the benchmark that makes the quantile model's extra machinery falsifiable
    -- if a divergence signal built on this performs as well, the conditional
    shape was not adding anything.

    The residual skew is a *sample* third moment of the training residuals,
    so it is one number reused for every forecast, and it is noisy: the
    standard error of a sample skew on n points is about sqrt(6/n).
    """

    def __init__(
        self, quantile_levels: tuple[float, ...] = _DEFAULT_LEVELS, ridge: float = 1.0
    ) -> None:
        if ridge < 0.0:
            raise ValueError(f"ridge must be non-negative, got {ridge}")
        if len(quantile_levels) < 2:
            raise ValueError(f"need at least 2 levels, got {len(quantile_levels)}")
        self.quantile_levels = tuple(float(tau) for tau in quantile_levels)
        self.ridge = float(ridge)
        self.feature_names_: tuple[str, ...] = ()
        self.beta_: NDArray[np.float64] | None = None
        self.center_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.residual_std_: float = float("nan")
        self.residual_skew_: float = float("nan")
        self.n_train_: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GaussianBaselineForecaster:
        """Fit ridge coefficients on standardized features, unpenalized intercept."""
        aligned = X.join(y.rename("__target__"), how="inner").replace([np.inf, -np.inf], np.nan)
        aligned = aligned.dropna(how="any")
        if len(aligned) < 10:
            raise ValueError(f"need at least 10 complete rows to fit, got {len(aligned)}")

        target = aligned["__target__"].to_numpy(dtype=np.float64)
        design = aligned.drop(columns="__target__")
        varying = [c for c in design.columns if float(design[c].std(ddof=0)) > _MIN_STD]
        if not varying:
            raise ValueError("every feature column is constant over the training window")
        design = design[varying]

        values = design.to_numpy(dtype=np.float64)
        center, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
        scale = np.where(scale > _MIN_STD, scale, 1.0)
        standardized = (values - center) / scale

        n, p = standardized.shape
        # Ridge with an unpenalized intercept: centring the target lets the
        # intercept be recovered as the target mean afterwards.
        gram = standardized.T @ standardized + self.ridge * np.eye(p)
        beta = np.linalg.solve(gram, standardized.T @ (target - target.mean()))
        fitted = standardized @ beta + target.mean()
        residuals = target - fitted

        self.feature_names_ = tuple(varying)
        self.beta_ = np.concatenate([[float(target.mean())], beta])
        self.center_, self.scale_ = center, scale
        self.residual_std_ = float(residuals.std(ddof=1))
        centered = residuals - residuals.mean()
        self.residual_skew_ = (
            float(np.mean(centered**3) / max(float(np.std(centered)) ** 3, _MIN_STD))
        )
        self.n_train_ = n
        return self

    def forecast(
        self, x: pd.Series | pd.DataFrame, horizon_days: int, consensus_is_modelled: bool = False
    ) -> MacroForecast:
        """Gaussian forecast centred on the ridge prediction.

        Quantiles are reported as mean + z * residual_std, i.e. ignoring the
        estimated residual skew, while the distribution's `skew` field
        carries it. That inconsistency is intentional and visible: this
        model has no mechanism to place a skewed quantile function, and
        papering over it with a fitted skew-normal would dress up an
        assumption as an estimate.
        """
        if self.beta_ is None:
            raise RuntimeError("forecaster is not fitted; call fit(X, y) first")
        frame = x.to_frame().T if isinstance(x, pd.Series) else x
        if len(frame) != 1:
            raise ValueError(f"forecast() takes one feature row, got {len(frame)}")
        missing = [c for c in self.feature_names_ if c not in frame.columns]
        if missing:
            raise KeyError(f"input is missing features the model was fitted on: {missing}")

        values = frame[list(self.feature_names_)].to_numpy(dtype=np.float64)[0]
        if not np.all(np.isfinite(values)):
            raise ValueError("feature values contain NaN or infinity")
        standardized = (values - self.center_) / self.scale_
        mean = float(self.beta_[0] + standardized @ self.beta_[1:])
        quantiles = mean + norm.ppf(np.asarray(self.quantile_levels)) * self.residual_std_

        return MacroForecast(
            distribution=ReturnDistribution(
                mean=mean,
                variance=float(self.residual_std_**2),
                skew=float(self.residual_skew_),
                quantile_levels=self.quantile_levels,
                quantiles=quantiles,
            ),
            horizon_days=int(horizon_days),
            n_train=self.n_train_,
            n_features=len(self.feature_names_),
            model="gaussian_baseline",
            in_sample_pinball=float("nan"),
            nonzero_coefficients=int(np.count_nonzero(np.abs(self.beta_[1:]) > 1e-12)),
            consensus_is_modelled=consensus_is_modelled,
        )
