"""Out-of-sample checks for a fitted quantile forecaster.

The tests in `tests/unit/` assert that the estimator does what its math
says: that a Gaussian input round-trips exactly, that quantiles come back
monotone, that the penalty shrinks. None of that answers the question a
forecast has to answer, which is whether the distribution it predicts is
*calibrated* -- whether the level it calls the 5th percentile is in fact
exceeded 95% of the time on data the fit never saw.

Two diagnostics, both out-of-sample:

**Coverage.** For each level tau, the fraction of realized outcomes at or
below the predicted tau-quantile. A calibrated model puts that fraction at
tau. Positive error means the quantile sits too high (too much mass below
it); negative means too low. Coverage is necessary and not sufficient -- a
model that ignores its features entirely and predicts the unconditional
quantiles is perfectly calibrated and useless, which is why pinball loss is
reported beside it.

**Pinball loss.** The estimator's own objective, evaluated on held-out
data. Compared against the same loss for a model that predicts the
*unconditional* training quantiles, it says whether conditioning on the
features bought anything. A skill score at or below zero means it did not.

**The overlap correction matters more than anything else here.** When the
target is an h-day forward return, consecutive rows share h-1 days of
returns and are not independent observations. Treating 400 overlapping
daily rows as 400 draws understates the standard error by roughly sqrt(h)
-- a factor of 4.6 at a one-month horizon -- and turns ordinary sampling
noise into confident-looking miscalibration. `horizon` divides the sample
into effective observations before any standard error is computed. The
correction is crude (it assumes rows h apart are independent, which is
optimistic at the tails) but it is far closer than ignoring the problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ["CoverageReport", "coverage_report", "pinball_loss"]


@dataclass(frozen=True)
class CoverageReport:
    """Out-of-sample calibration and skill for one fitted forecaster.

    Attributes
    ----------
    levels : the quantile levels checked.
    empirical : realized coverage at each level.
    error : `empirical - levels`. Positive means the quantile is too high.
    standard_error : one standard error of `empirical` under independence
        of the *effective* sample, i.e. sqrt(tau(1-tau)/n_effective).
    n_test, n_effective : held-out rows, and rows after the overlap
        correction.
    pinball, baseline_pinball : held-out pinball loss for the model and for
        the unconditional training quantiles.
    """

    levels: tuple[float, ...]
    empirical: NDArray[np.float64]
    error: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    n_test: int
    n_effective: int
    pinball: float
    baseline_pinball: float

    @property
    def skill_score(self) -> float:
        """1 - pinball/baseline. Positive means conditioning helped."""
        if self.baseline_pinball <= 0.0:
            return float("nan")
        return float(1.0 - self.pinball / self.baseline_pinball)

    @property
    def worst_level(self) -> float:
        """The level with the largest coverage error, in standard errors."""
        z = np.abs(self.error) / np.where(self.standard_error > 0, self.standard_error, np.inf)
        return float(self.levels[int(np.argmax(z))])

    def is_calibrated(self, tolerance_se: float = 2.0) -> bool:
        """True when every level's coverage error is inside `tolerance_se`."""
        return bool(
            np.all(np.abs(self.error) <= tolerance_se * self.standard_error)
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "level": self.levels,
                "empirical": self.empirical,
                "error": self.error,
                "standard_error": self.standard_error,
            }
        ).set_index("level")

    def render(self, title: str = "coverage") -> str:
        """A fixed-width text block, for printing next to a demo run."""
        header = (
            f"{title}   n_test={self.n_test}  n_effective={self.n_effective}  "
            f"pinball={self.pinball:.5f}  skill={self.skill_score:+.3f}"
        )
        lines = [header, f"  {'level':>6}  {'empirical':>10}  {'error':>8}  {'2se':>7}"]
        for level, empirical, error, se in zip(
            self.levels, self.empirical, self.error, self.standard_error
        ):
            flag = "  <-- outside 2se" if abs(error) > 2.0 * se else ""
            lines.append(f"  {level:>6.2f}  {empirical:>10.3f}  {error:>+8.3f}  {2*se:>7.3f}{flag}")
        return "\n".join(lines)


def pinball_loss(
    y: NDArray[np.float64] | pd.Series,
    predictions: NDArray[np.float64],
    levels: tuple[float, ...],
) -> float:
    """Mean pinball loss, averaged over observations and levels.

    rho_tau(u) = u(tau - 1{u < 0}), the loss quantile regression minimizes.
    """
    actual = np.asarray(y, dtype=np.float64).ravel()
    predicted = np.asarray(predictions, dtype=np.float64)
    if predicted.shape != (actual.size, len(levels)):
        raise ValueError(
            f"predictions must have shape (n, n_levels) = ({actual.size}, {len(levels)}), "
            f"got {predicted.shape}"
        )
    residual = actual[:, None] - predicted
    taus = np.asarray(levels, dtype=np.float64)
    return float(np.mean(np.maximum(taus * residual, (taus - 1.0) * residual)))


def coverage_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series | None = None,
    horizon: int = 1,
) -> CoverageReport:
    """Score a fitted forecaster on held-out data.

    Parameters
    ----------
    model : anything with `predict_quantiles(X)` and `quantile_levels`,
        already fitted -- `QuantileRegressionForecaster` or a stand-in.
    X_test, y_test : held-out features and realized outcomes. Rows with a
        missing feature or outcome are dropped.
    y_train : the training outcomes, used to build the unconditional
        baseline. When omitted the baseline is taken from `y_test` itself,
        which flatters the model less than it sounds -- an in-sample
        baseline is a *harder* benchmark to beat.
    horizon : the forecast horizon in rows. Values above 1 mark the target
        as overlapping and shrink the effective sample accordingly; see the
        module docstring on why this dominates the result.

    Raises ValueError for an empty test set or a horizon below 1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 row, got {horizon}")

    levels = tuple(float(t) for t in model.quantile_levels)
    joined = X_test.join(y_test.rename("__y__"), how="inner").replace(
        [np.inf, -np.inf], np.nan
    ).dropna(how="any")
    if joined.empty:
        raise ValueError(
            "no test rows have both complete features and a realized outcome; check that "
            "X_test and y_test share an index and that the target is not all NaN"
        )

    actual = joined["__y__"]
    predicted = model.predict_quantiles(joined.drop(columns="__y__"))

    empirical = (actual.to_numpy()[:, None] <= predicted).mean(axis=0)
    taus = np.asarray(levels, dtype=np.float64)
    n_effective = max(2, len(actual) // horizon)
    standard_error = np.sqrt(taus * (1.0 - taus) / n_effective)

    reference = y_train.dropna() if y_train is not None else actual
    baseline = np.broadcast_to(
        np.quantile(reference.to_numpy(dtype=np.float64), taus), predicted.shape
    )

    return CoverageReport(
        levels=levels,
        empirical=empirical,
        error=empirical - taus,
        standard_error=standard_error,
        n_test=len(actual),
        n_effective=int(n_effective),
        pinball=pinball_loss(actual, predicted, levels),
        baseline_pinball=pinball_loss(actual, baseline, levels),
    )
