"""Topological regime detection: persistent homology of rolling point clouds.

The idea, from Gidea & Katz (2018), is that the geometry of a multivariate
return cloud changes shape before a crash, not just scale. Take a sliding
window of daily log-returns on several indices as a cloud of points in R^d,
build a Vietoris-Rips filtration on it, and track the one-dimensional
homology -- loops. In calm markets the cloud is a diffuse blob with few
persistent loops. As correlation and volatility rise together, the cloud
collapses toward a lower-dimensional structure that oscillates, and the
persistent loops get longer-lived. Summarizing those loops by the norm of
their persistence landscape (Bubenik 2015) gives a scalar per day, and
Gidea & Katz report that this scalar rises sharply in the run-up to the
1987, 2000 and 2008 crashes.

**What this module actually computes.**

1. *Vietoris-Rips filtration.* For a threshold r, connect every pair of
   points within r and fill in every simplex all of whose edges are
   present. As r grows from 0, loops are born and die.
2. *H1 persistence.* Loops are found exactly, by reducing the boundary
   matrix from triangles to edges over GF(2) (Edelsbrunner, Letscher &
   Zomorodian 2002). H0 pairs are read off a union-find first, which is
   both faster and tells the H1 stage which edges can be pivots.
3. *Persistence landscape.* Each bar (b, d) becomes a tent function
   Lambda(t) = max(0, min(t - b, d - t)); lambda_k(t) is the k-th largest
   tent at t. The L1 and L2 norms of the landscape are the output. Unlike
   the bottleneck or Wasserstein distance between diagrams, a landscape
   norm is a vector-space quantity -- it can be z-scored and averaged,
   which is what makes it usable as a feature at all.

**The limitation, stated plainly.** The published result rests on three
pre-2010 crashes -- three events, one estimator, chosen after the fact.
Attempts to carry it to the drawdowns since then do much worse:

* *February and Q4 2018.* The VIX-complex unwind was a two-day
  microstructure event. A 50-day window barely registers it before it
  happens; the landscape norm rises largely *after* the move.
* *COVID, February-March 2020.* An exogenous shock with no multi-week
  build-up in the return geometry. The signal fires late, when the
  correlation spike is already in realized volatility -- which a plain
  realized-correlation feature gives for free, and sooner.
* *2022.* A slow, rates-driven drawdown with no single crash date. The
  landscape norm drifts up with volatility and produces no distinct
  warning, and its z-score spends much of the year above the "elevated"
  threshold without anything discrete happening.

The common thread: the signal responds to *coordinated oscillation building
over weeks*, which was the shape of the pre-2008 regime and is not the
shape of most modern drawdowns. That is the reason `ensemble` uses this
strictly as a conviction modulator. On its own it is a slow, noisy
volatility-and-correlation proxy with more machinery than the job needs;
its value is that it is not a monotone function of realized volatility, so
it can disagree with the volatility-based diagnostics in a way that is
informative about *how* the two disagree.

References: Gidea & Katz (2018), "Topological data analysis of financial
time series: Landscapes of crashes", Physica A 491, 820-834; Bubenik
(2015), JMLR 16, 77-102; Edelsbrunner, Letscher & Zomorodian (2002),
Discrete & Computational Geometry 28, 511-533.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform

__all__ = [
    "TDAConfig",
    "TDASignal",
    "landscape_norms",
    "persistence_landscape",
    "rips_h1_diagram",
    "rolling_tda_signal",
]

_LANDSCAPE_LAYERS = 5  # k in lambda_k; deeper layers carry negligible norm
_LANDSCAPE_GRID = 200


@dataclass(frozen=True)
class TDAConfig:
    """Window and threshold settings for the topological signal.

    Attributes
    ----------
    window : points per cloud. 50 follows Gidea & Katz. Larger windows
        smooth the signal but cost O(w^3) triangles in the filtration.
    zscore_window : trailing windows used to standardize the landscape norm.
        The raw norm has no natural scale -- it is in units of return
        distance -- so only its deviation from its own recent history is
        interpretable.
    elevated_z, unstable_z : z-score cuts for the three-level regime flag.
    max_points : cap on cloud size. A window longer than this is subsampled
        on an even grid, because triangle count grows as the cube.
    landscape_layers : how many landscape layers lambda_1..lambda_k to sum
        over in the norm.
    """

    window: int = 50
    zscore_window: int = 60
    elevated_z: float = 1.0
    unstable_z: float = 2.0
    max_points: int = 60
    landscape_layers: int = _LANDSCAPE_LAYERS

    def __post_init__(self) -> None:
        if self.window < 5:
            raise ValueError(f"window must be at least 5 points to contain a loop, got {self.window}")
        if self.zscore_window < 5:
            raise ValueError(f"zscore_window must be at least 5, got {self.zscore_window}")
        if not self.elevated_z < self.unstable_z:
            raise ValueError(
                f"elevated_z ({self.elevated_z}) must be below unstable_z ({self.unstable_z})"
            )
        if self.max_points < 5:
            raise ValueError(f"max_points must be at least 5, got {self.max_points}")
        if self.landscape_layers < 1:
            raise ValueError(f"landscape_layers must be at least 1, got {self.landscape_layers}")


@dataclass(frozen=True)
class TDASignal:
    """One day's topological summary.

    Attributes
    ----------
    l1_norm, l2_norm : persistence-landscape norms of the H1 diagram.
    n_loops : bars in the H1 diagram -- how many independent loops ever
        appeared, at any scale.
    max_persistence : longest bar (death - birth). The single most
        persistent loop, which is what a bottleneck-distance method would
        key on.
    enclosing_radius : the filtration value past which the Rips complex is
        a cone and no H1 class can survive. Reported because every bar is
        bounded by it, so it sets the scale the norms live on.
    """

    l1_norm: float
    l2_norm: float
    n_loops: int
    max_persistence: float
    enclosing_radius: float


def _union_find_h0(
    edges: list[tuple[float, int, int]], n_points: int
) -> NDArray[np.bool_]:
    """Classify edges as positive (loop-creating) or negative (merging).

    Processing edges in filtration order, an edge joining two distinct
    components destroys an H0 class and is *negative*; an edge inside a
    component closes a cycle and is *positive*. Only positive edges can be
    the pivot of a reduced triangle column, so this halves the work in the
    H1 reduction and is the standard first pass.

    Returns a boolean mask over `edges`, True where the edge is positive.
    """
    parent = list(range(n_points))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    positive = np.zeros(len(edges), dtype=bool)
    for index, (_, i, j) in enumerate(edges):
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            positive[index] = True
        else:
            parent[root_i] = root_j
    return positive


def rips_h1_diagram(
    points: ArrayLike, max_radius: float | None = None
) -> tuple[NDArray[np.float64], float]:
    """Exact one-dimensional persistence diagram of a Vietoris-Rips filtration.

    Parameters
    ----------
    points : (n, d) point cloud. n must be at least 3 for a loop to exist.
    max_radius : truncate the filtration here. Defaults to the *enclosing
        radius*, min_i max_j d(i, j): above it the complex is a cone over
        the minimizing point and therefore contractible, so no H1 class can
        survive past it. Truncating there is exact for H1, not an
        approximation, and it typically removes most of the triangles.

    Returns
    -------
    (diagram, enclosing_radius) where diagram is an (m, 2) array of
    (birth, death) pairs, sorted by persistence descending. Bars still alive
    at `max_radius` are recorded as dying there -- with the default radius
    there are none, since H1 is empty above it.

    Notes
    -----
    The reduction is the standard GF(2) column algorithm: process triangle
    columns in filtration order, and while a column's lowest non-zero row
    is already some other column's pivot, add that column to it. Columns are
    Python sets of edge indices and addition is symmetric difference, which
    is exactly GF(2) arithmetic on a sparse column.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2:
        raise ValueError(f"points must be a 2-D (n, d) array, got shape {cloud.shape}")
    n = cloud.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 points for a 1-cycle to be possible, got {n}")
    if not np.all(np.isfinite(cloud)):
        raise ValueError("point cloud contains NaN or infinite coordinates")

    distances = squareform(pdist(cloud))
    enclosing_radius = float(np.min(np.max(distances, axis=1)))
    cutoff = enclosing_radius if max_radius is None else float(max_radius)
    if cutoff <= 0.0:
        # All points coincide: no scale, no topology.
        return np.empty((0, 2), dtype=np.float64), enclosing_radius

    edges = [
        (float(distances[i, j]), i, j)
        for i, j in combinations(range(n), 2)
        if distances[i, j] <= cutoff
    ]
    if len(edges) < 3:
        return np.empty((0, 2), dtype=np.float64), enclosing_radius
    edges.sort()
    edge_index = {(i, j): index for index, (_, i, j) in enumerate(edges)}
    edge_value = np.array([value for value, _, _ in edges], dtype=np.float64)
    positive = _union_find_h0(edges, n)

    # A triangle enters the filtration when its longest edge does.
    triangles: list[tuple[float, int, int, int]] = []
    for i, j, k in combinations(range(n), 3):
        longest = max(distances[i, j], distances[i, k], distances[j, k])
        if longest <= cutoff:
            triangles.append((float(longest), i, j, k))
    if not triangles:
        return np.empty((0, 2), dtype=np.float64), enclosing_radius
    # Sort by filtration value, breaking ties by the triangle's vertices so
    # the order is deterministic across runs and platforms.
    triangles.sort()

    pivots: dict[int, set[int]] = {}
    paired_birth: dict[int, float] = {}
    bars: list[tuple[float, float]] = []

    for value, i, j, k in triangles:
        column = {edge_index[(i, j)], edge_index[(i, k)], edge_index[(j, k)]}
        while column:
            low = max(column)
            if low not in pivots:
                break
            column ^= pivots[low]
        if not column:
            continue  # the triangle closes an already-bounded cycle
        low = max(column)
        pivots[low] = column
        paired_birth[low] = float(edge_value[low])
        if value > edge_value[low]:  # zero-length bars carry no information
            bars.append((float(edge_value[low]), value))

    # Positive edges never used as a pivot bound an essential class at this
    # cutoff. With the enclosing radius there are none, but an explicit
    # max_radius can leave some; they are recorded as dying at the cutoff.
    for index in np.flatnonzero(positive):
        if index not in pivots and edge_value[index] < cutoff:
            bars.append((float(edge_value[index]), cutoff))

    if not bars:
        return np.empty((0, 2), dtype=np.float64), enclosing_radius
    diagram = np.array(bars, dtype=np.float64)
    return diagram[np.argsort(-(diagram[:, 1] - diagram[:, 0]))], enclosing_radius


def persistence_landscape(
    diagram: ArrayLike, layers: int = _LANDSCAPE_LAYERS, grid_points: int = _LANDSCAPE_GRID
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the persistence landscape lambda_1..lambda_k on a grid.

    Each bar (b, d) contributes the tent Lambda(t) = max(0, min(t-b, d-t)),
    which peaks at height (d-b)/2 at the bar's midpoint. lambda_k(t) is the
    k-th largest tent value at t, so lambda_1 traces the most persistent
    feature present at each scale and later layers pick up the rest.

    Returns (grid, landscape) with landscape of shape (layers, grid_points).
    An empty diagram gives an all-zero landscape on a unit grid, which is
    the correct answer -- no loops means zero norm -- and keeps the caller
    free of a special case.
    """
    bars = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    if layers < 1:
        raise ValueError(f"layers must be at least 1, got {layers}")
    if grid_points < 2:
        raise ValueError(f"grid_points must be at least 2, got {grid_points}")
    if bars.size == 0:
        return np.linspace(0.0, 1.0, grid_points), np.zeros((layers, grid_points))
    if np.any(bars[:, 1] < bars[:, 0]):
        raise ValueError("diagram contains a bar that dies before it is born")

    grid = np.linspace(float(bars[:, 0].min()), float(bars[:, 1].max()), grid_points)
    births, deaths = bars[:, 0][:, None], bars[:, 1][:, None]
    tents = np.maximum(0.0, np.minimum(grid - births, deaths - grid))  # (n_bars, grid)

    # Partial sort would suffice, but the diagrams here have tens of bars.
    ordered = np.sort(tents, axis=0)[::-1]
    landscape = np.zeros((layers, grid_points), dtype=np.float64)
    take = min(layers, ordered.shape[0])
    landscape[:take] = ordered[:take]
    return grid, landscape


def landscape_norms(
    diagram: ArrayLike, layers: int = _LANDSCAPE_LAYERS, grid_points: int = _LANDSCAPE_GRID
) -> tuple[float, float]:
    """L1 and L2 norms of the persistence landscape.

        ||lambda||_1 = sum_k integral |lambda_k(t)| dt
        ||lambda||_2 = sqrt( sum_k integral lambda_k(t)^2 dt )

    Integrals are trapezoidal on the landscape grid. L1 weights many
    moderate loops like a few long ones; L2 concentrates on the longest,
    so the two disagree exactly when the diagram's shape changes rather
    than its scale. Both are reported for that reason.
    """
    grid, landscape = persistence_landscape(diagram, layers, grid_points)
    if landscape.sum() == 0.0:
        return 0.0, 0.0
    l1 = float(np.sum(np.trapezoid(np.abs(landscape), grid, axis=1)))
    l2 = float(np.sqrt(np.sum(np.trapezoid(landscape**2, grid, axis=1))))
    return l1, l2


def tda_signal_for_window(points: ArrayLike, config: TDAConfig | None = None) -> TDASignal:
    """Compute the topological summary of a single point cloud.

    Clouds larger than `config.max_points` are subsampled on an even index
    grid rather than at random, so the result is deterministic and the
    subsample still spans the whole window.
    """
    cfg = config or TDAConfig()
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.shape[0] > cfg.max_points:
        keep = np.linspace(0, cloud.shape[0] - 1, cfg.max_points).round().astype(int)
        cloud = cloud[np.unique(keep)]

    diagram, enclosing_radius = rips_h1_diagram(cloud)
    l1, l2 = landscape_norms(diagram, cfg.landscape_layers)
    persistence = diagram[:, 1] - diagram[:, 0] if diagram.size else np.array([0.0])
    return TDASignal(
        l1_norm=l1,
        l2_norm=l2,
        n_loops=int(diagram.shape[0]),
        max_persistence=float(persistence.max()),
        enclosing_radius=enclosing_radius,
    )


def classify_regime(zscore: float, config: TDAConfig | None = None) -> str:
    """Map a landscape-norm z-score to "stable" / "elevated" / "unstable".

    NaN -- which is what a not-yet-full z-score window gives -- maps to
    "stable" rather than propagating, because the honest reading of "no
    baseline yet" for a *modulator* is "do not modulate". The caller can
    distinguish the two cases from the z-score column itself.
    """
    cfg = config or TDAConfig()
    if not np.isfinite(zscore):
        return "stable"
    if zscore >= cfg.unstable_z:
        return "unstable"
    if zscore >= cfg.elevated_z:
        return "elevated"
    return "stable"


def rolling_tda_signal(
    returns: pd.DataFrame, config: TDAConfig | None = None
) -> pd.DataFrame:
    """Compute the landscape norms, z-scores and flags over a return panel.

    Parameters
    ----------
    returns : (T, d) DataFrame of aligned daily returns, one column per
        series. Gidea & Katz use four equity indices; any d >= 2 works, but
        with d = 1 the cloud is collinear and H1 is trivially empty.
    config : windows and thresholds.

    Returns
    -------
    DataFrame indexed like `returns` with columns `tda_l1`, `tda_l2`,
    `tda_n_loops`, `tda_max_persistence`, `tda_z` (rolling z-score of the
    L1 norm) and `regime_flag`. The first `window - 1` rows have no cloud
    and are NaN; the next `zscore_window - 1` have norms but no baseline to
    standardize against, so `tda_z` stays NaN while `tda_l1` is populated.

    This is the expensive part of the pipeline: each window reduces a
    boundary matrix with O(w^3) triangles. At the default 50-point window
    that is tens of thousands of columns per day.
    """
    cfg = config or TDAConfig()
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a DataFrame of aligned series, got {type(returns).__name__}")
    if returns.shape[1] < 2:
        raise ValueError(
            f"need at least 2 series for a non-degenerate point cloud; a 1-column panel "
            f"gives a collinear cloud with no loops. Got {returns.shape[1]}."
        )
    if len(returns) < cfg.window:
        raise ValueError(
            f"need at least window={cfg.window} rows, got {len(returns)}"
        )

    values = returns.to_numpy(dtype=np.float64)
    records: list[dict[str, float]] = []
    for end in range(len(returns)):
        start = end - cfg.window + 1
        if start < 0 or not np.all(np.isfinite(values[start : end + 1])):
            records.append(
                {"tda_l1": np.nan, "tda_l2": np.nan, "tda_n_loops": np.nan, "tda_max_persistence": np.nan}
            )
            continue
        signal = tda_signal_for_window(values[start : end + 1], cfg)
        records.append(
            {
                "tda_l1": signal.l1_norm,
                "tda_l2": signal.l2_norm,
                "tda_n_loops": float(signal.n_loops),
                "tda_max_persistence": signal.max_persistence,
            }
        )

    frame = pd.DataFrame(records, index=returns.index)
    rolling = frame["tda_l1"].rolling(cfg.zscore_window, min_periods=max(5, cfg.zscore_window // 2))
    std = rolling.std()
    frame["tda_z"] = (frame["tda_l1"] - rolling.mean()) / std.where(std > 1e-15)
    frame["regime_flag"] = [classify_regime(z, cfg) for z in frame["tda_z"]]
    return frame
