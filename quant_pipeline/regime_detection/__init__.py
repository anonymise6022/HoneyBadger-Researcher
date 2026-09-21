"""Regime diagnostics: how much to trust today's signal, not what it is.

Two independent readings of market state, deliberately chosen to fail
differently:

    tda_signal.py  the *geometry* of a rolling multivariate return cloud,
                   via persistent homology and landscape norms
    rough_vol.py   the *roughness* of the volatility path, via the Hurst
                   exponent and the RFSV forecast

Neither produces a trade. Both produce a flag and a number that scale the
conviction of a signal generated elsewhere. The reason is stated at length
in each module, but the short version: the topological signal is validated
on three pre-2010 crashes and demonstrably weaker on every major drawdown
since, and a rolling Hurst estimate has sampling noise comparable to the
moves being read from it. Either is a reasonable input to a confidence
weight. Neither is a reasonable trigger.

They are combined by taking the *more severe* of the two flags, not by
averaging. Averaging would let a calm reading from one cancel an alarm from
the other, and for a risk modulator the asymmetry is the point: a warning
from either is a reason to size down.
"""

from __future__ import annotations

from .rough_vol import (
    HurstEstimate,
    RoughVolConfig,
    RoughVolEstimate,
    estimate_hurst,
    rfsv_forecast,
    rolling_rough_vol,
    rough_vol_estimate,
)
from .tda_signal import (
    TDAConfig,
    TDASignal,
    classify_regime,
    landscape_norms,
    persistence_landscape,
    rips_h1_diagram,
    rolling_tda_signal,
    tda_signal_for_window,
)

__all__ = [
    "REGIME_SEVERITY",
    "HurstEstimate",
    "RoughVolConfig",
    "RoughVolEstimate",
    "TDAConfig",
    "TDASignal",
    "classify_regime",
    "combine_regime_flags",
    "estimate_hurst",
    "landscape_norms",
    "persistence_landscape",
    "rfsv_forecast",
    "rips_h1_diagram",
    "rolling_rough_vol",
    "rolling_tda_signal",
    "rough_vol_estimate",
    "tda_signal_for_window",
]

#: The three regime levels, ordered from calmest to most stressed. The
#: ordering is what makes "take the worse of two flags" well defined.
REGIME_SEVERITY: dict[str, int] = {"stable": 0, "elevated": 1, "unstable": 2}


def combine_regime_flags(*flags: str) -> str:
    """Return the most severe of several regime flags.

    Raises ValueError on an unrecognized flag rather than treating it as
    stable, since a typo that silently reads as "no stress" is exactly the
    failure this layer exists to prevent.
    """
    if not flags:
        raise ValueError("combine_regime_flags needs at least one flag")
    unknown = [f for f in flags if f not in REGIME_SEVERITY]
    if unknown:
        raise ValueError(
            f"unrecognized regime flag(s) {unknown}; expected one of {sorted(REGIME_SEVERITY)}"
        )
    return max(flags, key=lambda flag: REGIME_SEVERITY[flag])
