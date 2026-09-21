"""Combination layer: difference the two views, modulate, decide.

    divergence_signal.py  moment gaps, premium baseline, regime modulation,
                          conviction and the long/short/no-trade decision

This is the only module that sees both the macro forecast and the market
view, and the only one that emits a decision.
"""

from __future__ import annotations

from .divergence_signal import (
    Decision,
    DivergenceSignal,
    MomentGap,
    PremiumBaseline,
    RegimeModulation,
    divergence_signal,
    moment_gaps,
    regime_modulation,
)

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
