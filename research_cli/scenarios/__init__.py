"""Deterministic "what if" scenarios, measured from history. No model involved."""

from .engine import DEFAULT_SCENARIO_YEARS, ScenarioPack, build_scenarios
from .library import SCENARIOS, ScenarioSpec, scenario_by_id

__all__ = [
    "DEFAULT_SCENARIO_YEARS",
    "SCENARIOS",
    "ScenarioPack",
    "ScenarioSpec",
    "build_scenarios",
    "scenario_by_id",
]
