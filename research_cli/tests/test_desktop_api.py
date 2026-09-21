"""Tests for the bridge between the window and Python.

Two properties matter here and nothing else does.

**Everything must survive JSON.** pywebview serializes across the bridge, so
a numpy float32, a pydantic model or a bare NaN either fails to encode or
arrives as something JavaScript cannot use. `json.dumps` on every response
is the test.

**Nothing may raise.** An exception crossing the bridge becomes an opaque
failure in a console the user never opens. Every failure path must come back
as a structured `{"ok": false}` payload instead.
"""

from __future__ import annotations

import json

import pytest

from research_cli.desktop.api import DesktopApi


@pytest.fixture(scope="module")
def api() -> DesktopApi:
    return DesktopApi(mock=True)


def _serializable(payload: object) -> str:
    """json.dumps with no fallback encoder: NaN and numpy types must fail."""
    return json.dumps(payload, allow_nan=False)


def test_bootstrap_describes_the_interface(api: DesktopApi) -> None:
    result = api.bootstrap()
    assert result["ok"]
    assert len(result["depths"]) == 3
    assert len(result["strategies"]) >= 6
    assert any(s["alpha"] for s in result["strategies"]), "quant strategies are flagged alpha"
    _serializable(result)


def test_research_returns_a_renderable_view(api: DesktopApi) -> None:
    result = api.research("why did SPY fall today", "beginner")
    assert result["ok"]
    view = result["view"]
    assert view["ticker"] == "SPY"
    assert view["kind"] == "attribution"
    assert view["factors"]
    assert result["validation"]["ok"]
    _serializable(result)


def test_snapshot_returns_fundamentals_and_peers(api: DesktopApi) -> None:
    result = api.research("is AAPL good to invest", "intermediate")
    assert result["ok"]
    assert result["view"]["kind"] == "snapshot"
    assert result["view"]["fundamentals"] is not None
    _serializable(result)


@pytest.mark.parametrize("depth", ["beginner", "intermediate", "analyst"])
def test_every_depth_renders_and_validates(api: DesktopApi, depth: str) -> None:
    result = api.research("why did SPY fall today", depth)
    assert result["ok"]
    assert result["validation"]["ok"], result["validation"]["issues"]
    assert result["view"]["depth"] == depth
    _serializable(result)


def test_depth_changes_the_wording_not_the_figures(api: DesktopApi) -> None:
    """The point of the depth setting: same numbers, different scaffolding."""
    beginner = api.research("why did SPY fall today", "beginner")["view"]
    analyst = api.research("why did SPY fall today", "analyst")["view"]

    assert beginner["observation"]["return_pct"] == analyst["observation"]["return_pct"]
    assert [f["hit_rate"] for f in beginner["factors"]] == [
        f["hit_rate"] for f in analyst["factors"]
    ]
    assert beginner["factors"][0]["reading"] != analyst["factors"][0]["reading"]


def test_a_bad_query_returns_a_payload_not_an_exception(api: DesktopApi) -> None:
    result = api.research("", "beginner")
    assert result["ok"] is False
    assert result["error"]
    _serializable(result)


def test_an_unknown_symbol_is_refused_cleanly(api: DesktopApi) -> None:
    """Offline mode still refuses nonsense rather than inventing a company."""
    live = DesktopApi(mock=False)
    result = live.research("why did ZZZZQQ fall today", "beginner")
    assert result["ok"] is False
    assert "symbol" in result["error"].lower()
    _serializable(result)


def test_price_series_is_downsampled_for_drawing(api: DesktopApi) -> None:
    result = api.price_series("SPY", 365)
    assert result["ok"]
    assert len(result["values"]) == len(result["dates"])
    assert 2 < len(result["values"]) <= 200, "the chart does not need one point per pixel"
    assert all(isinstance(v, float) for v in result["values"])
    _serializable(result)


def test_quant_payload_is_json_safe_and_marked_alpha(api: DesktopApi) -> None:
    result = api.quant("SPY")
    assert result["ok"]
    assert "Experimental" in result["notice"]
    assert result["regime"]["classification"] in {
        "trending", "mean-reverting", "random walk", "unclear",
    }
    _serializable(result)


def test_backtest_returns_curves_of_equal_length(api: DesktopApi) -> None:
    result = api.backtest("SPY", "ma-crossover", years=6.0)
    assert result["ok"]
    assert len(result["equity"]) == len(result["benchmark"]) == len(result["dates"])
    assert result["folds"] >= 1
    _serializable(result)


def test_backtest_rejects_an_unknown_strategy(api: DesktopApi) -> None:
    result = api.backtest("SPY", "not-a-strategy")
    assert result["ok"] is False
    assert "ma-crossover" in result["suggestion"]


def test_quant_strategies_are_reachable_from_the_bridge(api: DesktopApi) -> None:
    result = api.backtest("SPY", "vol-target", years=6.0)
    assert result["ok"]
    assert result["alpha"] is False, "vol targeting is well established, not alpha"


def test_saving_an_unknown_key_is_refused(api: DesktopApi) -> None:
    result = api.save_key("NOT_A_KEY", "x")
    assert result["ok"] is False


def test_nan_never_reaches_the_bridge() -> None:
    """JavaScript's JSON parser rejects NaN outright, so one unguarded numpy
    NaN would break an entire response rather than a single field."""
    from research_cli.desktop.api import _number

    assert _number(float("nan")) is None
    assert _number(float("inf")) is None
    assert _number(None) is None
    assert _number("nonsense") is None
    assert _number(3.5) == 3.5
