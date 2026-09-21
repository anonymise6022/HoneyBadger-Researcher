"""The bridge between the window and the Python that does the work.

Every method here is callable from JavaScript through pywebview. Three rules
shape the design:

**Return plain JSON-safe data, never objects.** pywebview serializes across
the bridge, and a pydantic model or a numpy float32 either fails to
serialize or arrives as something JavaScript cannot use. Everything is
converted to dict/list/str/float/bool/None at this boundary.

**Never raise across the bridge.** An exception becomes an opaque failure in
the browser console that the user never sees. Every method returns
`{"ok": false, "error": ..., "suggestion": ...}` instead, so the interface
can show the same friendly message the CLI would.

**Do no analysis here.** This layer marshals arguments in and results out.
The reports, hit rates and statistics come from exactly the same functions
the CLI calls -- including the validator, which still runs before any text
reaches the window. A second rendering path that could disagree with the
terminal would defeat the purpose of the evidence bundle.
"""

from __future__ import annotations

import math
import traceback
from datetime import date, timedelta
from typing import Any

from ..backtest.engine import BacktestConfig
from ..backtest.options import (
    OPTION_TEMPLATES,
    OptionBacktestConfig,
    run_option_backtest,
)
from ..backtest.strategies import STRATEGY_TEMPLATES
from ..backtest.strategy_lang import (
    INDICATORS,
    StrategyError,
    compile_expression,
    describe_expression,
    diagnose,
    parse_plain_english,
)
from ..backtest.surface import SURFACE_AXES, sweep
from ..backtest.walk_forward import WalkForwardConfig, walk_forward
from ..data.price_ingest import (
    PriceDataError,
    Source,
    UnknownSymbolError,
    fetch_history,
)
from ..data.symbol_search import search_symbols
from ..evidence.snapshot import build_snapshot_bundle
from ..evidence.why_moved import build_why_moved_bundle
from ..explain import DEPTH_LABELS, Lexicon
from ..query_parser import QueryParseError, parse_query
from ..settings import KNOWN_KEYS, describe_credentials, set_key
from ..synthesis.snapshot_report import SNAPSHOT_SECTIONS, render_snapshot_report
from ..synthesis.template_report import REQUIRED_SECTIONS, render_report
from ..synthesis.validator import validate_report

__all__ = ["DesktopApi"]

_MAX_SPARK_POINTS = 180


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _fail(error: str, suggestion: str = "") -> dict[str, Any]:
    return {"ok": False, "error": error, "suggestion": suggestion}


def _number(value: Any) -> float | None:
    """Coerce to a JSON-safe float, mapping NaN and infinity to None.

    JavaScript's JSON parser rejects NaN and Infinity outright, so a single
    unguarded numpy NaN anywhere in a payload breaks the entire response
    rather than one field.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class DesktopApi:
    """Methods exposed to the window's JavaScript."""

    def __init__(self, mock: bool = False) -> None:
        self.mock = mock

    @property
    def _source(self) -> Source:
        return "mock" if self.mock else "auto"

    # --- metadata ---------------------------------------------------------

    def bootstrap(self) -> dict[str, Any]:
        """Everything the interface needs before the user does anything."""
        return _ok(
            depths=[{"id": key, "label": label} for key, label in DEPTH_LABELS.items()],
            strategies=[
                {
                    "id": name,
                    "description": meta["description"],
                    "alpha": bool(meta.get("alpha", False)),
                    "has_surface": name in SURFACE_AXES,
                }
                for name, meta in STRATEGY_TEMPLATES.items()
            ],
            option_strategies=[
                {
                    "id": key,
                    "name": template.name,
                    "description": template.description,
                    "caveat": template.caveat,
                }
                for key, template in OPTION_TEMPLATES.items()
            ],
            indicators=[
                {"name": name, "summary": ind.summary}
                for name, ind in sorted(INDICATORS.items())
            ],
            credentials=[
                {
                    "name": status.name,
                    "present": status.present,
                    "source": status.source,
                    "hint": status.hint,
                    "unlocks": status.unlocks,
                }
                for status in describe_credentials()
            ],
            known_keys=sorted(KNOWN_KEYS),
            mock=self.mock,
        )

    def save_key(self, name: str, value: str) -> dict[str, Any]:
        """Store an API key from the settings panel."""
        try:
            if name not in KNOWN_KEYS:
                return _fail(f"{name} is not a key this tool uses.")
            set_key(name, value)
        except (ValueError, OSError) as exc:
            return _fail(f"Could not save that key: {exc}")
        return _ok()

    # --- research ---------------------------------------------------------

    def research(self, query: str, depth: str = "beginner") -> dict[str, Any]:
        """Answer a question. Returns a bundle shaped for display."""
        try:
            parsed = parse_query(query)
        except QueryParseError as exc:
            return _fail(str(exc), exc.suggestion)

        try:
            if parsed.question_type == "snapshot":
                bundle = build_snapshot_bundle(parsed, source=self._source)
                report = render_snapshot_report(bundle)
                sections = SNAPSHOT_SECTIONS
            else:
                bundle = build_why_moved_bundle(parsed, source=self._source)
                report = render_report(bundle, depth)
                sections = REQUIRED_SECTIONS
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))
        except Exception as exc:  # noqa: BLE001 - never surface a traceback
            traceback.print_exc()
            return _fail(
                f"Something went wrong ({type(exc).__name__}: {exc})",
                "Try again, or switch to offline mode to check the tool itself.",
            )

        validation = validate_report(report, bundle, required_sections=sections)
        return _ok(
            view=self._bundle_payload(bundle, Lexicon(depth)),  # type: ignore[arg-type]
            validation={
                "ok": validation.ok,
                "checked": validation.numbers_checked,
                "matched": validation.numbers_matched,
                "issues": [str(issue) for issue in validation.issues[:6]],
            },
        )

    def _bundle_payload(self, bundle: Any, lexicon: Lexicon) -> dict[str, Any]:
        """Flatten an EvidenceBundle into JSON the interface can render."""
        observation = bundle.observation
        payload: dict[str, Any] = {
            "kind": bundle.question_type,
            "ticker": bundle.ticker,
            "period": bundle.period_description,
            "query": bundle.query,
            "synthetic": bundle.is_synthetic,
            "counterevidence": [
                {"label": item.label, "detail": item.detail}
                for item in bundle.counterevidence
            ],
            "limitations": list(bundle.limitations),
            "warnings": list(bundle.warnings),
            "sources": list(bundle.data_sources),
            "depth": lexicon.depth,
        }

        if observation is not None:
            payload["observation"] = {
                "summary": observation.plain_summary,
                "return_pct": _number(observation.period_return_pct),
                "sigma": _number(observation.sigma_multiple),
                "daily_vol_pct": _number(observation.daily_vol_pct),
                "label": observation.outlier_label,
                "unusual": bool(observation.is_unusual),
                "close": _number(observation.close),
                "direction": observation.direction,
                "gloss": lexicon.gloss("standard_deviation", parenthesized=False),
            }

        if bundle.question_type == "attribution":
            payload["factors"] = [
                self._factor_payload(factor, lexicon)
                for factor in bundle.external_factors()
            ]
            payload["mechanical"] = [
                self._factor_payload(factor, lexicon)
                for factor in bundle.mechanical_factors()
            ]
            payload["macro"] = [
                {"label": release.label, "description": release.description,
                 "surprise_z": _number(release.surprise_z)}
                for release in bundle.macro_releases
            ]
        else:
            fundamentals = bundle.fundamentals
            payload["fundamentals"] = (
                {
                    "name": fundamentals.company_name,
                    "sector": fundamentals.sector,
                    "market_cap": _number(fundamentals.market_cap),
                    "profitable": fundamentals.is_profitable,
                    "metrics": [
                        {"key": "trailing_pe", "label": "Price to earnings",
                         "value": _number(fundamentals.trailing_pe), "format": "ratio",
                         "gloss": lexicon.gloss("pe_ratio", parenthesized=False)},
                        {"key": "revenue_growth", "label": "Revenue growth",
                         "value": _number(fundamentals.revenue_growth), "format": "rate",
                         "gloss": ""},
                        {"key": "profit_margin", "label": "Profit margin",
                         "value": _number(fundamentals.profit_margin), "format": "rate",
                         "gloss": lexicon.gloss("profit_margin", parenthesized=False)},
                        {"key": "return_on_equity", "label": "Return on equity",
                         "value": _number(fundamentals.return_on_equity), "format": "rate",
                         "gloss": ""},
                        {"key": "debt_to_equity", "label": "Debt to equity",
                         "value": _number(fundamentals.debt_to_equity), "format": "ratio",
                         "gloss": lexicon.gloss("debt_to_equity", parenthesized=False)},
                        {"key": "dividend_yield", "label": "Dividend yield",
                         "value": _number(fundamentals.dividend_yield), "format": "rate",
                         "gloss": lexicon.gloss("dividend_yield", parenthesized=False)},
                    ],
                }
                if fundamentals is not None
                else None
            )
            payload["peers"] = [
                {
                    "label": peer.label,
                    "metric": peer.metric,
                    "subject": _number(peer.subject_value),
                    "median": _number(peer.peer_median),
                    "vs_median_pct": _number(peer.vs_median_pct),
                    "is_rate": peer.metric in {
                        "profit_margin", "revenue_growth", "dividend_yield",
                        "return_on_equity",
                    },
                }
                for peer in bundle.peers
            ]
            trend = bundle.trend
            payload["trend"] = (
                {
                    "summary": trend.summary,
                    "price": _number(trend.current_price),
                    "return_1m": _number(trend.return_1m_pct),
                    "return_3m": _number(trend.return_3m_pct),
                    "return_1y": _number(trend.return_1y_pct),
                    "vol": _number(trend.annualized_vol_pct),
                    "high": _number(trend.high_52w),
                    "low": _number(trend.low_52w),
                }
                if trend is not None
                else None
            )
            payload["news"] = [
                {
                    "title": item.title,
                    "publisher": item.publisher,
                    "published": item.published.isoformat() if item.published else None,
                }
                for item in bundle.news
            ]
        return payload

    def _factor_payload(self, factor: Any, lexicon: Lexicon) -> dict[str, Any]:
        return {
            "id": factor.factor_id,
            "label": factor.label,
            "what_happened": factor.what_happened,
            "observed": _number(factor.observed_value),
            "units": factor.observed_units,
            "hit_rate": _number(factor.historical_hit_rate),
            "base_rate": _number(factor.base_rate),
            "lift": _number(factor.lift),
            "sample": factor.sample_size,
            "confidence": factor.confidence,
            "beats": bool(factor.beats_base_rate),
            "condition": factor.condition_description,
            "note": factor.note,
            "reading": lexicon.pick(
                "better than just guessing" if factor.beats_base_rate
                else "no better than guessing",
                "clears the base rate" if factor.beats_base_rate
                else "inside the margin of error",
                f"lift {factor.lift:+.1%}" if factor.lift is not None else "n/a",
            ),
        }

    # --- price series, for charts ----------------------------------------

    def price_series(self, ticker: str, days: int = 365) -> dict[str, Any]:
        """Daily closes for the chart, downsampled to a drawable length."""
        end = date.today()  # noqa: DTZ011
        try:
            frame, source, _ = fetch_history(
                ticker.upper(), end - timedelta(days=int(days * 1.5)), end, self._source
            )
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))

        closes = frame["Close"].tail(days)
        step = max(1, len(closes) // _MAX_SPARK_POINTS)
        sampled = closes.iloc[::step]
        return _ok(
            ticker=ticker.upper(),
            dates=[d.strftime("%Y-%m-%d") for d in sampled.index],
            values=[_number(v) for v in sampled.to_numpy()],
            synthetic=source == "mock",
        )

    # --- quant (alpha) ----------------------------------------------------

    def quant(self, ticker: str) -> dict[str, Any]:
        """Run the experimental quant diagnostics."""
        from ..quant import ALPHA_NOTICE, analyse

        end = date.today()  # noqa: DTZ011
        try:
            frame, source, _ = fetch_history(
                ticker.upper(), end - timedelta(days=1500), end, self._source
            )
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))

        result = analyse(frame, ticker.upper())
        volatility, regime, reversion = result.volatility, result.regime, result.mean_reversion
        return _ok(
            notice=ALPHA_NOTICE,
            ticker=ticker.upper(),
            synthetic=source == "mock",
            failures=list(result.failures),
            volatility=(
                {
                    "close_to_close": _number(volatility.close_to_close_pct),
                    "yang_zhang": _number(volatility.yang_zhang_pct),
                    "ewma": _number(volatility.ewma_pct),
                    "forecast": _number(volatility.forecast_pct),
                    "horizon": volatility.forecast_horizon_days,
                    "model": volatility.forecast_model,
                    "long_run": _number(volatility.long_run_pct),
                    "persistence": _number(volatility.persistence),
                    "regime": volatility.regime,
                    "notes": list(volatility.notes),
                }
                if volatility
                else None
            ),
            regime=(
                {
                    "classification": regime.classification,
                    "hurst": _number(regime.hurst),
                    "hurst_error": _number(regime.hurst_standard_error),
                    "variance_ratio": _number(regime.variance_ratio),
                    "variance_ratio_z": _number(regime.variance_ratio_z),
                    "describe": regime.describe(),
                }
                if regime
                else None
            ),
            mean_reversion=(
                {
                    "z_score": _number(reversion.z_score),
                    "half_life": _number(reversion.half_life_days),
                    "reverting": bool(reversion.is_mean_reverting),
                    "describe": reversion.describe(),
                }
                if reversion
                else None
            ),
        )

    # --- backtest ---------------------------------------------------------

    def backtest(self, ticker: str, strategy: str, years: float = 12.0) -> dict[str, Any]:
        """Walk-forward a strategy and return curves plus statistics."""
        if strategy not in STRATEGY_TEMPLATES:
            return _fail(
                f"{strategy} is not a strategy this tool knows.",
                "Available: " + ", ".join(sorted(STRATEGY_TEMPLATES)),
            )

        end = date.today()  # noqa: DTZ011
        start = end - timedelta(days=int(years * 365.25) + 40)
        try:
            frame, source, _ = fetch_history(ticker.upper(), start, end, self._source)
            outcome = walk_forward(
                frame, strategy, WalkForwardConfig(), BacktestConfig()
            )
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))
        except ValueError as exc:
            return _fail(str(exc), "Try more years of history.")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return _fail(f"The backtest failed ({type(exc).__name__}: {exc})")

        equity, benchmark = outcome.equity, outcome.benchmark_equity
        step = max(1, len(equity) // 400)
        return _ok(
            ticker=ticker.upper(),
            strategy=strategy,
            synthetic=source == "mock",
            alpha=bool(STRATEGY_TEMPLATES[strategy].get("alpha", False)),
            dates=[d.strftime("%Y-%m-%d") for d in equity.index[::step]],
            equity=[_number(v) for v in equity.to_numpy()[::step]],
            benchmark=[_number(v) for v in benchmark.to_numpy()[::step]],
            stats=dict(outcome.stats.summary_rows()),
            benchmark_stats=dict(outcome.benchmark_stats.summary_rows()),
            excess_pct=_number(outcome.excess_return_pct),
            folds=len(outcome.folds),
            folds_won=outcome.folds_beating_benchmark,
            stability=outcome.parameter_stability,
            costs=_number(outcome.total_costs),
        )

    # --- explore ----------------------------------------------------------

    def search(self, query: str, limit: int = 8) -> dict[str, Any]:
        """Find symbols by company name or ticker."""
        try:
            hits = search_symbols(query, limit=limit, offline=self.mock)
        except ValueError as exc:
            return _fail(str(exc))
        return _ok(
            results=[
                {"symbol": h.symbol, "name": h.name, "kind": h.kind, "exchange": h.exchange}
                for h in hits
            ]
        )

    def price_bars(self, ticker: str, days: int = 365) -> dict[str, Any]:
        """OHLC bars for the candlestick chart, downsampled to a drawable length."""
        end = date.today()  # noqa: DTZ011
        try:
            frame, source, _ = fetch_history(
                ticker.upper(), end - timedelta(days=int(days * 1.5) + 10), end, self._source
            )
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))

        bars = frame.tail(days)
        # Roughly 180 candles is as many as fit legibly at this width.
        step = max(1, len(bars) // 180)
        bars = bars.iloc[::step]
        closes = bars["Close"]
        first, last = float(closes.iloc[0]), float(closes.iloc[-1])
        return _ok(
            ticker=ticker.upper(),
            synthetic=source == "mock",
            bars=[
                {
                    "d": index.strftime("%Y-%m-%d"),
                    "o": _number(row.Open), "h": _number(row.High),
                    "l": _number(row.Low), "c": _number(row.Close),
                }
                for index, row in bars.iterrows()
            ],
            change_pct=_number((last / first - 1.0) * 100.0) if first else None,
            last=_number(last),
            high=_number(float(bars["High"].max())),
            low=_number(float(bars["Low"].min())),
        )

    # --- strategy building ------------------------------------------------

    def interpret_strategy(self, text: str, depth: str = "beginner") -> dict[str, Any]:
        """Turn a description into a rule and explain it back, without running it.

        Separate from `custom_backtest` so the interface can give feedback as
        the user types: understanding what was written is a different failure
        from the strategy performing badly, and they deserve different
        messages.
        """
        raw = (text or "").strip()
        if not raw:
            return _fail("Describe a strategy first.", "Try: buy when RSI is below 30")

        notes: list[str] = []
        expression = raw
        # Anything with a comparison or an indicator call is already an
        # expression; otherwise it is prose to be compiled.
        looks_like_code = any(token in raw for token in ("<", ">", "(", "==")) and not raw.endswith("?")
        try:
            if not looks_like_code:
                expression, notes = parse_plain_english(raw)
            compiled = compile_expression(expression)
        except StrategyError as exc:
            return _fail(str(exc), exc.hint)

        return _ok(
            expression=compiled.expression,
            explanation=describe_expression(compiled, depth),
            indicators=list(compiled.indicators_used),
            warmup=compiled.warmup_bars,
            notes=notes,
            was_prose=not looks_like_code,
        )

    def custom_backtest(
        self, text: str, ticker: str, years: float = 12.0, depth: str = "beginner"
    ) -> dict[str, Any]:
        """Compile a described strategy, check how it behaves, then test it."""
        interpreted = self.interpret_strategy(text, depth)
        if not interpreted["ok"]:
            return interpreted

        try:
            compiled = compile_expression(interpreted["expression"])
        except StrategyError as exc:
            return _fail(str(exc), exc.hint)

        end = date.today()  # noqa: DTZ011
        start = end - timedelta(days=int(years * 365.25) + 40)
        try:
            frame, source, _ = fetch_history(ticker.upper(), start, end, self._source)
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))

        # Sample the diagnosis on long histories: the shape of the answer is
        # the same and the wait is not.
        try:
            behaviour = diagnose(compiled, frame, step=max(1, len(frame) // 900))
        except StrategyError as exc:
            return _fail(str(exc), exc.hint)

        from ..backtest.engine import run_backtest

        class _Compiled:
            """Adapter onto the engine's strategy protocol."""

            name = "Your rule"

            def __init__(self, rule):
                self.rule = rule
                self.warmup_bars = rule.warmup_bars

            def signal(self, window):
                return self.rule.evaluate(window.frame)

        try:
            outcome = run_backtest(frame, _Compiled(compiled), BacktestConfig())
        except ValueError as exc:
            return _fail(
                str(exc),
                "Your rule needs more history than this. Try more years, or shorter windows.",
            )

        payload = self._backtest_payload(outcome, ticker.upper(), source == "mock")
        payload.update(
            expression=compiled.expression,
            explanation=interpreted["explanation"],
            notes=interpreted["notes"],
            behaviour={
                "summary": behaviour.summary,
                "fraction_true": _number(behaviour.fraction_true),
                "flips": behaviour.flips,
                "warnings": list(behaviour.warnings),
                "ok": behaviour.ok,
            },
        )
        return _ok(**payload)

    def _backtest_payload(self, outcome: Any, ticker: str, synthetic: bool) -> dict[str, Any]:
        """Shared shape for every kind of backtest, so the UI has one path."""
        equity, benchmark = outcome.equity, outcome.benchmark_equity
        step = max(1, len(equity) // 400)
        return {
            "ticker": ticker,
            "synthetic": synthetic,
            "dates": [d.strftime("%Y-%m-%d") for d in equity.index[::step]],
            "equity": [_number(v) for v in equity.to_numpy()[::step]],
            "benchmark": [_number(v) for v in benchmark.to_numpy()[::step]],
            "stats": dict(outcome.stats.summary_rows()),
            "benchmark_stats": dict(outcome.benchmark_stats.summary_rows()),
            "excess_pct": _number(outcome.excess_return_pct),
            "costs": _number(outcome.total_costs),
            "trades": outcome.trades,
            "strategy_notes": list(getattr(outcome, "notes", []))[:6],
        }

    # --- options ----------------------------------------------------------

    def option_backtest(
        self, ticker: str, strategy: str, years: float = 10.0, moneyness: float | None = None
    ) -> dict[str, Any]:
        """Paper-trade an option overlay on a real underlying."""
        if strategy not in OPTION_TEMPLATES:
            return _fail(
                f"{strategy} is not an option strategy this tool knows.",
                "Available: " + ", ".join(sorted(OPTION_TEMPLATES)),
            )
        end = date.today()  # noqa: DTZ011
        start = end - timedelta(days=int(years * 365.25) + 60)
        try:
            frame, source, _ = fetch_history(ticker.upper(), start, end, self._source)
            outcome = run_option_backtest(
                frame, strategy, OptionBacktestConfig(moneyness=moneyness)
            )
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))
        except ValueError as exc:
            return _fail(str(exc), "Try a longer history.")

        template = OPTION_TEMPLATES[strategy]
        payload = self._backtest_payload(outcome, ticker.upper(), source == "mock")

        # A payoff diagram at expiry, which is the clearest way to show what
        # an option strategy actually does.
        spot = float(frame["Close"].iloc[-1])
        strike = spot * (moneyness if moneyness is not None else (template.moneyness_override or 1.0))
        payload["payoff"] = self._payoff_curve(template, spot, strike)
        payload["spot"] = _number(spot)
        payload["strike"] = _number(strike)
        payload["name"] = template.name
        payload["caveat"] = template.caveat
        return _ok(**payload)

    def _payoff_curve(self, template: Any, spot: float, strike: float) -> list[dict[str, float]]:
        """Profit at expiry across a range of underlying prices."""
        points: list[dict[str, float]] = []
        for step in range(41):
            price = spot * (0.7 + 0.6 * step / 40)
            value = template.share_exposure * (price - spot)
            for kind, sign in template.leg_kinds:
                intrinsic = (
                    max(price - strike, 0.0) if kind == "call" else max(strike - price, 0.0)
                )
                value += sign * intrinsic
            points.append({"x": _number(price), "y": _number(value)})
        return points

    # --- parameter surface ------------------------------------------------

    def parameter_surface(self, ticker: str, strategy: str, years: float = 10.0) -> dict[str, Any]:
        """Sweep two parameters and return a grid for the 3D surface."""
        if strategy not in SURFACE_AXES:
            return _fail(
                f"No parameter surface is defined for {strategy}.",
                "Available: " + ", ".join(sorted(SURFACE_AXES)),
            )
        end = date.today()  # noqa: DTZ011
        start = end - timedelta(days=int(years * 365.25) + 40)
        try:
            frame, _, _ = fetch_history(ticker.upper(), start, end, self._source)
            result = sweep(frame, strategy)
        except (UnknownSymbolError, PriceDataError) as exc:
            return _fail(str(exc), getattr(exc, "suggestion", ""))
        except ValueError as exc:
            return _fail(str(exc))

        best_cell = None
        if result.best is not None:
            best_cell = {
                "col": result.x_values.index(result.best[0]),
                "row": result.y_values.index(result.best[1]),
                "x": result.best[0],
                "y": result.best[1],
            }
        return _ok(
            strategy=strategy,
            x_name=result.x_name,
            y_name=result.y_name,
            x_values=result.x_values,
            y_values=result.y_values,
            sharpe=result.sharpe,
            total_return=result.total_return,
            best=best_cell,
            best_sharpe=_number(result.best_sharpe),
            ruggedness=_number(result.ruggedness),
            verdict=result.verdict,
            notes=list(result.notes),
        )
