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
import time
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
from ..options.chain import (
    RISK_FREE_RATE,
    Chain,
    ChainError,
    ContractQuote,
    fetch_chain,
    list_expiries,
)
from ..options.greeks import (
    CONTRACT_SIZE,
    black_scholes_price,
    intrinsic_value,
)
from ..options.greeks import greeks as bs_greeks
from ..options.paper import (
    Account,
    OrderError,
    load_account,
    mark_price,
    save_account,
)
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
        #: Chains, keyed by symbol and expiry, with the moment they were
        #: fetched. See the options desk section for why this exists.
        self._chains: dict[tuple[str, str], tuple[float, Chain]] = {}
        #: The paper account, read from disk on first use and written back
        #: after anything that changes it.
        self._paper: Account | None = None

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

    # --- options desk -----------------------------------------------------
    #
    # A live chain, a paper account and the greeks. Three things are worth
    # knowing about this section.
    #
    # Quotes are cached for a few seconds. Yahoo's chain endpoint is slow and
    # rate limited, one screen refresh can want the same chain three times
    # over, and the data is a quarter of an hour delayed in any case -- so a
    # cache this short costs nothing in accuracy and is the difference
    # between a desk that responds and one that stalls.
    #
    # Orders are filled against the same quote the screen is showing, at the
    # ask to buy and the bid to sell. Filling at the midpoint would be the
    # single most flattering thing this code could do.
    #
    # Expiries settle before anything else is reported, against the
    # underlying's close on the contract's own expiry day. An account that
    # quietly keeps a position that expired last week is not a paper account,
    # it is a fiction.

    _CHAIN_CACHE_SECONDS = 8.0

    def _chain(self, ticker: str, expiry: str) -> Chain:
        """A chain, from a short-lived cache when one is warm."""
        key = (ticker.upper(), str(expiry))
        cached = self._chains.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < self._CHAIN_CACHE_SECONDS:
            return cached[1]
        chain = fetch_chain(ticker, expiry, self._source)
        self._chains[key] = (now, chain)
        return chain

    def _settlement_price(self, symbol: str, when: date) -> float | None:
        """The underlying's close on an expiry day, for settling against."""
        try:
            frame, _, _ = fetch_history(
                symbol, when - timedelta(days=10), when + timedelta(days=2), self._source
            )
        except (UnknownSymbolError, PriceDataError):
            return None
        on_or_before = frame.loc[: str(when)]
        if not len(on_or_before):
            return None
        return float(on_or_before["Close"].iloc[-1])

    @staticmethod
    def _quote_payload(row: ContractQuote, spot: float, years: float) -> dict[str, Any]:
        """One side of one strike, priced and differentiated."""
        volatility, origin = row.implied_vol(spot, years)
        sensitivities = row.greeks(spot, years)
        payload: dict[str, Any] = {
            "kind": row.kind,
            "strike": _number(row.strike),
            "bid": _number(row.bid),
            "ask": _number(row.ask),
            "last": _number(row.last),
            "mid": _number(row.mid),
            "spread_pct": _number(row.spread_pct),
            "volume": row.volume,
            "open_interest": row.open_interest,
            "iv": _number(volatility),
            "iv_source": origin,
            "tradeable": row.tradeable,
            "in_the_money": (spot > row.strike) if row.kind == "call" else (spot < row.strike),
        }
        if sensitivities is not None:
            payload.update({
                "delta": _number(sensitivities.delta),
                "gamma": _number(sensitivities.gamma),
                "theta": _number(sensitivities.theta),
                "vega": _number(sensitivities.vega),
                "rho": _number(sensitivities.rho),
            })
        return payload

    def option_expiries(self, ticker: str) -> dict[str, Any]:
        """Every expiry the market lists for this underlying."""
        try:
            expiries, source, warnings = list_expiries(ticker, self._source)
        except ChainError as exc:
            return _fail(str(exc), exc.suggestion)
        today = date.today()  # noqa: DTZ011
        return _ok(
            ticker=ticker.upper(),
            synthetic=source == "mock",
            warnings=list(warnings),
            expiries=[
                {
                    "date": value.isoformat(),
                    "label": value.strftime("%d %b %Y"),
                    "days": (value - today).days,
                }
                for value in expiries
            ],
        )

    def option_chain(self, ticker: str, expiry: str) -> dict[str, Any]:
        """The whole chain at one expiry, laid out strike by strike.

        Calls and puts share a row because that is how a chain is read: the
        question is almost always "what is this strike worth both ways",
        never "list me the calls".
        """
        try:
            chain = self._chain(ticker, expiry)
        except ChainError as exc:
            return _fail(str(exc), exc.suggestion)

        years = chain.years_to_expiry()
        by_strike: dict[float, dict[str, Any]] = {}
        for row in chain.rows:
            entry = by_strike.setdefault(row.strike, {"strike": _number(row.strike)})
            entry[row.kind] = self._quote_payload(row, chain.spot, years)

        strikes = sorted(by_strike)
        nearest = min(strikes, key=lambda value: abs(value - chain.spot)) if strikes else None
        return _ok(
            ticker=chain.symbol,
            expiry=chain.expiry.isoformat(),
            spot=_number(chain.spot),
            synthetic=chain.synthetic,
            source=chain.source,
            quoted_at=chain.quoted_at.isoformat(),
            days_to_expiry=chain.days_to_expiry(),
            years_to_expiry=_number(years),
            atm_strike=_number(nearest),
            warnings=list(chain.warnings),
            rows=[by_strike[strike] for strike in strikes],
        )

    def option_contract(self, ticker: str, expiry: str, strike: float, kind: str) -> dict[str, Any]:
        """One contract in full: quote, greeks, greek curves and its history.

        The two pictures here answer the two questions a chain cannot. The
        curves say how this contract behaves if the underlying moves *now* --
        which is what delta and gamma mean, drawn rather than tabulated. The
        history says how its price has tracked the underlying's, which is
        where the leverage in an option becomes visible: a 3% move in the
        stock is a 30% move in the option, and no single number conveys that
        as well as the two lines side by side.
        """
        try:
            chain = self._chain(ticker, expiry)
        except ChainError as exc:
            return _fail(str(exc), exc.suggestion)
        row = chain.find(kind, float(strike))
        if row is None:
            return _fail(
                f"No {kind} is listed at {strike} for {chain.expiry.isoformat()}.",
                "Pick a strike from the chain.",
            )

        spot = chain.spot
        years = chain.years_to_expiry()
        volatility, iv_origin = row.implied_vol(spot, years)
        quote = self._quote_payload(row, spot, years)

        payload: dict[str, Any] = {
            "ticker": chain.symbol,
            "expiry": chain.expiry.isoformat(),
            "days_to_expiry": chain.days_to_expiry(),
            "spot": _number(spot),
            "synthetic": chain.synthetic,
            "quoted_at": chain.quoted_at.isoformat(),
            "contract": quote,
            "label": f"{chain.symbol} {chain.expiry.strftime('%d %b %y')} {strike:g} {kind[0].upper()}",
            "cost_per_contract": _number((row.ask or 0.0) * CONTRACT_SIZE),
            "credit_per_contract": _number((row.bid or 0.0) * CONTRACT_SIZE),
            "warnings": list(chain.warnings),
        }

        reference = row.mid or row.ask or row.last
        if reference:
            payload["break_even"] = _number(
                strike + reference if kind == "call" else strike - reference
            )
        if volatility is None:
            payload["curves"] = None
            payload["history"] = None
            payload["note"] = (
                "This contract's price does not pin down an implied volatility, so its "
                "greeks cannot be drawn. That happens deep in the money near expiry, "
                "where the contract is simply worth what it would settle for."
            )
            return _ok(**payload)

        payload["iv"] = _number(volatility)
        payload["iv_source"] = iv_origin
        if reference and reference > 0:
            sensitivities = bs_greeks(spot, strike, years, RISK_FREE_RATE, volatility, kind)
            # Elasticity: the percentage move in the option for a one percent
            # move in the underlying. The honest word for "leverage".
            payload["leverage"] = _number(abs(sensitivities.delta) * spot / reference)

        payload["curves"] = self._greek_curves(spot, float(strike), years, volatility, kind)
        payload["history"] = self._option_history(
            chain.symbol, float(strike), chain.expiry, volatility, kind
        )
        return _ok(**payload)

    @staticmethod
    def _greek_curves(
        spot: float, strike: float, years: float, volatility: float, kind: str
    ) -> dict[str, Any]:
        """Every greek across a range of underlying prices, plus the payoff.

        Drawn against the underlying rather than against time, because that
        is the axis a trader is exposed to. The band is a third either way,
        which covers any move worth planning for at these tenors.
        """
        points = 61
        prices = [spot * (0.67 + 0.66 * step / (points - 1)) for step in range(points)]
        curves: dict[str, list[float | None]] = {
            "spots": [], "price": [], "expiry": [],
            "delta": [], "gamma": [], "theta": [], "vega": [], "rho": [],
        }
        for price in prices:
            sensitivities = bs_greeks(price, strike, years, RISK_FREE_RATE, volatility, kind)
            curves["spots"].append(_number(price))
            curves["price"].append(_number(sensitivities.price))
            curves["expiry"].append(_number(intrinsic_value(price, strike, kind)))
            curves["delta"].append(_number(sensitivities.delta))
            curves["gamma"].append(_number(sensitivities.gamma))
            curves["theta"].append(_number(sensitivities.theta))
            curves["vega"].append(_number(sensitivities.vega))
            curves["rho"].append(_number(sensitivities.rho))
        return curves

    def _option_history(
        self, symbol: str, strike: float, expiry: date, volatility: float, kind: str
    ) -> dict[str, Any] | None:
        """The underlying's last six months beside what this contract would
        have been worth on each of those days.

        Modelled, and labelled as modelled: there is no free source of
        historical option quotes, so the price on each past day is Black-
        Scholes at today's implied volatility with that day's close and that
        day's time to expiry. The level is therefore a guess; the *shape* --
        how much harder the option moves than the stock -- is the point, and
        that is governed by delta, which the model gets approximately right.
        """
        end = date.today()  # noqa: DTZ011
        try:
            frame, _, _ = fetch_history(symbol, end - timedelta(days=200), end, self._source)
        except (UnknownSymbolError, PriceDataError):
            return None
        frame = frame.tail(126)
        if len(frame) < 10:
            return None

        dates: list[str] = []
        underlying: list[float | None] = []
        option: list[float | None] = []
        for stamp, row in frame.iterrows():
            close = float(row.Close)
            remaining = max((expiry - stamp.date()).days / 365.0, 1.0 / 365.0)
            dates.append(stamp.strftime("%Y-%m-%d"))
            underlying.append(_number(close))
            option.append(
                _number(black_scholes_price(close, strike, remaining, RISK_FREE_RATE, volatility, kind))
            )

        first_underlying = next((v for v in underlying if v), None)
        first_option = next((v for v in option if v), None)
        if not first_underlying or not first_option:
            return None
        return {
            "dates": dates,
            "underlying": underlying,
            "option": option,
            # Indexed to 100 so two series that differ by two orders of
            # magnitude can share an axis. Reading them against each other is
            # the whole purpose; reading either one's level is not.
            "underlying_indexed": [_number(v / first_underlying * 100) if v else None for v in underlying],
            "option_indexed": [_number(v / first_option * 100) if v else None for v in option],
            "modelled": True,
        }

    # --- the paper account ------------------------------------------------

    def _account(self) -> Account:
        if self._paper is None:
            self._paper = load_account()
        return self._paper

    def _save(self) -> None:
        if self._paper is not None:
            save_account(self._paper)

    def option_account(self) -> dict[str, Any]:
        """Cash, open positions marked to the market, and the blotter."""
        account = self._account()
        events = account.settle_expired(self._settlement_price)
        if events:
            self._save()

        marks: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        exposure = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        stale: list[str] = []

        for position in account.positions:
            quote = None
            spot = None
            try:
                chain = self._chain(position.symbol, position.expiry)
                spot = chain.spot
                quote = chain.find(position.kind, position.strike)
                years = chain.years_to_expiry()
            except ChainError:
                years = None
            if quote is None or spot is None:
                stale.append(position.label())

            if spot is not None:
                intrinsic = intrinsic_value(spot, position.strike, position.kind)
                mark = mark_price(
                    position,
                    quote.bid if quote else None,
                    quote.ask if quote else None,
                    intrinsic,
                )
            else:
                # Nothing can be priced: no quote, and no underlying either.
                # Marking at zero would book a total loss the market never
                # delivered, so the line is held at what it cost until a
                # price comes back.
                intrinsic = 0.0
                mark = position.average_price
            marks[position.id] = mark
            sensitivities = quote.greeks(spot, years) if quote and years else None
            if sensitivities is not None:
                shares = position.contracts * CONTRACT_SIZE
                exposure["delta"] += sensitivities.delta * shares
                exposure["gamma"] += sensitivities.gamma * shares
                exposure["theta"] += sensitivities.theta * shares
                exposure["vega"] += sensitivities.vega * shares

            rows.append({
                "id": position.id,
                "label": position.label(),
                "symbol": position.symbol,
                "expiry": position.expiry,
                "strike": _number(position.strike),
                "kind": position.kind,
                "contracts": position.contracts,
                "average_price": _number(position.average_price),
                "mark": _number(mark),
                "value": _number(position.value_at(mark)),
                "cost_basis": _number(position.cost_basis),
                "unrealized": _number(position.unrealized(mark)),
                "collateral": _number(position.collateral),
                "days_to_expiry": (position.expiry_date - date.today()).days,  # noqa: DTZ011
                "underlying": _number(spot),
                "bid": _number(quote.bid) if quote else None,
                "ask": _number(quote.ask) if quote else None,
                "delta": _number(sensitivities.delta) if sensitivities else None,
                "theta": _number(sensitivities.theta) if sensitivities else None,
                "assignment_risk": bool(
                    position.contracts < 0 and spot and intrinsic > 0
                ),
            })

        unrealized = sum(row["unrealized"] or 0.0 for row in rows)
        equity = account.equity(marks)
        return _ok(
            cash=_number(account.cash),
            starting_cash=_number(account.starting_cash),
            buying_power=_number(account.buying_power),
            collateral_held=_number(account.collateral_held),
            equity=_number(equity),
            realized=_number(account.realized),
            unrealized=_number(unrealized),
            total_return_pct=_number((equity / account.starting_cash - 1.0) * 100.0),
            positions=rows,
            exposure={key: _number(value) for key, value in exposure.items()},
            expiries_settled=events,
            stale=stale,
            ledger=list(reversed(account.ledger[-40:])),
        )

    def option_order(
        self, ticker: str, expiry: str, strike: float, kind: str, side: str, contracts: int
    ) -> dict[str, Any]:
        """Buy or sell contracts at the price the book is actually showing."""
        try:
            chain = self._chain(ticker, expiry)
        except ChainError as exc:
            return _fail(str(exc), exc.suggestion)
        row = chain.find(kind, float(strike))
        if row is None:
            return _fail(f"No {kind} is listed at {strike}.", "Pick a strike from the chain.")

        try:
            size = int(contracts)
        except (TypeError, ValueError):
            return _fail("Order size must be a whole number of contracts.")
        fill = row.fill_price(side) if side in ("buy", "sell") else None
        if side not in ("buy", "sell"):
            return _fail(f"Unknown order side {side!r}.")

        account = self._account()
        try:
            position, ticket = account.open_position(
                chain.symbol, chain.expiry.isoformat(), float(strike), kind,
                side, size, fill, chain.spot,
            )
        except OrderError as exc:
            return _fail(str(exc), exc.suggestion)
        self._save()

        return _ok(
            filled={
                "label": position.label(),
                "side": side,
                "contracts": size,
                "price": _number(ticket.fill_price),
                "premium": _number(ticket.premium),
                "commission": _number(ticket.commission),
                "collateral": _number(ticket.collateral),
                "cash_effect": _number(ticket.cash_effect),
                "note": ticket.note,
            },
            account=self.option_account(),
        )

    def option_preview(
        self, ticker: str, expiry: str, strike: float, kind: str, side: str, contracts: int
    ) -> dict[str, Any]:
        """What an order would cost, before anyone commits to it."""
        try:
            chain = self._chain(ticker, expiry)
        except ChainError as exc:
            return _fail(str(exc), exc.suggestion)
        row = chain.find(kind, float(strike))
        if row is None:
            return _fail(f"No {kind} is listed at {strike}.")
        if side not in ("buy", "sell"):
            return _fail(f"Unknown order side {side!r}.")

        account = self._account()
        try:
            ticket = account.quote_order(
                side, int(contracts), row.fill_price(side), float(strike), kind, chain.spot
            )
        except OrderError as exc:
            return _fail(str(exc), exc.suggestion)

        needed = -ticket.buying_power_effect
        return _ok(
            side=side,
            contracts=int(contracts),
            fill_price=_number(ticket.fill_price),
            premium=_number(ticket.premium),
            commission=_number(ticket.commission),
            collateral=_number(ticket.collateral),
            cash_effect=_number(ticket.cash_effect),
            buying_power_effect=_number(ticket.buying_power_effect),
            buying_power=_number(account.buying_power),
            affordable=bool(needed <= account.buying_power + 1e-9),
            note=ticket.note,
        )

    def option_close(self, position_id: str, contracts: int | None = None) -> dict[str, Any]:
        """Close a line at the price the other side of the book is showing."""
        account = self._account()
        position = account.find(position_id)
        if position is None:
            return _fail("That position is not open any more.")

        try:
            chain = self._chain(position.symbol, position.expiry)
            row = chain.find(position.kind, position.strike)
        except ChainError as exc:
            return _fail(str(exc), exc.suggestion)
        if row is None:
            return _fail(
                "That contract is no longer quoted.",
                "It may have been delisted; it will settle at expiry.",
            )

        # Closing a long sells into the bid; closing a short buys the offer.
        side = "sell" if position.contracts > 0 else "buy"
        try:
            outcome = account.close_position(position_id, row.fill_price(side), contracts)
        except OrderError as exc:
            return _fail(str(exc), exc.suggestion)
        self._save()
        return _ok(
            closed={
                "label": position.label(),
                "contracts": outcome["contracts"],
                "realized": _number(outcome["realized"]),
                "commission": _number(outcome["commission"]),
            },
            account=self.option_account(),
        )

    def option_reset(self) -> dict[str, Any]:
        """Start the paper account again from the opening balance."""
        self._account().reset()
        self._save()
        return _ok(account=self.option_account())

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
