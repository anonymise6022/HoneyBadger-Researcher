"""Let a person describe a strategy, in words or in a small expression language.

Two front ends onto one representation.

**The expression language** is a deliberately tiny arithmetic-and-comparison
grammar over a handful of indicator functions:

    sma(50) > sma(200)
    rsi(14) < 30
    close > sma(200) and vol(20) < 20

**Plain English** is compiled down to the same expressions by keyword
matching -- the same deterministic approach `query_parser` uses, and for the
same reason: a strategy that compiles differently on two runs is not a
strategy.

**On executing text the user typed.** This runs on the user's own machine at
their own request, so the threat model is mistakes rather than attackers.
That is not a reason to use `eval`. The expression is parsed with `ast` and
then walked against a whitelist of node types; anything not on it -- an
import, an attribute access, a lambda, a comprehension, a call to a name
that is not a known indicator -- is rejected before evaluation with a
message naming the offending construct. `eval` on the raw string would give
a typo the same power as the interpreter, and would make
`__import__('os').system(...)` a working strategy.

**Feedback matters as much as execution.** A rule that never fires, or fires
every single bar, is a far more common mistake than a syntax error, and it
looks like a working strategy. `describe` explains back what was understood,
and `diagnose` reports how often the rule actually triggered over the
history, so a rule that turns out to be always-true says so.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "INDICATORS",
    "CompiledStrategy",
    "StrategyDiagnosis",
    "StrategyError",
    "compile_expression",
    "describe_expression",
    "diagnose",
    "parse_plain_english",
]


class StrategyError(ValueError):
    """Raised when a description cannot be turned into a runnable rule.

    Carries `hint` so the interface can suggest a fix rather than only
    reporting a failure.
    """

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


# --- indicators ------------------------------------------------------------

@dataclass(frozen=True)
class Indicator:
    """One callable available inside an expression.

    `arity` is how many numeric arguments it takes; `warmup` maps those
    arguments to the bars of history it needs, so the engine can skip a
    warm-up period long enough for every indicator in the rule.
    """

    name: str
    arity: int
    warmup: Callable[[tuple[float, ...]], int]
    compute: Callable[[pd.DataFrame, tuple[float, ...]], float]
    summary: str


def _closes(frame: pd.DataFrame) -> np.ndarray:
    return frame["Close"].to_numpy(dtype=float)


def _sma(frame: pd.DataFrame, args: tuple[float, ...]) -> float:
    n = int(args[0])
    values = _closes(frame)[-n:]
    return float(values.mean()) if len(values) else float("nan")


def _ema(frame: pd.DataFrame, args: tuple[float, ...]) -> float:
    n = int(args[0])
    series = frame["Close"].astype(float)
    return float(series.ewm(span=n, adjust=False).mean().iloc[-1])


def _rsi(frame: pd.DataFrame, args: tuple[float, ...]) -> float:
    """Wilder's RSI, matching what a broker's chart shows."""
    n = int(args[0])
    deltas = np.diff(_closes(frame))
    if len(deltas) < n:
        return float("nan")
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    average_gain = float(gains[:n].mean())
    average_loss = float(losses[:n].mean())
    for i in range(n, len(deltas)):
        average_gain = (average_gain * (n - 1) + gains[i]) / n
        average_loss = (average_loss * (n - 1) + losses[i]) / n
    if average_loss <= 1e-12:
        return 100.0 if average_gain > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + average_gain / average_loss))


def _volatility(frame: pd.DataFrame, args: tuple[float, ...]) -> float:
    """Annualized volatility in percent, so `vol(20) < 15` reads naturally."""
    n = int(args[0])
    returns = np.diff(np.log(_closes(frame)))[-n:]
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(252.0) * 100.0)


def _zscore(frame: pd.DataFrame, args: tuple[float, ...]) -> float:
    n = int(args[0])
    values = _closes(frame)[-n:]
    if len(values) < 2:
        return float("nan")
    spread = float(values.std(ddof=1))
    return float((values[-1] - values.mean()) / spread) if spread > 1e-12 else 0.0


def _return_pct(frame: pd.DataFrame, args: tuple[float, ...]) -> float:
    n = int(args[0])
    values = _closes(frame)
    if len(values) <= n or values[-1 - n] <= 0:
        return float("nan")
    return float((values[-1] / values[-1 - n] - 1.0) * 100.0)


def _highest(frame: pd.DataFrame, args: tuple[float, ...]) -> float:
    n = int(args[0])
    return float(frame["High"].to_numpy(dtype=float)[-n:].max())


def _lowest(frame: pd.DataFrame, args: tuple[float, ...]) -> float:
    n = int(args[0])
    return float(frame["Low"].to_numpy(dtype=float)[-n:].min())


#: Every function a strategy may call. Nothing outside this table is
#: reachable from an expression.
INDICATORS: dict[str, Indicator] = {
    "sma": Indicator("sma", 1, lambda a: int(a[0]) + 1, _sma,
                     "the average closing price over the last {0:g} days"),
    "ema": Indicator("ema", 1, lambda a: int(a[0]) * 3, _ema,
                     "a weighted average over about {0:g} days, which reacts faster than a plain average"),
    "rsi": Indicator("rsi", 1, lambda a: int(a[0]) * 3, _rsi,
                     "a 0-100 score of how much of the last {0:g} days' movement was upward"),
    "vol": Indicator("vol", 1, lambda a: int(a[0]) + 2, _volatility,
                     "how much the price has been bouncing around over {0:g} days, as a yearly percentage"),
    "zscore": Indicator("zscore", 1, lambda a: int(a[0]) + 1, _zscore,
                        "how far the price is from its {0:g}-day average, counted in normal-sized moves"),
    "ret": Indicator("ret", 1, lambda a: int(a[0]) + 1, _return_pct,
                     "the percentage change over the last {0:g} days"),
    "highest": Indicator("highest", 1, lambda a: int(a[0]) + 1, _highest,
                         "the highest price reached in the last {0:g} days"),
    "lowest": Indicator("lowest", 1, lambda a: int(a[0]) + 1, _lowest,
                        "the lowest price reached in the last {0:g} days"),
}

#: Bare names usable without a call.
FIELDS: dict[str, Callable[[pd.DataFrame], float]] = {
    "close": lambda f: float(f["Close"].iloc[-1]),
    "open": lambda f: float(f["Open"].iloc[-1]),
    "high": lambda f: float(f["High"].iloc[-1]),
    "low": lambda f: float(f["Low"].iloc[-1]),
    "price": lambda f: float(f["Close"].iloc[-1]),
}

#: AST node types an expression may contain. Everything else is refused.
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
    ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
    ast.Call, ast.Name, ast.Load, ast.Constant,
)


@dataclass
class CompiledStrategy:
    """A parsed rule, ready to run and able to explain itself."""

    source: str
    expression: str
    tree: ast.Expression = field(repr=False)
    warmup_bars: int
    indicators_used: tuple[str, ...]
    name: str = "Custom rule"

    def evaluate(self, frame: pd.DataFrame) -> float:
        """Evaluate the rule against history ending at the latest bar.

        Returns a target exposure: a comparison yields 1.0 or 0.0, and a
        bare arithmetic expression is used as-is so `zscore(20) / -2` can
        express a scaled position.
        """
        value = _evaluate(self.tree.body, frame)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)


def _evaluate(node: ast.AST, frame: pd.DataFrame) -> Any:
    """Walk the whitelisted tree. Never calls eval or compile."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise StrategyError(f"{node.value!r} is not a number")
        return float(node.value)

    if isinstance(node, ast.Name):
        field_fn = FIELDS.get(node.id)
        if field_fn is None:
            raise StrategyError(
                f"'{node.id}' is not something this language knows",
                hint="Try close, or an indicator like sma(50) or rsi(14).",
            )
        return field_fn(frame)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise StrategyError("only plain function names may be called")
        indicator = INDICATORS.get(node.func.id)
        if indicator is None:
            raise StrategyError(
                f"'{node.func.id}' is not an indicator this language knows",
                hint="Available: " + ", ".join(sorted(INDICATORS)),
            )
        if node.keywords:
            raise StrategyError(f"{indicator.name}() does not take named arguments")
        args = tuple(float(_evaluate(a, frame)) for a in node.args)
        if len(args) != indicator.arity:
            raise StrategyError(
                f"{indicator.name}() takes {indicator.arity} number, got {len(args)}"
            )
        return indicator.compute(frame, args)

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, frame)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        return +operand

    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left, frame), _evaluate(node.right, frame)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            # A division by zero in a strategy is a modelling mistake, not a
            # crash: return NaN and let the engine hold its position.
            return left / right if abs(right) > 1e-12 else float("nan")
        raise StrategyError("that arithmetic operator is not supported")

    if isinstance(node, ast.BoolOp):
        values = [_evaluate(v, frame) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, frame)
        for operator, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, frame)
            if not isinstance(left, float) or not isinstance(right, float):
                left, right = float(left), float(right)
            if np.isnan(left) or np.isnan(right):
                return False
            ok = (
                left > right if isinstance(operator, ast.Gt)
                else left >= right if isinstance(operator, ast.GtE)
                else left < right if isinstance(operator, ast.Lt)
                else left <= right if isinstance(operator, ast.LtE)
                else left == right if isinstance(operator, ast.Eq)
                else left != right if isinstance(operator, ast.NotEq)
                else None
            )
            if ok is None:
                raise StrategyError("that comparison is not supported")
            if not ok:
                return False
            left = right
        return True

    raise StrategyError(f"{type(node).__name__} is not allowed in a strategy")


def _validate(tree: ast.Expression) -> tuple[int, tuple[str, ...]]:
    """Check every node against the whitelist; return warm-up and indicators.

    Walking the tree *before* evaluating is what makes this safe: a rejected
    construct never runs, so a malformed or malicious expression cannot have
    side effects on the way to being refused.
    """
    used: list[str] = []
    warmup = 2

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise StrategyError(
                f"{type(node).__name__} is not allowed in a strategy",
                hint=(
                    "Strategies are simple comparisons like sma(50) > sma(200). "
                    "Imports, attributes and assignments are not available."
                ),
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            indicator = INDICATORS.get(node.func.id)
            if indicator is None:
                raise StrategyError(
                    f"'{node.func.id}' is not an indicator this language knows",
                    hint="Available: " + ", ".join(sorted(INDICATORS)),
                )
            constants = [
                a.value for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, (int, float))
            ]
            if len(constants) != len(node.args):
                raise StrategyError(
                    f"{indicator.name}() needs a plain number, like {indicator.name}(20)"
                )
            if any(c <= 0 for c in constants):
                raise StrategyError(f"{indicator.name}() needs a positive number of days")
            if any(c > 2000 for c in constants):
                raise StrategyError(
                    f"{indicator.name}({constants[0]:g}) asks for more history than "
                    "any backtest here has"
                )
            used.append(f"{indicator.name}({', '.join(f'{c:g}' for c in constants)})")
            warmup = max(warmup, indicator.warmup(tuple(float(c) for c in constants)))

    return warmup, tuple(dict.fromkeys(used))


def compile_expression(expression: str, name: str = "Custom rule") -> CompiledStrategy:
    """Parse and validate an expression into a runnable rule.

    Raises StrategyError with a `hint` for anything unusable.
    """
    text = (expression or "").strip()
    if not text:
        raise StrategyError(
            "the rule is empty",
            hint="Try: sma(50) > sma(200)",
        )
    if len(text) > 500:
        raise StrategyError("that rule is longer than this language is meant for")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise StrategyError(
            f"that is not a rule this language can read ({exc.msg})",
            hint="Rules look like: sma(50) > sma(200), or rsi(14) < 30",
        ) from exc

    warmup, used = _validate(tree)
    return CompiledStrategy(
        source=text, expression=text, tree=tree,
        warmup_bars=warmup, indicators_used=used, name=name,
    )


# --- plain English ---------------------------------------------------------

_NUMBER = r"(\d{1,4})"

#: Ordered patterns: the first match wins, so more specific phrasings are
#: listed before the general ones they would otherwise be swallowed by.
_PHRASES: tuple[tuple[str, str], ...] = (
    ((rf"{_NUMBER}\s*(?:-|\s)?day\s+(?:moving\s+)?average\s+(?:is\s+)?(?:cross(?:es|ing)?\s+)?"
     rf"above\s+(?:the\s+)?{_NUMBER}"), "sma({0}) > sma({1})"),
    ((rf"{_NUMBER}\s*(?:-|\s)?day\s+(?:moving\s+)?average\s+(?:is\s+)?(?:cross(?:es|ing)?\s+)?"
     rf"below\s+(?:the\s+)?{_NUMBER}"), "sma({0}) < sma({1})"),
    (rf"rsi\s*\(?\s*{_NUMBER}?\s*\)?\s*(?:is\s+)?(?:below|under|less than)\s*{_NUMBER}",
     "rsi({0}) < {1}"),
    (rf"rsi\s*\(?\s*{_NUMBER}?\s*\)?\s*(?:is\s+)?(?:above|over|greater than)\s*{_NUMBER}",
     "rsi({0}) > {1}"),
    (rf"(?:price|close|it)\s+(?:is\s+)?above\s+(?:the\s+)?{_NUMBER}\s*(?:-|\s)?day",
     "close > sma({0})"),
    (rf"(?:price|close|it)\s+(?:is\s+)?below\s+(?:the\s+)?{_NUMBER}\s*(?:-|\s)?day",
     "close < sma({0})"),
    (rf"volatility\s+(?:is\s+)?(?:below|under|less than)\s*{_NUMBER}", "vol(20) < {0}"),
    (rf"volatility\s+(?:is\s+)?(?:above|over|greater than)\s*{_NUMBER}", "vol(20) > {0}"),
    ((rf"(?:up|risen|gained)\s+(?:more than\s+)?{_NUMBER}\s*%?\s*(?:over|in)\s+"
     rf"(?:the\s+)?(?:last\s+)?{_NUMBER}\s*days?"), "ret({1}) > {0}"),
    ((rf"(?:down|fallen|dropped|lost)\s+(?:more than\s+)?{_NUMBER}\s*%?\s*(?:over|in)\s+"
     rf"(?:the\s+)?(?:last\s+)?{_NUMBER}\s*days?"), "ret({1}) < -{0}"),
    (rf"{_NUMBER}\s+standard deviations?\s+below", "zscore(63) < -{0}"),
    (rf"{_NUMBER}\s+standard deviations?\s+above", "zscore(63) > {0}"),
)

#: Whole-phrase shorthands for the strategies people name rather than describe.
_NAMED: tuple[tuple[str, str, str], ...] = (
    (r"\bgolden cross\b", "sma(50) > sma(200)",
     "the 50-day average above the 200-day, which traders call a golden cross"),
    (r"\bdeath cross\b", "sma(50) < sma(200)",
     "the 50-day average below the 200-day, which traders call a death cross"),
    (r"\bbuy (?:the )?dip\b", "zscore(63) < -1.5",
     "buying when the price is unusually low against its recent average"),
    (r"\boversold\b", "rsi(14) < 30", "buying when RSI says the market is oversold"),
    (r"\boverbought\b", "rsi(14) > 70", "selling when RSI says the market is overbought"),
    (r"\bbuy and hold\b", "1", "simply holding the whole time"),
    (r"\bmomentum\b", "ret(252) > 0", "holding while the past year's return is positive"),
    (r"\btrend follow(?:ing)?\b", "close > sma(200)",
     "holding while the price is above its 200-day average"),
)


def parse_plain_english(text: str) -> tuple[str, list[str]]:
    """Compile a description into an expression. Returns (expression, notes).

    Deterministic keyword matching, not a model: the same sentence must
    always produce the same rule. Every clause it *fails* to understand is
    reported in `notes` rather than dropped, because a strategy that
    silently ignores half of what you asked for is worse than one that
    refuses.
    """
    raw = (text or "").strip()
    if not raw:
        raise StrategyError("no strategy was described", hint="Try: buy when RSI is below 30")

    lowered = raw.lower()
    notes: list[str] = []

    for pattern, expression, explanation in _NAMED:
        if re.search(pattern, lowered):
            notes.append(f"Read as {explanation}.")
            return expression, notes

    clauses: list[str] = []
    consumed = lowered
    for pattern, template in _PHRASES:
        match = re.search(pattern, consumed)
        if not match:
            continue
        groups = [g for g in match.groups() if g is not None]
        # RSI's period is optional in speech ("rsi below 30"); default to 14.
        if template.startswith("rsi(") and len(groups) == 1:
            groups = ["14", groups[0]]
        try:
            clauses.append(template.format(*groups))
        except IndexError:
            continue
        consumed = consumed.replace(match.group(0), " ")

    if not clauses:
        raise StrategyError(
            "none of that could be turned into a rule",
            hint=(
                "Try something like: buy when the 50 day average crosses above the "
                "200 day, or: buy when RSI is below 30. You can also write the rule "
                "directly, such as sma(50) > sma(200)."
            ),
        )

    joiner = " or " if re.search(r"\bor\b", lowered) else " and "
    leftover = re.sub(r"[^a-z]+", " ", consumed).split()
    filler = {
        "buy", "sell", "when", "the", "and", "or", "if", "then", "day", "days",
        "is", "a", "an", "to", "go", "long", "short", "hold", "it", "price",
        "stock", "while", "on", "at", "of", "in", "for", "that", "this", "we",
        "i", "my", "should", "would", "use", "using", "strategy", "rule", "get",
        "out", "into", "with", "than", "over", "under", "above", "below", "exit",
    }
    unread = [w for w in leftover if w not in filler and len(w) > 2]
    if unread:
        distinct = list(dict.fromkeys(unread))[:8]
        notes.append("These words were not used: " + ", ".join(distinct))
    return joiner.join(clauses), notes


def describe_expression(strategy: CompiledStrategy, depth: str = "beginner") -> str:
    """Explain a compiled rule back in words, at the reader's level."""
    from ..explain import Lexicon

    lexicon = Lexicon(depth)  # type: ignore[arg-type]
    if lexicon.is_analyst:
        return (
            f"{strategy.expression}  "
            f"[warm-up {strategy.warmup_bars} bars; "
            f"{', '.join(strategy.indicators_used) or 'no indicators'}]"
        )

    parts: list[str] = []
    for used in strategy.indicators_used:
        name, _, rest = used.partition("(")
        indicator = INDICATORS.get(name)
        if indicator is None:
            continue
        numbers = [float(x) for x in rest.rstrip(")").split(",") if x.strip()]
        parts.append(f"{used} is {indicator.summary.format(*numbers)}")

    lead = (
        "This holds the investment on any day the rule below is true, and sits in "
        "cash otherwise."
        if lexicon.is_beginner
        else "Exposure is 1 when the condition holds and 0 otherwise."
    )
    if not parts:
        return lead
    return lead + " Here, " + "; ".join(parts) + "."


# --- diagnosis -------------------------------------------------------------

@dataclass(frozen=True)
class StrategyDiagnosis:
    """How a rule actually behaves over history, before any P&L is computed.

    This catches the mistakes that a backtest result cannot: a rule that is
    always true is buy-and-hold wearing a costume, and one that never fires
    produces a flat line that looks like a safe strategy rather than a
    broken one.
    """

    bars_tested: int
    times_true: int
    fraction_true: float
    flips: int
    warnings: tuple[str, ...]
    ok: bool

    @property
    def summary(self) -> str:
        if not self.bars_tested:
            return "The rule could not be tested."
        return (
            f"Over {self.bars_tested} days the rule was true on {self.times_true} "
            f"({self.fraction_true:.0%}) and changed its mind {self.flips} times."
        )


def diagnose(strategy: CompiledStrategy, frame: pd.DataFrame, step: int = 1) -> StrategyDiagnosis:
    """Run the rule across history and report how it behaved.

    `step` samples every nth bar. The default walks every bar; the interface
    raises it on long histories, where the shape of the answer is identical
    and the wait is not.
    """
    closes = frame.sort_index()
    start = strategy.warmup_bars
    if len(closes) <= start + 5:
        return StrategyDiagnosis(0, 0, 0.0, 0, ("not enough history to test the rule",), False)

    results: list[bool] = []
    for position in range(start, len(closes), max(1, step)):
        try:
            value = strategy.evaluate(closes.iloc[: position + 1])
        except StrategyError:
            raise
        except Exception:  # noqa: BLE001, S112 - one bad bar must not sink the diagnosis
            continue
        results.append(bool(value) if isinstance(value, bool) else value > 0.5)

    if not results:
        return StrategyDiagnosis(0, 0, 0.0, 0, ("the rule never produced a value",), False)

    times_true = sum(results)
    fraction = times_true / len(results)
    flips = sum(1 for a, b in pairwise(results) if a != b)

    warnings: list[str] = []
    if fraction >= 0.99:
        warnings.append(
            "This rule is true almost every day, so it is effectively just holding "
            "the investment. Compare it against buy-and-hold before reading anything "
            "into the result."
        )
    elif fraction <= 0.01:
        warnings.append(
            "This rule is almost never true, so it will sit in cash and show a flat "
            "line. That is a broken rule, not a safe one."
        )
    if flips > len(results) * 0.4:
        warnings.append(
            f"It changes its mind {flips} times over {len(results)} days. Each change "
            "costs a spread and a commission, which will dominate the result."
        )
    return StrategyDiagnosis(
        bars_tested=len(results),
        times_true=times_true,
        fraction_true=fraction,
        flips=flips,
        warnings=tuple(warnings),
        ok=0.01 < fraction < 0.99,
    )
