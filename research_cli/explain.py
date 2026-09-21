"""Explanation depth: the same evidence, pitched at the reader's level.

A report that explains what a standard deviation is will bore an analyst;
one that says "2.6 sigma, 76% conditional hit rate, lift +21pp" will lose a
beginner entirely. Both readers want the same *numbers* -- what differs is
how much scaffolding goes around them.

So depth is a presentation parameter, never a data one. Every level renders
from an identical `EvidenceBundle`, and `synthesis.validator` runs against
all three, so switching level can change the words and never the figures.
That is the point: a beginner and an analyst reading the same report must
not be able to reach different conclusions about what happened.

Three levels:

    beginner      every term glossed in passing; short sentences; the
                  question "is this even unusual" answered before anything
                  else, because that is the thing beginners most often miss
    intermediate  assumes P/E, volatility and a base rate need no
                  introduction; spends its words on interpretation instead
    analyst       terse and quantitative; shows standard errors, sample
                  sizes and effect sizes inline; no reassurance, no glosses

`Lexicon` is the mechanism. Rather than maintaining three copies of every
sentence -- which drift apart the moment anyone edits one -- a writer asks
the lexicon whether a term needs explaining and gets back either a gloss or
an empty string. One sentence, three renderings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["DEPTH_LABELS", "Depth", "Lexicon", "describe_depth"]

Depth = Literal["beginner", "intermediate", "analyst"]

DEPTH_LABELS: dict[Depth, str] = {
    "beginner": "Beginner -- every term explained",
    "intermediate": "Intermediate -- assumes the basics",
    "analyst": "Analyst -- terse and quantitative",
}

#: Ordering, for comparisons like "at least intermediate".
_RANK: dict[Depth, int] = {"beginner": 0, "intermediate": 1, "analyst": 2}

#: term -> (beginner gloss, intermediate gloss). The analyst level gets
#: neither. Glosses are written to sit inside a sentence in parentheses, so
#: they are lower-case and do not end in a full stop.
_GLOSSARY: dict[str, tuple[str, str]] = {
    "standard_deviation": (
        "a measure of how big a normal day's move is for this investment",
        "its typical daily move",
    ),
    "base_rate": (
        ("how often this happens on any random day, which is what you would "
        "guess knowing nothing at all"),
        "the unconditional frequency",
    ),
    "hit_rate": (
        ("the share of past days like this one where the price moved the same "
        "way it did today"),
        "the conditional frequency",
    ),
    "lift": (
        "how much better than guessing this factor actually does",
        "hit rate minus base rate",
    ),
    "volatility": (
        "how much the price bounces around day to day",
        "annualized standard deviation of returns",
    ),
    "pe_ratio": (
        "what you pay for each dollar of the company's yearly profit",
        "price divided by trailing earnings",
    ),
    "profit_margin": (
        "how much of each dollar of sales is left over as profit",
        "net income over revenue",
    ),
    "dividend_yield": (
        "the cash the company pays you each year, as a share of the price",
        "annual dividend over price",
    ),
    "debt_to_equity": (
        "how much the company has borrowed compared with what the owners put in",
        "total debt over shareholders' equity",
    ),
    "drawdown": (
        ("the worst fall from a previous high point -- how bad it got before "
        "it recovered"),
        "peak-to-trough decline",
    ),
    "sharpe": (
        ("return compared with how bumpy the ride was; higher is better for "
        "the same return"),
        "excess return per unit of volatility",
    ),
    "walk_forward": (
        ("the settings were chosen using only older data, then tested on newer "
        "data the method had never seen"),
        "parameters fitted in-sample, scored out-of-sample",
    ),
    "hurst": (
        ("a number describing how jagged the volatility path is; below 0.5 "
        "means rougher than a random walk"),
        "the self-similarity exponent of log-volatility",
    ),
    "mean_reversion": (
        ("the tendency of a price to drift back toward its recent average "
        "after moving away from it"),
        "negative autocorrelation in the price level",
    ),
    "half_life": (
        ("roughly how long it has historically taken to close half the gap "
        "back to the average"),
        "the decay constant of the fitted OU process",
    ),
}


@dataclass(frozen=True)
class Lexicon:
    """Renders terminology at one depth.

    Held by the writers rather than consulted globally so that a single
    render is internally consistent -- a report cannot explain a base rate
    in one paragraph and assume it in the next.
    """

    depth: Depth = "beginner"

    def __post_init__(self) -> None:
        if self.depth not in _RANK:
            raise ValueError(
                f"depth must be one of {sorted(_RANK)}, got {self.depth!r}"
            )

    def at_least(self, level: Depth) -> bool:
        """Whether this depth is `level` or more advanced."""
        return _RANK[self.depth] >= _RANK[level]

    @property
    def is_beginner(self) -> bool:
        return self.depth == "beginner"

    @property
    def is_analyst(self) -> bool:
        return self.depth == "analyst"

    def gloss(self, term: str, parenthesized: bool = True) -> str:
        """An inline explanation of `term`, or "" when none is warranted.

        Unknown terms return "" rather than raising: a missing gloss should
        degrade to a slightly terser sentence, never to a crashed report.
        """
        entry = _GLOSSARY.get(term)
        if entry is None or self.is_analyst:
            return ""
        text = entry[0] if self.is_beginner else entry[1]
        return f" ({text})" if parenthesized else text

    def pick(self, beginner: str, intermediate: str, analyst: str) -> str:
        """Choose between three phrasings of the same fact."""
        return {
            "beginner": beginner,
            "intermediate": intermediate,
            "analyst": analyst,
        }[self.depth]

    def number(self, value: float, unit: str = "", decimals: int | None = None) -> str:
        """Format a figure at a precision suited to the reader.

        Beginners get fewer decimals: "2.6 standard deviations" is as
        actionable as "2.65" and easier to hold in your head. An analyst
        gets the extra digit because they may be comparing two of them.
        """
        if decimals is None:
            decimals = 2 if self.is_analyst else 1
        return f"{value:.{decimals}f}{unit}"


def describe_depth(depth: Depth) -> str:
    """One line describing a depth, for menus and help text."""
    return DEPTH_LABELS[depth]
