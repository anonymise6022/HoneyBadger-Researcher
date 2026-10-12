"""Extra language rules for debate prose: stances allowed, certainty banned.

The debate layer needs words the plain report never used -- "the bull case
says", "a value investor would read this as" -- because presenting several
readings of the same evidence is its whole job. Those are *stance* phrases:
they attribute a view to a named perspective, and are allowed.

What stays banned is the same as everywhere else (causes, verdicts), plus
**certainty about the future**: "will rise", "guaranteed", "bound to". A
case can say what history shows and what would prove it wrong; it cannot
say what is going to happen.

`trace_numbers` is the claim-level version of the report validator's
numeric check: a claim may only quote figures that appear in the evidence
items it cites, which is stricter than "somewhere in the bundle".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .validator import CAUSAL_PATTERNS, VERDICT_PATTERNS, _matches, extract_numbers

__all__ = [
    "CERTAINTY_PATTERNS",
    "STANCE_PHRASES",
    "allowed_numbers",
    "banned_phrases",
    "trace_numbers",
]

_FUTURE_VERBS = (
    "go|rise|fall|drop|climb|crash|rally|soar|plunge|recover|double|halve|continue|"
    "keep|beat|outperform|underperform|grow|shrink|rebound|collapse|surge|tank|moon|"
    "increase|decrease|gain|lose"
)

#: Statements of certainty about what a price or company will do.
CERTAINTY_PATTERNS: tuple[tuple[str, str], ...] = (
    (rf"\bwill (?:definitely |certainly |surely |likely |probably )?(?:{_FUTURE_VERBS})\b",
     "will <move>"),
    (rf"\b(?:is|are) going to (?:{_FUTURE_VERBS})\b", "is going to <move>"),
    (r"\bguarantee(?:d|s)?\b", "guaranteed"),
    (r"\bcertain(?:ly)? to\b", "certain to"),
    (r"\bdefinitely\b", "definitely"),
    (r"\b(?:sure|bound) to\b", "sure/bound to"),
    (r"\bno doubt\b", "no doubt"),
    (r"\bcan(?:no|')t lose\b", "can't lose"),
    (r"\brisk[- ]free\b", "risk-free"),
    (r"\bwill (?:make|lose) (?:you )?money\b", "will make/lose money"),
)

#: Phrases that attribute a reading to a perspective. Listed for the
#: prompts and the docs; they need no exemption because none of them
#: collides with a banned pattern.
STANCE_PHRASES = (
    "the bull case says", "the bear case says", "a cautious reader would say",
    "a value investor would read this as", "a momentum reader would read this as",
)

_BANNED = (
    [(p, label, "causal_language") for p, label in CAUSAL_PATTERNS]
    + [(p, label, "verdict_language") for p, label in VERDICT_PATTERNS]
    + [(p, label, "certainty_language") for p, label in CERTAINTY_PATTERNS]
)


def banned_phrases(text: str, include_certainty: bool = True) -> list[tuple[str, str, str]]:
    """Every banned phrase in `text`, as (kind, label, excerpt)."""
    lowered = text.lower()
    found: list[tuple[str, str, str]] = []
    for pattern, label, kind in _BANNED:
        if kind == "certainty_language" and not include_certainty:
            continue
        for match in re.finditer(pattern, lowered):
            start, stop = max(0, match.start() - 30), min(len(text), match.end() + 30)
            found.append((kind, label, " ".join(text[start:stop].split())))
    return found


def allowed_numbers(texts: Iterable[str]) -> set[float]:
    """Numbers a claim may quote: those printed in the evidence it cites.

    The same transformations the report validator allows -- sign dropped,
    percentage <-> fraction -- plus the small structural numbers every
    sentence uses.
    """
    values = {0.0, 1.0, 2.0, 100.0}
    for text in texts:
        for value, _, _ in extract_numbers(text):
            values.update({value, abs(value), value * 100.0, value / 100.0})
    return values


def trace_numbers(text: str, allowed: set[float]) -> list[str]:
    """Figures in `text` that do not match any allowed value, with context."""
    unsupported: list[str] = []
    for value, token, excerpt in extract_numbers(text):
        _, _, fraction = token.partition(".")
        if not _matches(value, allowed, len(fraction)):
            unsupported.append(f"{value:g} ({excerpt})")
    return unsupported
