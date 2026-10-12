"""Read the three answers a plan needs: how much, how long, how much to lose.

Forgiving about format ("$1,000", "1k", "6 months", "20%") and strict about
meaning: a horizon shorter than a month or a loss limit of nothing is
refused with a sentence that says why, not silently adjusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["PlanInputError", "PlanRequest", "parse_amount", "parse_horizon", "parse_max_loss"]

_TRADING_DAYS = {"day": 1, "week": 5, "month": 21, "year": 252}
MIN_HORIZON_DAYS = 21
MAX_HORIZON_DAYS = 1260


class PlanInputError(ValueError):
    """An answer that cannot become part of a plan, with a reason to show."""


@dataclass(frozen=True)
class PlanRequest:
    symbol: str
    display_symbol: str
    amount: float
    horizon_days: int
    horizon_label: str
    max_loss: float

    @property
    def max_loss_pct(self) -> float:
        return self.max_loss / self.amount * 100.0


def _dollars(text: str) -> float:
    cleaned = text.strip().lower().replace(",", "").replace("$", "").replace("usd", "").strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k|thousand|m|million)?", cleaned)
    if not match:
        raise PlanInputError(f"{text!r} is not an amount -- try something like $1000")
    return float(match.group(1)) * {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(
        match.group(2) or "", 1.0)


def parse_amount(text: str) -> float:
    """'$1,000', '1000', '2.5k' -> dollars."""
    value = _dollars(text)
    if value < 50:
        raise PlanInputError("the amount needs to be at least $50 for fees not to dominate")
    return value


def parse_horizon(text: str) -> tuple[int, str]:
    """'6 months', '1 year', '3y', '90 days' -> (trading days, label)."""
    cleaned = text.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(d|days?|w|weeks?|m|mo|months?|y|yr|years?)", cleaned)
    if not match:
        raise PlanInputError(f"{text!r} is not a length of time -- try '6 months' or '1 year'")
    count, unit = float(match.group(1)), match.group(2)
    key = ("day" if unit.startswith("d") else "week" if unit.startswith("w")
           else "year" if unit.startswith("y") else "month")
    days = int(round(count * _TRADING_DAYS[key]))
    if days < MIN_HORIZON_DAYS:
        raise PlanInputError("a plan needs a horizon of at least a month; shorter is trading, not investing")
    if days > MAX_HORIZON_DAYS:
        raise PlanInputError("horizons longer than five years are beyond what this history can test")
    label = f"{count:g} {key}{'' if count == 1 else 's'}"
    return days, label


def parse_max_loss(text: str, amount: float) -> float:
    """'$200' or '20%' -> dollars, at most the amount."""
    cleaned = text.strip().lower().replace(",", "")
    percent = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", cleaned)
    if percent:
        value = amount * float(percent.group(1)) / 100.0
    else:
        value = _dollars(cleaned) if cleaned.replace("$", "").strip() else 0.0
    if value <= 0:
        raise PlanInputError("the most you could stand to lose has to be more than zero")
    return min(value, amount)
