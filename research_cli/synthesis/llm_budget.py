"""Rationing model calls: a hard cap, a disk cache, and parallel fan-out.

`claude -p` on a subscription draws from a limited monthly allowance, so a
single question must not be able to spend it freely. Three mechanisms:

**A hard cap per question.** `CallBudget` counts calls that actually reach a
model and refuses the one past the limit with `BudgetExhausted`. The caller
treats that exactly like an unavailable model: it stops asking and falls
back to the deterministic report.

**A cache keyed by the exact request.** The key hashes provider, model,
system prompt and prompt, so the same question over the same evidence costs
nothing the second time. A cached answer is not trusted any more than a fresh
one -- it goes through the same validation -- and it does not count against
the cap. Prompts must therefore not embed volatile values such as a
generation timestamp, or every key is new.

**Parallel calls.** Independent agents are asked at the same time through a
small thread pool. The budget's counter is locked, so concurrent calls cannot
overspend it.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .llm_report import LLMUnavailable
from .llm_provider import LLMProvider

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MAX_CALLS",
    "BudgetExhausted",
    "BudgetedProvider",
    "CallBudget",
    "Outcome",
    "run_parallel",
]

DEFAULT_MAX_CALLS = 12
#: Beside the source when run from a checkout; in the user's cache folder
#: when frozen into an app, which must never write inside its own bundle.
DEFAULT_CACHE_DIR = (
    Path.home() / "Library" / "Caches" / "research_cli" / "llm"
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[1] / ".cache" / "llm"
)
_MAX_WORKERS = 4

T = TypeVar("T")
R = TypeVar("R")


class BudgetExhausted(LLMUnavailable):
    """The per-question call limit was reached. Handled like no model at all."""


class CallBudget:
    """A thread-safe count of model calls against a fixed ceiling."""

    def __init__(self, max_calls: int = DEFAULT_MAX_CALLS) -> None:
        if max_calls < 0:
            raise ValueError(f"max_calls cannot be negative, got {max_calls}")
        self.max_calls = max_calls
        self.used = 0
        self.cache_hits = 0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        return max(self.max_calls - self.used, 0)

    def take(self) -> None:
        """Reserve one call, or raise when none are left."""
        with self._lock:
            if self.used >= self.max_calls:
                raise BudgetExhausted(
                    f"the limit of {self.max_calls} model calls for this question was reached"
                )
            self.used += 1

    def note_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def summary(self) -> str:
        cached = f", {self.cache_hits} answered from cache" if self.cache_hits else ""
        return f"LLM calls used: {self.used} of {self.max_calls}{cached}"


class BudgetedProvider:
    """A provider wrapped with the call cap and the on-disk cache.

    `cache_dir=None` disables caching, which tests use to keep each run
    independent of the last.
    """

    def __init__(
        self,
        provider: LLMProvider,
        budget: CallBudget | None = None,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
    ) -> None:
        self.provider = provider
        self.budget = budget or CallBudget()
        self.cache_dir = cache_dir
        self.name = provider.name
        self.model = provider.model

    def _key(self, prompt: str, system: str | None) -> str:
        material = json.dumps([self.provider.name, self.provider.model, system, prompt])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _read(self, key: str) -> str | None:
        if self.cache_dir is None:
            return None
        try:
            return json.loads((self.cache_dir / f"{key}.json").read_text("utf-8"))["text"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write(self, key: str, text: str) -> None:
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_dir / f"{key}.tmp"
            temporary.write_text(json.dumps({"text": text}), "utf-8")
            temporary.replace(self.cache_dir / f"{key}.json")
        except OSError:  # caching is best-effort and must never fail a question
            pass

    def forget(self, prompt: str, system: str | None = None) -> None:
        """Drop a cached answer that failed its checks, so a re-run asks afresh.

        Without this a rejected draft would be replayed from disk on every
        later run of the same question, and the user could never get past it.
        """
        if self.cache_dir is None:
            return
        try:
            (self.cache_dir / f"{self._key(prompt, system)}.json").unlink(missing_ok=True)
        except OSError:
            pass

    def ask(self, prompt: str, *, system: str | None = None, timeout: int = 90) -> str:
        key = self._key(prompt, system)
        cached = self._read(key)
        if cached is not None:
            self.budget.note_cache_hit()
            return cached
        self.budget.take()
        text = self.provider.ask(prompt, system=system, timeout=timeout)
        self._write(key, text)
        return text


@dataclass(frozen=True)
class Outcome:
    """One parallel call's result: a value, or the error that replaced it."""

    value: Any = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_parallel(
    function: Callable[[T], R], items: Sequence[T], max_workers: int = _MAX_WORKERS
) -> list[Outcome]:
    """Apply `function` to each item concurrently, keeping input order.

    Errors are captured per item rather than raised, so one agent failing
    costs that agent's answer and not the whole round.
    """
    def guarded(item: T) -> Outcome:
        try:
            return Outcome(value=function(item))
        except Exception as exc:  # noqa: BLE001 - reported per item, never swallowed
            return Outcome(error=exc)

    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(guarded, items))
