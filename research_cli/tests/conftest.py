"""Shared fixtures. Every test runs on synthetic data -- no network, no keys."""

from __future__ import annotations

from datetime import date

import pytest

from research_cli import settings
from research_cli.settings import KNOWN_KEYS


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    """Run every test as though the machine has no API keys configured.

    Without this the suite depends on developer machine state: once a real
    FRED key is stored, the "no key falls back to mock" tests start failing
    on that machine and passing everywhere else. A test that flips based on
    whose laptop it runs on is worse than no test.

    Both credential sources are neutralized -- the environment variables and
    the on-disk config file, which is redirected into a temp directory so a
    test that writes a key cannot touch the real one.
    """
    for name in KNOWN_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(settings, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config" / "keys.env")
    return tmp_path

from research_cli.evidence.why_moved import build_why_moved_bundle
from research_cli.query_parser import parse_query

TODAY = date(2026, 9, 21)


@pytest.fixture(scope="session")
def today() -> date:
    return TODAY


@pytest.fixture(scope="session")
def spy_bundle():
    """A full attribution bundle built entirely from synthetic data."""
    return build_why_moved_bundle(
        parse_query("why did SPY fall today", today=TODAY), source="mock", history_years=4.0
    )


@pytest.fixture(scope="session")
def aapl_bundle():
    """A single-stock bundle, which exercises the market-beta branch."""
    return build_why_moved_bundle(
        parse_query("why did AAPL fall today", today=TODAY), source="mock", history_years=4.0
    )
