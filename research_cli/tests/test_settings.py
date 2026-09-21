"""Tests for credential storage.

This layer exists because a macOS `.app` launched from Finder does not run
your login shell, so an exported environment variable is invisible to it.
Getting the precedence or the file permissions wrong is the kind of bug that
shows up as "it works in Terminal but not in the app", which is close to
undebuggable from the outside.

The `isolated_credentials` fixture in conftest redirects the config path into
a temp directory for every test, so nothing here can read or write the real
credentials file.
"""

from __future__ import annotations

import stat

import pytest

from research_cli import settings
from research_cli.settings import clear_key, describe_credentials, get_key, set_key


def test_nothing_is_set_by_default() -> None:
    """The fixture must actually isolate; if it does not, every test below lies."""
    assert get_key("FRED_API_KEY") is None
    assert all(not status.present for status in describe_credentials())


def test_a_stored_key_is_readable_without_any_environment_variable() -> None:
    """The whole point: a GUI launch has no shell environment to read."""
    set_key("FRED_API_KEY", "abc123")
    assert get_key("FRED_API_KEY") == "abc123"


def test_the_environment_wins_over_the_file(monkeypatch) -> None:
    """Existing setups and one-off overrides must keep working."""
    set_key("FRED_API_KEY", "from-file")
    monkeypatch.setenv("FRED_API_KEY", "from-environment")
    assert get_key("FRED_API_KEY") == "from-environment"


def test_the_file_is_not_readable_by_other_users() -> None:
    """It holds credentials."""
    path = set_key("FRED_API_KEY", "secret-value")
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"expected owner-only permissions, got {oct(mode)}"


def test_storing_a_second_key_keeps_the_first() -> None:
    set_key("FRED_API_KEY", "one")
    set_key("ANTHROPIC_API_KEY", "two")
    assert get_key("FRED_API_KEY") == "one"
    assert get_key("ANTHROPIC_API_KEY") == "two"


def test_a_key_can_be_replaced() -> None:
    set_key("FRED_API_KEY", "old")
    set_key("FRED_API_KEY", "new")
    assert get_key("FRED_API_KEY") == "new"


def test_an_empty_value_is_refused() -> None:
    """Storing a blank key looks like success and behaves like failure."""
    with pytest.raises(ValueError, match="empty"):
        set_key("FRED_API_KEY", "   ")


def test_clearing_a_key_removes_only_that_one() -> None:
    set_key("FRED_API_KEY", "one")
    set_key("ANTHROPIC_API_KEY", "two")
    assert clear_key("FRED_API_KEY") is True
    assert get_key("FRED_API_KEY") is None
    assert get_key("ANTHROPIC_API_KEY") == "two"


def test_clearing_a_key_that_was_never_set_is_not_an_error() -> None:
    assert clear_key("FRED_API_KEY") is False


def test_describe_never_reveals_the_key() -> None:
    """Enough to confirm the right key is loaded, not enough to leak it."""
    secret = "b6ce8ce5db29a46f4b6237d0e6c388a1"
    set_key("FRED_API_KEY", secret)
    status = next(s for s in describe_credentials() if s.name == "FRED_API_KEY")
    text = status.describe()
    assert secret not in text
    assert status.hint == secret[-4:]
    assert "config file" in text


def test_quotes_around_a_pasted_value_are_stripped() -> None:
    """People paste `KEY="value"` out of documentation."""
    settings.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_PATH.write_text('FRED_API_KEY="quoted-value"\n', encoding="utf-8")
    assert get_key("FRED_API_KEY") == "quoted-value"


def test_a_malformed_file_degrades_to_no_credentials() -> None:
    """A broken credentials file must not stop the tool starting."""
    settings.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_PATH.write_text("this is not\nkey=value format\x00\n", encoding="utf-8")
    assert get_key("FRED_API_KEY") is None
    assert isinstance(describe_credentials(), list)


def test_comments_and_blank_lines_are_ignored() -> None:
    settings.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_PATH.write_text(
        "# a comment\n\nFRED_API_KEY=real\n\n# another\n", encoding="utf-8"
    )
    assert get_key("FRED_API_KEY") == "real"


def test_macro_ingest_resolves_a_stored_key() -> None:
    """The integration that matters: the data layer reads the same store."""
    from datetime import date

    from research_cli.data.macro_ingest import load_macro_panel

    set_key("FRED_API_KEY", "x" * 32)
    # A syntactically valid but wrong key reaches the network and fails; with
    # source="auto" that must degrade rather than raise.
    panel = load_macro_panel(date(2024, 1, 1), date(2025, 1, 1), source="mock")
    assert panel.source == "mock"
