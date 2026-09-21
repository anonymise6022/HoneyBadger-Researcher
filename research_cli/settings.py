"""Where API keys live, so the CLI and the desktop app agree.

macOS makes this less trivial than it looks. A `.app` launched from Finder
does **not** run your login shell, so `export FRED_API_KEY=...` in `.zshrc`
is invisible to it. A user who sets the variable, confirms it works in
Terminal, then double-clicks the app and sees synthetic data again has no
way to guess why. Windows has the same split between a shell session and a
double-clicked executable.

So credentials resolve from two places, in order:

1. **The environment.** Always wins, so CI, a one-off
   `FRED_API_KEY=... research ...`, and existing shell setups keep working.
2. **A config file** at `~/.config/research_cli/keys.env`, which every
   launch path can read regardless of how it was started.

The file is plain `KEY=value` lines. It is written with owner-only
permissions (0600) because it holds credentials, and its contents are never
logged or echoed -- `describe_credentials` reports only whether a key is
present and what its last four characters are, which is enough to confirm
the right key is loaded without putting it on screen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CONFIG_PATH",
    "CredentialStatus",
    "clear_key",
    "describe_credentials",
    "get_key",
    "set_key",
]

CONFIG_DIR = Path.home() / ".config" / "research_cli"
CONFIG_PATH = CONFIG_DIR / "keys.env"

#: Keys the tool knows about, with what each unlocks.
KNOWN_KEYS: dict[str, str] = {
    "FRED_API_KEY": "real macro data (inflation, unemployment, yields, credit spreads)",
    "ANTHROPIC_API_KEY": "reports written in plainer language by Claude",
    "POLYGON_API_KEY": "option chains, for the separate options pipeline",
}


@dataclass(frozen=True)
class CredentialStatus:
    """Whether one key is available, and where it came from."""

    name: str
    present: bool
    source: str  # "environment", "config file", or "not set"
    hint: str    # last four characters, or ""
    unlocks: str

    def describe(self) -> str:
        if not self.present:
            return f"{self.name}: not set -- without it, {self.unlocks} is unavailable"
        return f"{self.name}: set (...{self.hint}), from the {self.source}"


def _read_config() -> dict[str, str]:
    """Parse the config file. A malformed file is ignored, never fatal.

    A broken credentials file must degrade to "no credentials", not to a
    crash on startup -- the tool's whole promise is that it runs without any.
    """
    try:
        if not CONFIG_PATH.is_file():
            return {}
        values: dict[str, str] = {}
        for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            cleaned = value.strip().strip("'\"")
            if cleaned:
                values[name.strip()] = cleaned
        return values
    except OSError:
        return {}


def get_key(name: str) -> str | None:
    """Resolve one credential: environment first, then the config file."""
    from_environment = os.environ.get(name)
    if from_environment:
        return from_environment
    return _read_config().get(name)


def set_key(name: str, value: str) -> Path:
    """Store a credential in the config file, replacing any existing value.

    Written with owner-only permissions. Raises ValueError for an empty
    value, since silently storing a blank key would look like success and
    behave like failure.
    """
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"the value for {name} is empty")

    values = _read_config()
    values[name] = cleaned
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    body = (
        "# research_cli credentials. Owner-readable only.\n"
        "# Delete a line to remove that key. Environment variables win over this file.\n"
        + "\n".join(f"{key}={val}" for key, val in sorted(values.items()))
        + "\n"
    )
    CONFIG_PATH.write_text(body, encoding="utf-8")
    CONFIG_PATH.chmod(0o600)
    return CONFIG_PATH


def clear_key(name: str) -> bool:
    """Remove a credential from the config file. Returns whether it was there."""
    values = _read_config()
    if name not in values:
        return False
    del values[name]
    if values:
        CONFIG_PATH.write_text(
            "# research_cli credentials. Owner-readable only.\n"
            + "\n".join(f"{key}={val}" for key, val in sorted(values.items()))
            + "\n",
            encoding="utf-8",
        )
        CONFIG_PATH.chmod(0o600)
    else:
        CONFIG_PATH.unlink(missing_ok=True)
    return True


def describe_credentials() -> list[CredentialStatus]:
    """Report every known key without revealing any of them."""
    config = _read_config()
    statuses: list[CredentialStatus] = []
    for name, unlocks in KNOWN_KEYS.items():
        environment_value = os.environ.get(name)
        file_value = config.get(name)
        value = environment_value or file_value
        statuses.append(
            CredentialStatus(
                name=name,
                present=bool(value),
                source=(
                    "environment" if environment_value
                    else "config file" if file_value
                    else "not set"
                ),
                hint=value[-4:] if value and len(value) >= 4 else "",
                unlocks=unlocks,
            )
        )
    return statuses
