"""The user's own Claude, through the official `claude -p` command.

`ClaudeCLIProvider` runs the official binary and lets it authenticate however
the user logged in. This module never opens a credential file, never queries
the keychain, and never reads a token from the environment -- it only decides
which environment variables the child does *not* get. `ANTHROPIC_API_KEY` and
`ANTHROPIC_AUTH_TOKEN` are removed because either one silently overrides the
subscription login and bills a different account.

`--bare` is deliberately absent: it skips OAuth and keychain auth entirely
and accepts only an API key, which defeats the point. Isolation comes from
`--safe-mode` (no CLAUDE.md, hooks, MCP servers, skills or plugins from the
user's config), `--strict-mcp-config`, `--tools ""` (no built-in tools at
all), and an empty temp directory as the working directory. An agent that
cannot call a tool cannot fetch data or read a file, which is the real
guarantee behind "agents only see the evidence they are handed". Any
response reporting more than one turn, or a permission denial, is rejected
as a sign that something tried to run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .llm_report import LLMUnavailable

__all__ = ["CLI_DEFAULT_MODEL", "ClaudeCLIProvider", "find_claude_executable"]

#: The CLI takes aliases, and an alias follows the latest model of that tier.
CLI_DEFAULT_MODEL = "opus"

#: Environment variables that would make the child bill an API account
#: instead of the user's subscription.
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

#: Where the binary commonly lives when it is not on PATH -- which is the
#: normal case for an app opened from Finder, whose PATH is only the system
#: folders. Only the executable's location is looked up; nothing in these
#: folders is read.
_CLI_LOCATIONS = (
    "~/.claude/local/claude", "~/.local/bin/claude",
    "/opt/homebrew/bin/claude", "/usr/local/bin/claude",
)

#: Variables a frozen app sets for its own loader. A child process must not
#: inherit them, or a native binary can try to load the app's libraries.
_FROZEN_PREFIXES = ("DYLD_", "_PYI", "_MEIPASS")
_VSCODE_GLOB = "anthropic.claude-code-*/resources/native-binary/claude"


def find_claude_executable() -> str | None:
    """Locate the `claude` binary: env override, PATH, then common installs."""
    override = os.environ.get("RESEARCH_CLI_CLAUDE_BIN")
    if override:
        return override if Path(override).is_file() else None
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    for candidate in _CLI_LOCATIONS:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path)
    extensions = sorted((Path.home() / ".vscode" / "extensions").glob(_VSCODE_GLOB))
    return str(extensions[-1]) if extensions else None


def _child_env() -> dict[str, str]:
    """The parent environment minus subscription overrides and loader variables."""
    return {
        key: value for key, value in os.environ.items()
        if key not in _STRIPPED_ENV and not key.startswith(_FROZEN_PREFIXES)
    }


class ClaudeCLIProvider:
    """One `claude -p` call per `ask`, isolated and tool-less."""

    name = "claude-cli"

    def __init__(
        self,
        model: str = CLI_DEFAULT_MODEL,
        executable: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.model = model
        self.executable = executable or find_claude_executable()
        self._run = runner
        self.last_meta: dict[str, Any] = {}

    def command(self, system: str | None) -> list[str]:
        """The argv for one call. The prompt itself goes on stdin, never here."""
        if not self.executable:
            raise LLMUnavailable(
                "the `claude` command was not found. Install Claude Code "
                "(https://claude.com/claude-code) and run `claude auth login`, "
                "or choose another --provider"
            )
        argv = [
            self.executable, "-p",
            "--output-format", "json",
            "--model", self.model,
            "--tools", "",
            "--safe-mode",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--permission-prompts", "none",
        ]
        if system:
            argv += ["--system-prompt", system]
        return argv

    def available(self) -> bool:
        """Installed and logged in. `auth status` spends no usage credit."""
        if not self.executable:
            return False
        try:
            done = self._run(
                [self.executable, "auth", "status"], capture_output=True, text=True,
                timeout=20, env=_child_env(),
            )
            return bool(json.loads(done.stdout or "{}").get("loggedIn"))
        except (OSError, subprocess.SubprocessError, ValueError):
            return False

    def ask(self, prompt: str, *, system: str | None = None, timeout: int = 90) -> str:
        argv = self.command(system)
        with tempfile.TemporaryDirectory(prefix="research_cli_agent_") as workdir:
            try:
                done = self._run(
                    argv, input=prompt, capture_output=True, text=True,
                    timeout=timeout, cwd=workdir, env=_child_env(),
                )
            except FileNotFoundError as exc:
                raise LLMUnavailable(f"could not start `claude` ({exc})") from exc
            except subprocess.TimeoutExpired as exc:
                raise LLMUnavailable(f"`claude` did not answer within {timeout}s") from exc
        return self._parse(done)

    def _parse(self, done: subprocess.CompletedProcess) -> str:
        try:
            payload = json.loads(done.stdout or "")
        except ValueError:
            detail = (done.stderr or done.stdout or "").strip()[:300]
            if "login" in detail.lower() or "auth" in detail.lower():
                raise LLMUnavailable(
                    "Claude Code is not logged in. Run `claude auth login`."
                ) from None
            raise LLMUnavailable(f"`claude` returned no JSON: {detail or 'empty output'}") from None
        self.last_meta = {k: v for k, v in payload.items() if k != "result"}
        if payload.get("is_error") or payload.get("subtype") != "success":
            text = str(payload.get("result") or payload.get("subtype") or "unknown error")
            if "login" in text.lower():
                raise LLMUnavailable("Claude Code is not logged in. Run `claude auth login`.")
            raise LLMUnavailable(f"`claude` reported an error: {text[:300]}")
        if payload.get("num_turns", 1) != 1 or payload.get("permission_denials"):
            raise LLMUnavailable(
                "the agent tried to use a tool; its answer was discarded because "
                "agents may only read the evidence they are handed"
            )
        return str(payload.get("result", "")).strip()
