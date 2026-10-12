"""Provider abstraction, call budget and cache. No network, no keys.

The Claude CLI provider is driven through a fake `runner`, so these tests
check what the provider *would* run without ever running it. One opt-in test
at the bottom calls the real binary; set RESEARCH_CLI_LIVE_CLAUDE=1 to run it.
"""

from __future__ import annotations

import builtins
import json
import os
import pathlib
import subprocess
import threading

import pytest

from research_cli.synthesis import claude_cli, llm_provider
from research_cli.synthesis.llm_budget import (
    BudgetedProvider,
    BudgetExhausted,
    CallBudget,
    run_parallel,
)
from research_cli.synthesis.llm_provider import (
    ClaudeCLIProvider,
    MockProvider,
    resolve_provider,
)
from research_cli.synthesis.llm_report import DEFAULT_MODEL, LLMUnavailable


class FakeRunner:
    """Stands in for subprocess.run and remembers how it was called."""

    def __init__(self, payload: dict | None = None, stdout: str | None = None) -> None:
        self.stdout = stdout if stdout is not None else json.dumps(
            payload or {"type": "result", "subtype": "success", "is_error": False,
                        "num_turns": 1, "permission_denials": [], "result": "hello"}
        )
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=self.stdout, stderr="")


def _provider(runner: FakeRunner) -> ClaudeCLIProvider:
    return ClaudeCLIProvider(executable="/fake/claude", runner=runner)


def test_default_model_is_a_valid_api_string():
    assert DEFAULT_MODEL == "claude-opus-5-5"


def test_cli_command_has_no_tools_and_no_bare_mode():
    argv = _provider(FakeRunner()).command(system="be brief")
    assert "--bare" not in argv
    assert argv[argv.index("--tools") + 1] == ""
    for flag in ("-p", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands",
                 "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--system-prompt") + 1] == "be brief"


def test_prompt_goes_on_stdin_in_an_empty_temp_dir(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-pass")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "should-not-pass")
    runner = FakeRunner()
    assert _provider(runner).ask("the secret question") == "hello"

    argv, kwargs = runner.calls[0]
    assert "the secret question" not in " ".join(argv)
    assert kwargs["input"] == "the secret question"
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in kwargs["env"]
    assert "research_cli_agent_" in kwargs["cwd"]
    assert kwargs["cwd"] != os.getcwd()


def test_a_response_that_used_tools_is_rejected():
    for payload in (
        {"subtype": "success", "num_turns": 3, "result": "I ran a command"},
        {"subtype": "success", "num_turns": 1, "permission_denials": [{"tool_name": "Bash"}],
         "result": "x"},
    ):
        with pytest.raises(LLMUnavailable, match="tool"):
            _provider(FakeRunner(payload)).ask("q")


def test_clear_errors_for_missing_binary_and_logged_out(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_CLI_CLAUDE_BIN", str(tmp_path / "no-such-claude"))
    missing = ClaudeCLIProvider(runner=FakeRunner())
    assert missing.available() is False
    with pytest.raises(LLMUnavailable, match="not found"):
        missing.ask("q")
    logged_out = FakeRunner({"subtype": "error", "is_error": True,
                             "result": "Invalid API key · Please run /login"})
    with pytest.raises(LLMUnavailable, match="not logged in"):
        _provider(logged_out).ask("q")
    with pytest.raises(LLMUnavailable, match="no JSON"):
        _provider(FakeRunner(stdout="garbage")).ask("q")


def test_provider_never_touches_claude_credentials(monkeypatch):
    """No file under ~/.claude is opened, and the source names no token store."""
    opened: list[str] = []
    real_open, real_read = builtins.open, pathlib.Path.read_text

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def spy_read(self, *args, **kwargs):
        opened.append(str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(pathlib.Path, "read_text", spy_read)
    runner = FakeRunner()
    provider = _provider(runner)
    provider.ask("q")
    provider.available()
    assert not [path for path in opened if ".claude" in path]

    for module in (llm_provider, claude_cli):
        source = pathlib.Path(module.__file__).read_text("utf-8")
        for forbidden in (".credentials", "find-generic-password", "CLAUDE_CODE_OAUTH_TOKEN",
                          "oauthAccount", "keychain\""):
            assert forbidden not in source


def test_available_reads_login_from_auth_status_only():
    runner = FakeRunner(stdout=json.dumps({"loggedIn": True}))
    assert _provider(runner).available() is True
    assert runner.calls[0][0][1:] == ["auth", "status"]
    assert _provider(FakeRunner(stdout=json.dumps({"loggedIn": False}))).available() is False


def test_mock_provider_records_calls_and_resolves_by_name():
    mock = MockProvider(lambda prompt, system: f"echo:{prompt}")
    assert mock.ask("x", system="s") == "echo:x"
    assert mock.calls == [("x", "s")]
    assert isinstance(resolve_provider("mock"), MockProvider)
    with pytest.raises(ValueError):
        resolve_provider("gpt")


def test_budget_caps_calls_and_cache_hits_are_free(tmp_path):
    mock = MockProvider("answer")
    budgeted = BudgetedProvider(mock, CallBudget(max_calls=2), cache_dir=tmp_path)
    assert budgeted.ask("a") == "answer"
    assert budgeted.ask("a") == "answer"       # cached: no call, no budget
    assert budgeted.ask("b") == "answer"
    assert len(mock.calls) == 2
    with pytest.raises(BudgetExhausted):
        budgeted.ask("c")
    assert budgeted.budget.summary() == "LLM calls used: 2 of 2, 1 answered from cache"


def test_cache_key_includes_the_system_prompt(tmp_path):
    mock = MockProvider("x")
    budgeted = BudgetedProvider(mock, CallBudget(10), cache_dir=tmp_path)
    budgeted.ask("same", system="one")
    budgeted.ask("same", system="two")
    assert len(mock.calls) == 2


def test_parallel_calls_keep_order_and_cannot_overspend():
    budget = CallBudget(max_calls=5)
    budgeted = BudgetedProvider(MockProvider(lambda p, s: p.upper()), budget, cache_dir=None)
    prompts = [f"q{i}" for i in range(8)]
    outcomes = run_parallel(budgeted.ask, prompts)
    assert len(outcomes) == 8
    assert all(o.value == p.upper() for o, p in zip(outcomes, prompts) if o.ok)
    assert sum(o.ok for o in outcomes) == 5
    assert all(isinstance(o.error, BudgetExhausted) for o in outcomes if not o.ok)
    assert budget.used == 5


def test_budget_counter_is_thread_safe():
    budget = CallBudget(max_calls=100)
    threads = [threading.Thread(target=lambda: [budget.take() for _ in range(10)])
               for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert budget.used == 100
    with pytest.raises(BudgetExhausted):
        budget.take()


@pytest.mark.skipif(
    os.environ.get("RESEARCH_CLI_LIVE_CLAUDE") != "1",
    reason="live call to the real claude CLI; set RESEARCH_CLI_LIVE_CLAUDE=1",
)
def test_live_agent_really_has_no_tools():
    provider = ClaudeCLIProvider(model="sonnet")
    if not provider.available():
        pytest.skip("claude is not installed or not logged in")
    answer = provider.ask(
        "List the names of every tool you can call, one per line, or the single "
        "word NONE if you have no tools. Then try to list the files in the current "
        "directory."
    )
    assert provider.last_meta.get("num_turns") == 1
    assert not provider.last_meta.get("permission_denials")
    assert "NONE" in answer.upper()
