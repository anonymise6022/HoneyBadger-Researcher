"""The local model backend, tested without the model.

Four gigabytes of weights cannot be a test dependency, so everything here
drives the surrounding logic with the generation function faked: which
backend gets chosen, what happens when the weights are missing, whether the
extra prompt block reaches the local path and stays away from the Claude
one, and whether a local failure degrades to the template rather than
raising.

The generation itself is not tested here, because there is nothing this
suite could assert about it that the validator does not already enforce at
runtime on every report.
"""

from __future__ import annotations

import pytest

from research_cli.query_parser import parse_query
from research_cli.evidence.why_moved import build_why_moved_bundle
from research_cli.synthesis import llm_report, local_llm
from research_cli.synthesis.llm_report import (
    FINAL_CHECKLIST,
    LLMUnavailable,
    build_messages,
    generate_report,
    resolve_backend,
)


@pytest.fixture
def bundle():
    return build_why_moved_bundle(parse_query("why did SPY fall today"), source="mock")


def _ready(monkeypatch, ready: bool, detail: str = "not downloaded") -> None:
    """Pretend the local backend is or is not ready to run."""
    status = local_llm.LocalStatus(
        supported=ready, installed=ready, downloaded=ready,
        model=local_llm.DEFAULT_LOCAL_MODEL, detail=detail,
    )
    monkeypatch.setattr(local_llm, "describe", lambda model=None: status)


class TestStatus:
    def test_a_machine_without_apple_silicon_is_reported_not_supported(self, monkeypatch) -> None:
        monkeypatch.setattr(local_llm.platform, "system", lambda: "Linux")
        monkeypatch.setattr(local_llm.platform, "machine", lambda: "x86_64")
        status = local_llm.describe()
        assert not status.supported and not status.ready
        assert "Apple Silicon" in status.detail

    def test_missing_weights_are_not_ready_but_are_supported(self, monkeypatch) -> None:
        monkeypatch.setattr(local_llm, "_apple_silicon", lambda: True)
        monkeypatch.setattr(local_llm, "_mlx_installed", lambda: True)
        monkeypatch.setattr(local_llm, "weights_present", lambda model=None: False)
        status = local_llm.describe()
        assert status.supported and status.installed and not status.downloaded
        assert not status.ready
        assert "download" in status.detail.lower()

    def test_asking_whether_it_is_available_never_downloads(self, monkeypatch) -> None:
        """The check must not be the thing that pulls four gigabytes."""
        called = []
        monkeypatch.setattr(local_llm, "ensure_weights", lambda *a, **k: called.append(1))
        monkeypatch.setattr(local_llm, "_apple_silicon", lambda: True)
        monkeypatch.setattr(local_llm, "_mlx_installed", lambda: True)
        monkeypatch.setattr(local_llm, "weights_present", lambda model=None: False)
        assert local_llm.available() is False
        assert called == []

    def test_loading_refuses_when_the_weights_are_absent(self, monkeypatch) -> None:
        _ready(monkeypatch, False)
        local_llm._loaded.clear()
        with pytest.raises(local_llm.LocalUnavailable):
            local_llm._load(local_llm.DEFAULT_LOCAL_MODEL)


class TestThinkingBlocks:
    def test_reasoning_is_stripped_from_the_report(self) -> None:
        """Qwen writes its scratchpad first; it must not reach the reader."""
        raw = "<think>The user wants a note. Let me plan.</think>\n\nQUESTION\nWhy did it fall?"
        assert local_llm._THINK_BLOCK.sub("", raw).strip().startswith("QUESTION")

    def test_text_without_a_block_is_untouched(self) -> None:
        raw = "QUESTION\nWhy did it fall?"
        assert local_llm._THINK_BLOCK.sub("", raw) == raw


class TestBackendSelection:
    def test_local_is_preferred_when_it_is_ready(self, monkeypatch) -> None:
        _ready(monkeypatch, True)
        monkeypatch.setattr(llm_report, "claude_available", lambda: True)
        assert resolve_backend() == "local"

    def test_claude_is_used_when_no_local_model_is_downloaded(self, monkeypatch) -> None:
        _ready(monkeypatch, False)
        monkeypatch.setattr(llm_report, "claude_available", lambda: True)
        assert resolve_backend() == "claude"

    def test_neither_available_is_not_an_error_but_a_refusal(self, monkeypatch) -> None:
        _ready(monkeypatch, False)
        monkeypatch.setattr(llm_report, "claude_available", lambda: False)
        with pytest.raises(LLMUnavailable):
            resolve_backend()
        assert llm_report.llm_available() is False

    def test_an_explicit_preference_beats_the_default_order(self, monkeypatch) -> None:
        _ready(monkeypatch, True)
        monkeypatch.setattr(llm_report, "claude_available", lambda: True)
        assert resolve_backend("claude") == "claude"

    def test_asking_for_local_when_it_is_not_ready_says_why(self, monkeypatch) -> None:
        _ready(monkeypatch, False, detail="Ready to download the model (about 4GB, once).")
        with pytest.raises(LLMUnavailable, match="4GB"):
            resolve_backend("local")

    def test_an_unknown_backend_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError):
            resolve_backend("gpt")


class TestPrompt:
    def test_the_checklist_reaches_the_local_model(self, bundle) -> None:
        instruction = build_messages(bundle, emphasise_rules=True)[0]["content"]
        assert FINAL_CHECKLIST.strip() in instruction
        assert instruction.rstrip().endswith("in the order given.")

    def test_claude_does_not_get_it(self, bundle) -> None:
        """One prompt with one appended block, not two prompts that drift."""
        assert FINAL_CHECKLIST.strip() not in build_messages(bundle)[0]["content"]

    def test_the_checklist_names_the_banned_words_again(self) -> None:
        """The failure it exists to fix: a small model forgetting a rule
        stated two thousand words earlier."""
        for word in ("because", "caused", "due to", "triggered"):
            assert word in FINAL_CHECKLIST


class TestGeneration:
    def test_a_local_report_runs_the_whole_validation_loop(self, bundle, monkeypatch) -> None:
        from research_cli.tests.test_llm_report import _beginner_prose

        _ready(monkeypatch, True)
        monkeypatch.setattr(
            local_llm, "generate",
            lambda messages, system, model=None, max_tokens=None: (_beginner_prose(bundle), 900, 400),
        )
        result = generate_report(bundle, backend="local")
        assert result.usable
        assert result.model == local_llm.DEFAULT_LOCAL_MODEL
        assert result.attempts == 1
        assert result.validation.ok

    def test_prose_that_fails_the_trace_is_discarded(self, bundle, monkeypatch) -> None:
        """Same guarantee as the Claude path: unvalidated prose is never
        shown, whoever wrote it."""
        _ready(monkeypatch, True)
        monkeypatch.setattr(
            local_llm, "generate",
            lambda *a, **k: ("SUMMARY\nSPY fell 93.7% because of the news.", 900, 400),
        )
        result = generate_report(bundle, backend="local")
        assert not result.usable
        assert result.attempts == 2  # one retry with the failures quoted back

    def test_a_model_failure_becomes_an_unavailable_not_a_crash(self, bundle, monkeypatch) -> None:
        _ready(monkeypatch, True)

        def explode(*args, **kwargs):
            raise local_llm.LocalUnavailable("out of memory")

        monkeypatch.setattr(local_llm, "generate", explode)
        with pytest.raises(LLMUnavailable, match="out of memory"):
            generate_report(bundle, backend="local")

    def test_supplying_a_client_still_means_claude(self, bundle, monkeypatch) -> None:
        """Every existing caller and test passes a fake Anthropic client, and
        a client can only be one thing."""
        from research_cli.tests.test_llm_report import FakeClient, _beginner_prose

        _ready(monkeypatch, True)  # local is ready, and must still lose
        result = generate_report(bundle, client=FakeClient([_beginner_prose(bundle)]))
        assert result.model == llm_report.DEFAULT_MODEL
