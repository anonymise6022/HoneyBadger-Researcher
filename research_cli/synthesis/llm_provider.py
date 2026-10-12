"""One interface over every model that can write for this tool.

The debate layer asks several agents for short JSON answers. It should not
care whether those answers come from the user's own Claude subscription, an
API key, a model on this laptop, or a test fixture, so each of those sits
behind the same one-method protocol: `ask(prompt, system=..., timeout=...)`.

The subscription path lives in `claude_cli`, which states exactly how it
keeps credentials untouched and agents tool-less. `resolve_provider("auto")`
prefers it, then an API key, then the local model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .claude_cli import CLI_DEFAULT_MODEL, ClaudeCLIProvider, find_claude_executable
from .llm_report import DEFAULT_MODEL, LLMUnavailable

__all__ = [
    "PROVIDER_NAMES",
    "APIProvider",
    "ClaudeCLIProvider",
    "LLMProvider",
    "LocalProvider",
    "MockProvider",
    "find_claude_executable",
    "resolve_provider",
]

PROVIDER_NAMES = ("auto", "claude-cli", "api", "local", "mock")


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that turns a prompt into text."""

    name: str
    model: str

    def ask(self, prompt: str, *, system: str | None = None, timeout: int = 90) -> str:
        ...


class APIProvider:
    """Claude through an API key, for users who have one configured."""

    name = "api"

    def __init__(self, model: str = DEFAULT_MODEL, client: Any | None = None) -> None:
        self.model = model
        self._client = client

    def ask(self, prompt: str, *, system: str | None = None, timeout: int = 90) -> str:
        from ..settings import get_key  # noqa: PLC0415
        from .llm_report import claude_available  # noqa: PLC0415

        if self._client is None and not claude_available():
            raise LLMUnavailable("no ANTHROPIC_API_KEY is configured")
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise LLMUnavailable("the anthropic package is not installed") from exc

        client = self._client
        if client is None:
            stored = get_key("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=stored) if stored else anthropic.Anthropic()
        request: dict[str, Any] = {
            "model": self.model, "max_tokens": 4096, "timeout": timeout,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system
        try:
            response = client.messages.create(**request)
        except anthropic.APIError as exc:
            raise LLMUnavailable(f"Claude API error: {exc}") from exc
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()


class LocalProvider:
    """The on-device MLX model. Weak at strict JSON, so expect more fallbacks."""

    name = "local"

    def __init__(self, model: str | None = None) -> None:
        from . import local_llm  # noqa: PLC0415

        self.model = model or local_llm.DEFAULT_LOCAL_MODEL

    def ask(self, prompt: str, *, system: str | None = None, timeout: int = 90) -> str:
        from . import local_llm  # noqa: PLC0415

        try:
            text, _, _ = local_llm.generate(
                [{"role": "user", "content": prompt}], system or "", model=self.model
            )
        except local_llm.LocalUnavailable as exc:
            raise LLMUnavailable(str(exc)) from exc
        return text


class MockProvider:
    """Canned answers for tests: no network, no keys, every call recorded.

    `responder` is a fixed string or a function of (prompt, system).
    """

    name = "mock"

    def __init__(
        self, responder: str | Callable[[str, str | None], str] = "{}", model: str = "mock"
    ) -> None:
        self.model = model
        self._responder = responder
        self.calls: list[tuple[str, str | None]] = []

    def ask(self, prompt: str, *, system: str | None = None, timeout: int = 90) -> str:
        self.calls.append((prompt, system))
        if callable(self._responder):
            return self._responder(prompt, system)
        return self._responder


def resolve_provider(name: str = "auto", model: str | None = None) -> LLMProvider:
    """Build a provider by name. `auto` prefers the user's own subscription.

    Raises `LLMUnavailable` when the named provider (or, for `auto`, every
    provider) cannot be used, which the caller treats as "use the template".
    """
    if name not in PROVIDER_NAMES:
        raise ValueError(f"provider must be one of {', '.join(PROVIDER_NAMES)}, got {name!r}")
    if name == "mock":
        from ..debate.demo import scripted  # noqa: PLC0415

        return MockProvider(scripted())
    if name in ("auto", "claude-cli"):
        cli = ClaudeCLIProvider(model=model or CLI_DEFAULT_MODEL)
        if cli.available():
            return cli
        if name == "claude-cli":
            raise LLMUnavailable(
                "Claude Code is not installed or not logged in. Install it and run "
                "`claude auth login`, or choose another --provider"
            )
    if name in ("auto", "api"):
        from .llm_report import claude_available  # noqa: PLC0415

        if claude_available():
            return APIProvider(model=model or DEFAULT_MODEL)
        if name == "api":
            raise LLMUnavailable("no ANTHROPIC_API_KEY is configured")
    from . import local_llm  # noqa: PLC0415

    if local_llm.available():
        return LocalProvider(model)
    raise LLMUnavailable(
        "no model is available: Claude Code is not logged in, no API key is set, "
        "and no local model is downloaded"
    )
