"""Report writing on a model that runs on this machine.

The same job as `llm_report`'s Claude path, with the same prompt and the same
validator afterwards, but the weights sit on the user's disk and nothing
leaves it. That matters here for three reasons beyond privacy: the tool's
standing promise is that it works with no configuration, and an API key is
configuration; a local model costs nothing per report, so there is no reason
to ration them; and the evidence bundle is the user's own research, which
they have not agreed to send anywhere.

**Why a small model is enough.** Normally the answer to "can a 4B model do
this" is "not reliably". Here the system does not have to trust the model at
all: `validator.validate_report` checks afterwards that every figure in the
prose traces back to a bundle field, rejects the draft if it does not, and
regenerates once with the failures quoted back. The model's job is to rewrite
supplied numbers into plain English in a fixed section order -- faithful
rewriting, not reasoning -- and a 4B instruction-tuned model does that well.
The guardrail is mechanical, so the model can be modest.

**MLX rather than llama.cpp or PyTorch.** MLX is Apple's own array framework
and runs on the unified memory directly, which on an M-series machine is
both the fastest option and the one with no CUDA-shaped dependency tree.
PyTorch is deliberately excluded from the application bundle -- it would add
hundreds of megabytes to a few-megabyte app -- so a torch-based runtime was
never available here.

**The weights are not bundled.** Four gigabytes cannot ship inside an
application, so they are fetched once, on request, with the user told what is
being downloaded and how large it is. Until then this backend reports itself
unavailable and the caller falls back exactly as it does without an API key.

Everything is imported lazily. Someone who never turns this on never pays for
it, and someone on a machine without Apple Silicon gets a clear answer rather
than an import error.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "LocalUnavailable",
    "available",
    "describe",
    "ensure_weights",
    "generate",
    "weights_present",
]

#: Qwen3.5 4B, quantised to four bits for MLX. Chosen for the licence as much
#: as the size: Apache 2.0 carries no restriction on use or redistribution,
#: which a packaged application needs and which the Gemma and Llama licences
#: do not give. Four gigabytes leaves room to work on a 16GB machine, and it
#: runs several times faster than the 12B alternatives -- which matters when
#: somebody is watching a spinner wait for a report they already have the
#: template version of.
DEFAULT_LOCAL_MODEL = "mlx-community/Qwen3.5-4B-OptiQ-4bit"

#: Roughly what the download costs, for the message shown before it starts.
APPROXIMATE_SIZE_GB = 4.0

#: Report writing is rewriting, not invention. Sampling at zero keeps the
#: model on the figures it was given, and makes a failed validation
#: reproducible instead of a coin flip.
TEMPERATURE = 0.0

MAX_TOKENS = 2048

#: Qwen emits its reasoning inside these before the answer. The prompt asks
#: for it to be off, but a template that ignores the flag would otherwise
#: leak the scratchpad into the report.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

#: Loading costs several seconds and a few gigabytes, so it happens once per
#: process rather than once per report.
_loaded: dict[str, Any] = {}


class LocalUnavailable(RuntimeError):
    """No local model can be run, with a reason the user can act on."""


@dataclass(frozen=True)
class LocalStatus:
    """Everything the interface needs to explain the state of this backend."""

    supported: bool       # this machine could run it at all
    installed: bool       # mlx-lm is importable
    downloaded: bool      # the weights are on disk
    model: str
    detail: str

    @property
    def ready(self) -> bool:
        return self.supported and self.installed and self.downloaded


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_installed() -> bool:
    try:
        import mlx_lm  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def weights_present(model: str = DEFAULT_LOCAL_MODEL) -> bool:
    """Whether the weights are already cached, without touching the network.

    Checked by asking the hub cache rather than by calling the loader: the
    loader would happily start a four gigabyte download, and a question
    about availability must never do that.
    """
    try:
        from huggingface_hub import scan_cache_dir  # noqa: PLC0415
    except ImportError:
        return False
    try:
        cache = scan_cache_dir()
    except Exception:  # noqa: BLE001 - a missing or unreadable cache is just "no"
        return False
    for repo in cache.repos:
        if repo.repo_id == model and repo.revisions:
            return True
    return False


def describe(model: str = DEFAULT_LOCAL_MODEL) -> LocalStatus:
    """The state of this backend, in terms that can be shown to a user."""
    if not _apple_silicon():
        return LocalStatus(
            False, False, False, model,
            "A local model needs an Apple Silicon Mac; this machine is "
            f"{platform.system()} {platform.machine()}.",
        )
    if not _mlx_installed():
        return LocalStatus(
            True, False, False, model,
            "Not installed yet. Run: pip install mlx-lm",
        )
    if not weights_present(model):
        return LocalStatus(
            True, True, False, model,
            f"Ready to download {model} (about {APPROXIMATE_SIZE_GB:.0f}GB, once).",
        )
    return LocalStatus(True, True, True, model, f"Ready, running {model} on this machine.")


def available(model: str = DEFAULT_LOCAL_MODEL) -> bool:
    """Whether a report could be written right now with no download."""
    return describe(model).ready


def ensure_weights(model: str = DEFAULT_LOCAL_MODEL) -> str:
    """Fetch the weights if they are not already here. Returns the local path.

    Separate from `generate` on purpose: downloading four gigabytes is a
    thing a user consents to, not a side effect of asking a question.
    """
    status = describe(model)
    if not status.supported or not status.installed:
        raise LocalUnavailable(status.detail)
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from huggingface_hub.utils import enable_progress_bars  # noqa: PLC0415

        enable_progress_bars()  # a four gigabyte download should be visible
        return snapshot_download(model)
    except Exception as exc:  # noqa: BLE001 - network, disk, auth: all the same to the caller
        raise LocalUnavailable(f"Could not download {model}: {exc}") from exc


def _load(model: str) -> tuple[Any, Any]:
    if model in _loaded:
        return _loaded[model]
    status = describe(model)
    if not status.supported or not status.installed:
        raise LocalUnavailable(status.detail)
    if not status.downloaded:
        raise LocalUnavailable(
            f"{model} has not been downloaded yet ({APPROXIMATE_SIZE_GB:.0f}GB). "
            "Download it first, then ask again."
        )
    from mlx_lm import load  # noqa: PLC0415

    try:
        # The hub prints a "Fetching N files" progress bar even when every
        # file is already cached, which on a CLI looks like a download
        # starting. Downloads have their own reporting in `ensure_weights`.
        try:
            from huggingface_hub.utils import disable_progress_bars  # noqa: PLC0415

            disable_progress_bars()
        except ImportError:
            pass
        _loaded[model] = load(model)
    except Exception as exc:  # noqa: BLE001
        raise LocalUnavailable(f"Could not load {model}: {exc}") from exc
    return _loaded[model]


def _render_prompt(tokenizer: Any, system: str, messages: list[dict[str, Any]]) -> str:
    """Turn the Claude-shaped messages into this model's own prompt format.

    The Claude call carries the system prompt in its own parameter; a chat
    template wants it as the first turn. Thinking is switched off where the
    template understands the flag: the reasoning would be several times
    longer than the report, and the report is a rewrite of numbers that have
    already been computed.
    """
    turns = [{"role": "system", "content": system}]
    for message in messages:
        content = message["content"]
        if isinstance(content, list):  # Claude's block form
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        turns.append({"role": message["role"], "content": content})

    try:
        return tokenizer.apply_chat_template(
            turns, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        # Older templates do not accept the flag; the block is stripped after
        # generation instead.
        return tokenizer.apply_chat_template(
            turns, add_generation_prompt=True, tokenize=False
        )


def generate(
    messages: list[dict[str, Any]],
    system: str,
    model: str = DEFAULT_LOCAL_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, int, int]:
    """One generation. Returns (text, prompt tokens, completion tokens).

    The token counts are measured by encoding the prompt and the completion,
    since nothing is being billed and the figures exist only so the caller
    can report what the work cost.
    """
    handle, tokenizer = _load(model)
    prompt = _render_prompt(tokenizer, system, messages)

    from mlx_lm import generate as mlx_generate  # noqa: PLC0415
    from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

    try:
        text = mlx_generate(
            handle,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=TEMPERATURE),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise LocalUnavailable(f"{model} failed while generating: {exc}") from exc

    cleaned = _THINK_BLOCK.sub("", text).strip()
    return cleaned, len(tokenizer.encode(prompt)), len(tokenizer.encode(text))
