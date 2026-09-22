"""Turn an EvidenceBundle into beginner-friendly prose with Claude.

The template in `template_report.py` is correct and unreadable. It says
"sigma_multiple +2.65" where a beginner needs "this was a much bigger move
than SPY usually has in a day". This module keeps the same evidence and the
same section structure, and replaces the wording.

**The model receives the bundle and nothing else.** No market knowledge it
might have, no headlines, no prices from its training data -- the prompt
says so explicitly, and `validator.validate_report` afterwards checks that
every number in the prose traces back to a bundle field. That check is not
advisory: a report that fails it is regenerated with the failures quoted
back, and if the second attempt also fails the caller is told, keeps the
template output, and the LLM text is discarded.

**Four rules the prompt enforces and the validator verifies:**

1. Only bundle fields may be referenced.
2. Factors are ranked by `rank_score` (hit rate x confidence), which the
   bundle has already computed -- never by which one makes a better story.
3. No causal language. "Caused", "because", "due to", "drove", "triggered"
   are forbidden; "coincided with" and "associated with" are the vocabulary.
4. Counterevidence must be surfaced when the bundle holds any.

Rules 3 and 4 are checked mechanically afterwards, so the prompt is a
request and the validator is the guarantee. Rule 1 is checked numerically.
Rule 2 is the one the validator cannot verify -- ordering is not a number --
so it is stated twice in the prompt and the ranked list is handed over
pre-sorted, leaving the model nothing to decide.

Without an API key this module raises `LLMUnavailable` and the caller falls
back to the template. That is a normal path, not an error: the tool's
promise is that it works with no configuration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ..evidence.schema import EvidenceBundle
from ..settings import get_key
from .snapshot_report import SNAPSHOT_SECTIONS
from .template_report import REQUIRED_SECTIONS
from .validator import ValidationReport, validate_report

__all__ = [
    "DEFAULT_MODEL",
    "SYSTEM_PROMPT",
    "LLMReportResult",
    "LLMUnavailable",
    "build_messages",
    "claude_available",
    "generate_report",
    "llm_available",
    "resolve_backend",
]

#: Claude Opus 5. Report writing under hard constraints is a reasoning task,
#: not a formatting one -- the model has to decide what a beginner needs told
#: about a statistic while touching none of the numbers.
DEFAULT_MODEL = "claude-opus-5"

#: Effort is set to medium rather than the default high: this is a CLI a
#: person is waiting on, the evidence is already computed, and the remaining
#: work is exposition. Raise it if reports come back shallow.
DEFAULT_EFFORT = "medium"

MAX_TOKENS = 4096
MAX_ATTEMPTS = 2


class LLMUnavailable(RuntimeError):
    """Raised when no model can be reached. The caller falls back, not fails."""


SYSTEM_PROMPT = """\
You write short research notes for people who are new to investing. Your \
reader has maybe bought a few shares and does not know what volatility, a \
standard deviation, or a base rate is. Explain those ideas in passing, in \
plain words, without lecturing.

You will be given an EVIDENCE BUNDLE as JSON. It is your only source.

ABSOLUTE RULES

1. USE ONLY THE BUNDLE. Every number, date, ticker and fact in your note must \
come from the bundle. You may round a number and you may convert a rate like \
0.68 into 68%. You may not introduce any other figure -- not from memory, not \
from general knowledge about markets, not estimated, not "roughly". If you \
want to say something you cannot support from the bundle, leave it out. \
Do not mention news events, company announcements, central bank decisions or \
anything else that is not in the bundle. You do not know what happened in \
the news; the bundle is all you have.

2. NEVER CLAIM A CAUSE. These are things that happened at the same time. You \
may write "coincided with", "was associated with", "happened alongside", \
"one thing that moved at the same time". You may NOT write "caused", \
"because", "due to", "drove", "driven by", "triggered", "led to", \
"resulted in", "thanks to", or "the reason for". This is not a style \
preference: the data cannot support a causal claim, and a beginner will \
believe one if you make it.

3. RANK BY THE EVIDENCE, NOT THE STORY. The bundle's factors arrive already \
sorted by rank_score, which is the historical hit rate weighted by how much \
data supports it. Keep that order. Never promote a factor because it makes a \
more interesting narrative. If a factor's "beats_base_rate" is false, say \
plainly that its track record is no better than chance -- do not dress it up.

4. ALWAYS SURFACE THE COUNTEREVIDENCE. If the bundle has a counterevidence \
list, every item must appear in your note, under whichever of the given \
headings covers what argues against the evidence. Do not soften them.

5. NEVER GIVE ADVICE. No buy, sell, hold, recommend, "worth buying", \
"undervalued", "overvalued", or price targets. You are describing evidence.

WHAT TO EXPLAIN

- If observation.is_unusual is false, say clearly and early that the move was \
  ordinary and probably needs no explanation at all. This matters more than \
  anything else in the note.
- When you first use a hit rate, explain what it means: of the past days when \
  this factor looked like it does now, the share on which the symbol moved \
  this way.
- Always give the base rate next to the hit rate, and explain that the base \
  rate is what you would expect knowing nothing.
- Factors whose "relationship" is "mechanical" are parts of the symbol \
  itself. Their high hit rates are arithmetic, not evidence. Say so.

FORMAT

Plain text. Use exactly the section headings you are given below, in that \
order, each on its own line in capitals.

Under the final RAW DATA heading, list the key figures as label/value lines. \
Keep the whole note under 700 words. No markdown, no bullets with asterisks \
-- use "- " for lists. Write in short paragraphs a nervous beginner can \
actually finish.

IF THE QUESTION IS ABOUT WHETHER TO INVEST

Never answer it. Lay out the evidence and close by naming, from these \
specific figures, what is still unresolved -- and say plainly that the answer \
depends on the reader's own horizon, holdings and tolerance for a fall, none \
of which appear in the bundle. Do not end on a generic disclaimer.
"""


@dataclass
class LLMReportResult:
    """A generated report plus everything needed to judge whether to use it.

    Attributes
    ----------
    report : the prose. Empty when every attempt failed validation.
    validation : the validator's verdict on `report`.
    attempts : how many generations were needed.
    model : the model id used.
    usable : whether the caller should show this instead of the template.
    repair_notes : the validation failures fed back on a retry, kept so a
        recurring failure mode is visible rather than silently retried away.
    input_tokens, output_tokens : usage, summed over attempts.
    """

    report: str
    validation: ValidationReport | None
    attempts: int
    model: str
    usable: bool
    repair_notes: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def resolve_backend(preference: str = "auto") -> str:
    """Pick which model writes the report.

    Local first when it is ready, for the reason the whole tool is built
    around: it needs no key, costs nothing per report, and keeps the bundle
    on the machine that produced it. Claude is used when it is configured
    and no local model is, and is better at this job -- so an explicit
    preference always wins over the default order.

    Raises `LLMUnavailable` when neither can be reached, which the caller
    treats as "use the template", not as an error.
    """
    if preference not in ("auto", "local", "claude"):
        raise ValueError(f"backend must be 'auto', 'local' or 'claude', got {preference!r}")

    if preference == "local":
        from . import local_llm  # noqa: PLC0415

        if not local_llm.available():
            raise LLMUnavailable(local_llm.describe().detail)
        return "local"
    if preference == "claude":
        if not claude_available():
            raise LLMUnavailable("no Claude credentials are configured")
        return "claude"

    try:
        from . import local_llm  # noqa: PLC0415

        if local_llm.available():
            return "local"
    except ImportError:  # pragma: no cover - the module is part of the package
        pass
    if claude_available():
        return "claude"
    raise LLMUnavailable(
        "no model is available: no Claude credentials, and no local model "
        "downloaded"
    )


def llm_available(preference: str = "auto") -> bool:
    """Whether any model could write a report right now."""
    try:
        resolve_backend(preference)
    except LLMUnavailable:
        return False
    return True


def claude_available() -> bool:
    """Whether a Claude call could plausibly be made.

    An unset ANTHROPIC_API_KEY does not by itself mean there are no
    credentials -- the SDK also resolves ANTHROPIC_AUTH_TOKEN and an
    `ant auth login` profile -- so this reports whether the SDK is installed
    and some credential source is configured, and lets the call itself be
    the real test.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if get_key("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # An `ant auth login` profile lives on disk and is picked up by a
    # zero-argument client.
    from pathlib import Path

    return (Path.home() / ".config" / "anthropic").exists()


def _bundle_payload(bundle: EvidenceBundle) -> str:
    """The bundle as JSON, pre-sorted so the model has no ordering to decide.

    Factors are emitted in `rank_score` order and split into external and
    mechanical groups. Handing the model a pre-sorted list is the only
    reliable way to enforce rule 3 -- ordering is not a number, so the
    validator cannot check it after the fact.
    """
    payload: dict[str, Any] = json.loads(bundle.model_dump_json())
    payload["factors_ranked_external"] = [
        json.loads(f.model_dump_json()) for f in bundle.external_factors()
    ]
    payload["factors_mechanical"] = [
        json.loads(f.model_dump_json()) for f in bundle.mechanical_factors()
    ]
    payload.pop("factors", None)
    return json.dumps(payload, indent=2, default=str)


#: Restated at the very end of the instruction for models that need it.
#:
#: A small model holds the beginning of a long prompt less firmly than the
#: end of it, and the failure this fixes was exactly that: a 4B model that
#: had traced all 86 of its figures correctly still wrote "Because the
#: sample size is relatively small", having forgotten a rule stated two
#: thousand words earlier. The banned words are spelled out again, and the
#: statistical sense is closed off explicitly, because that is the sense a
#: model reaches for once it has stopped talking about the market.
FINAL_CHECKLIST = """\

BEFORE YOU FINISH, CHECK THE NOTE YOU HAVE WRITTEN:

- These words must not appear anywhere in it: because, caused, due to, \
drove, driven by, triggered, led to, resulted in, thanks to, the reason for. \
Not about the market, and not about statistics or sample sizes either. Start \
a new sentence instead.
- Every number in your note appears in the bundle above.
- You wrote sentences, not a list of fields. Never copy a bundle field name \
such as high_52w, annualized_vol_pct or beats_base_rate into the note; say \
what it means in words. A field name carries digits of its own and they are \
not figures you were given.
- Every item in the bundle's counterevidence list is mentioned, in the \
section that says what the numbers do not tell you. Leaving one out is the \
one omission this note is never allowed to make.
- Long numbers are rounded the way a person would say them: $1.5 trillion, \
not 1508144331448.53; 733 rather than 733.1641514191865.
- You never gave a figure for a named peer company. The bundle holds this \
company's own value and the median across its peers, and nothing else. \
Writing "Microsoft trades at 27.9" states a figure you were not given, even \
if you believe it. Compare against "the typical company in this group".
- You used every section heading given, in the order given.
"""


def build_messages(
    bundle: EvidenceBundle,
    repair_notes: list[str] | None = None,
    emphasise_rules: bool = False,
) -> list[dict[str, Any]]:
    """Build the message list for one generation attempt.

    Pure and testable: the prompt can be inspected in a unit test without a
    network call, which is where prompt regressions are cheapest to catch.

    On a repair attempt the previous failures are quoted back verbatim. They
    are stated as facts about the previous draft rather than as
    instructions, so the model corrects the specific figures rather than
    rewriting the whole note.

    `emphasise_rules` appends `FINAL_CHECKLIST`, and is set for the local
    backend. It is not sent to Claude, which does not need it -- and a prompt
    that changes for every model is a prompt nobody can reason about, so the
    difference is one appended block rather than a second prompt.
    """
    sections = (
        SNAPSHOT_SECTIONS if bundle.question_type == "snapshot" else REQUIRED_SECTIONS
    )
    instruction = (
        "Write the research note for this evidence bundle.\n\n"
        "Use exactly these section headings, in this order:\n"
        + "\n".join(sections)
        + f"\n\nEVIDENCE BUNDLE:\n{_bundle_payload(bundle)}"
    )
    if repair_notes:
        joined = "\n".join(f"- {note}" for note in repair_notes)
        instruction += (
            "\n\nYour previous draft was rejected by an automatic check. These "
            f"problems were found:\n{joined}\n\n"
            "Write the note again. Every number must appear in the bundle above. "
            "If you cannot support a statement from the bundle, remove it."
        )
    if emphasise_rules:
        instruction += FINAL_CHECKLIST
    return [{"role": "user", "content": instruction}]


def _call_claude(
    messages: list[dict[str, Any]], model: str, effort: str, client: Any | None
) -> tuple[str, int, int]:
    """One Claude call. Returns (text, input_tokens, output_tokens)."""
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailable(
            "the anthropic package is not installed; "
            "pip install -r research_cli/requirements.txt"
        ) from exc

    if client is not None:
        active = client
    else:
        # The key may live in the config file rather than the environment,
        # which is how a double-clicked app gets credentials at all.
        stored = get_key("ANTHROPIC_API_KEY")
        active = anthropic.Anthropic(api_key=stored) if stored else anthropic.Anthropic()
    try:
        response = active.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            # The system prompt is long, identical on every call and sits
            # ahead of the bundle, so caching it is close to free and saves
            # most of the input cost on the second and later queries.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=messages,
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable(f"Claude rejected the credentials: {exc}") from exc
    except anthropic.NotFoundError as exc:
        raise LLMUnavailable(f"model {model!r} is not available: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable(f"rate limited by Claude: {exc}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable(f"could not reach Claude: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"Claude returned an error ({exc.status_code}): {exc}") from exc

    if getattr(response, "stop_reason", None) == "refusal":
        raise LLMUnavailable("Claude declined to write this report")

    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    usage = getattr(response, "usage", None)
    return (
        text.strip(),
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _call_local(
    messages: list[dict[str, Any]], model: str
) -> tuple[str, int, int]:
    """One local generation, with the same signature as the Claude call."""
    from . import local_llm  # noqa: PLC0415

    try:
        return local_llm.generate(messages, SYSTEM_PROMPT, model=model)
    except local_llm.LocalUnavailable as exc:
        raise LLMUnavailable(str(exc)) from exc


def generate_report(
    bundle: EvidenceBundle,
    model: str | None = None,
    effort: str = DEFAULT_EFFORT,
    client: Any | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    backend: str = "auto",
) -> LLMReportResult:
    """Generate a validated report, retrying once if validation fails.

    Parameters
    ----------
    bundle : the evidence. The model sees this and nothing else.
    model : model id, or None for whichever the chosen backend defaults to.
    effort : "low" through "max". Claude only; a local model has no such dial.
    client : an Anthropic client, or None to construct one. Injectable so
        tests can drive the whole loop -- including the repair path -- with a
        fake and no network. Supplying one selects the Claude backend, since
        that is the only thing a client can be.
    backend : "auto", "local" or "claude". Auto prefers a downloaded local
        model over an API key.
    max_attempts : generations before giving up. Two is the useful number:
        one retry fixes a stray figure, and a model that fails twice with
        the failures quoted back is not going to succeed on a third try.

    Returns an `LLMReportResult` whose `usable` flag is False if no attempt
    validated. The caller keeps the template output in that case; it never
    shows unvalidated prose.

    Raises `LLMUnavailable` when no model can be reached at all.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    # A supplied client can only be an Anthropic one, so it names the backend
    # and keeps every existing caller and test on the path it expects.
    chosen = "claude" if client is not None else resolve_backend(backend)
    if model is None:
        if chosen == "local":
            from . import local_llm  # noqa: PLC0415

            model = local_llm.DEFAULT_LOCAL_MODEL
        else:
            model = DEFAULT_MODEL

    repair_notes: list[str] = []
    last_report, last_validation = "", None
    input_tokens = output_tokens = 0

    for attempt in range(1, max_attempts + 1):
        messages = build_messages(
            bundle,
            repair_notes if attempt > 1 else None,
            emphasise_rules=chosen == "local",
        )
        if chosen == "local":
            text, used_in, used_out = _call_local(messages, model)
        else:
            text, used_in, used_out = _call_claude(messages, model, effort, client)
        input_tokens += used_in
        output_tokens += used_out
        validation = validate_report(
            text,
            bundle,
            required_sections=(
                SNAPSHOT_SECTIONS
                if bundle.question_type == "snapshot"
                else REQUIRED_SECTIONS
            ),
        )
        last_report, last_validation = text, validation

        if validation.ok:
            return LLMReportResult(
                report=text, validation=validation, attempts=attempt, model=model,
                usable=True, repair_notes=repair_notes,
                input_tokens=input_tokens, output_tokens=output_tokens,
            )
        repair_notes = [str(issue) for issue in validation.errors]

    return LLMReportResult(
        report=last_report, validation=last_validation, attempts=max_attempts,
        model=model, usable=False, repair_notes=repair_notes,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
