"""Tests for the LLM synthesis path, driven end to end with a fake client.

No network. A fake Anthropic client returns scripted text, which lets the
whole loop -- generate, validate, quote the failures back, regenerate, give
up -- run deterministically. That loop is the part worth testing: the prompt
is a request, and the validator plus this retry logic are the guarantee.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from research_cli.evidence.why_moved import build_why_moved_bundle
from research_cli.query_parser import parse_query
from research_cli.synthesis.llm_report import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    LLMUnavailable,
    build_messages,
    generate_report,
)
from research_cli.synthesis.template_report import REQUIRED_SECTIONS, render_report

TODAY = date(2026, 9, 21)


class FakeMessages:
    """Records every request and replays a scripted list of responses."""

    def __init__(self, replies: list[str], error: Exception | None = None) -> None:
        self._replies = list(replies)
        self._error = error
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        text = self._replies.pop(0) if self._replies else ""
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
        )


class FakeClient:
    def __init__(self, replies: list[str], error: Exception | None = None) -> None:
        self.messages = FakeMessages(replies, error)


def _beginner_prose(bundle) -> str:
    """A plausible, fully supported report -- what a good generation looks like.

    Built from bundle fields only, so it passes the validator for the same
    reason a real generation would.
    """
    observation = bundle.observation
    external = bundle.external_factors()
    lines = [
        "QUESTION",
        (
            f'You asked: "{bundle.query}". This covers {bundle.ticker} on '
            f"{bundle.period_description}."
        ),
        "",
        "SUMMARY",
        observation.plain_summary,
        (
            "A standard deviation is just a way of saying how big a normal day is "
            "for this investment."
        ),
        "",
        "KEY EVIDENCE",
    ]
    for factor in external:
        lines.append(f"- {factor.label}: {factor.what_happened}.")
        if factor.historical_hit_rate is not None:
            lines.append(
                f"  On similar days in the past, it moved the same way as "
                f"{bundle.ticker} {factor.historical_hit_rate:.0%} of the time. On all "
                f"days the figure was {factor.base_rate:.0%}, so the difference is "
                + ("worth noting." if factor.beats_base_rate else "inside the margin of error.")
            )
    lines += ["", "COMPETING EXPLANATIONS"]
    for item in bundle.counterevidence:
        lines.append(f"- {item.label}: {item.detail}")
    if not bundle.counterevidence:
        lines.append("- Nothing in the data collected argues against the evidence above.")
    lines += [
        "",
        "CONFIDENCE & SAMPLE SIZE",
        f"The figures come from {bundle.history_years:.0f} years of daily history.",
        "",
        "RAW DATA",
        f"Close: {observation.close:.2f}",
        f"Move: {observation.period_return_pct:+.2f}%",
        f"Typical daily swing: {observation.daily_vol_pct:.2f}%",
        "",
        "This is evidence, not investment advice.",
    ]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def bundle():
    return build_why_moved_bundle(
        parse_query("why did SPY fall today", today=TODAY), source="mock"
    )


# --- the prompt ------------------------------------------------------------

def test_system_prompt_forbids_causal_language() -> None:
    for word in ("caused", "because", "due to", "drove", "triggered"):
        assert word in SYSTEM_PROMPT


def test_system_prompt_forbids_advice() -> None:
    for word in ("buy", "sell", "undervalued", "price target"):
        assert word in SYSTEM_PROMPT.lower()


def test_request_names_the_sections_for_the_question_type(bundle) -> None:
    """Attribution and snapshot reports have different sections.

    The list therefore lives in the per-request instruction rather than the
    shared system prompt, so the cached prefix stays identical across both.
    """
    from research_cli.synthesis.snapshot_report import SNAPSHOT_SECTIONS

    attribution = build_messages(bundle)[0]["content"]
    for section in REQUIRED_SECTIONS:
        assert section in attribution

    snapshot_bundle = bundle.model_copy(update={"question_type": "snapshot"})
    snapshot = build_messages(snapshot_bundle)[0]["content"]
    for section in SNAPSHOT_SECTIONS:
        assert section in snapshot


def test_system_prompt_hardcodes_no_type_specific_sections() -> None:
    """It is sent with cache_control, so it must be identical every request.

    Baking one question type's headings into it would either break the other
    type or force two different cached prefixes.
    """
    from research_cli.synthesis.snapshot_report import SNAPSHOT_SECTIONS

    type_specific = set(REQUIRED_SECTIONS) ^ set(SNAPSHOT_SECTIONS)
    for heading in type_specific:
        assert heading not in SYSTEM_PROMPT, heading


def test_prompt_hands_over_a_pre_sorted_factor_list(bundle) -> None:
    """Ordering is not a number, so the validator cannot check it afterwards.

    The only reliable enforcement is to leave the model nothing to decide.
    """
    import json

    payload = json.loads(
        build_messages(bundle)[0]["content"].split("EVIDENCE BUNDLE:\n", 1)[1]
    )
    ranked = [f["factor_id"] for f in payload["factors_ranked_external"]]
    assert ranked == [f.factor_id for f in bundle.external_factors()]
    assert "factors" not in payload, "the unsorted list must not also be present"


def test_prompt_separates_mechanical_factors(bundle) -> None:
    import json

    payload = json.loads(
        build_messages(bundle)[0]["content"].split("EVIDENCE BUNDLE:\n", 1)[1]
    )
    assert {f["factor_id"] for f in payload["factors_mechanical"]} == {
        f.factor_id for f in bundle.mechanical_factors()
    }


def test_repair_prompt_quotes_the_specific_failures(bundle) -> None:
    messages = build_messages(bundle, ["[error] unsupported_number: 7.83 does not trace"])
    assert "7.83" in messages[0]["content"]
    assert "rejected by an automatic check" in messages[0]["content"]


# --- the generation loop ---------------------------------------------------

def test_valid_output_is_accepted_on_the_first_attempt(bundle) -> None:
    client = FakeClient([_beginner_prose(bundle)])
    result = generate_report(bundle, client=client)
    assert result.usable
    assert result.attempts == 1
    assert result.validation.ok
    assert result.input_tokens == 1000


def test_request_uses_the_documented_api_shape(bundle) -> None:
    client = FakeClient([_beginner_prose(bundle)])
    generate_report(bundle, client=client)
    request = client.messages.requests[0]
    assert request["model"] == DEFAULT_MODEL
    assert request["thinking"] == {"type": "adaptive"}
    assert "effort" in request["output_config"]
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "budget_tokens" not in str(request), "removed on Opus 5; sending it is a 400"


def test_a_hallucinated_number_triggers_one_repair(bundle) -> None:
    """The whole point of the loop."""
    bad = _beginner_prose(bundle) + "\n  The index fell 7.83% on record volume."
    client = FakeClient([bad, _beginner_prose(bundle)])
    result = generate_report(bundle, client=client)

    assert result.usable
    assert result.attempts == 2
    assert any("7.83" in note for note in result.repair_notes)
    assert "7.83" in client.messages.requests[1]["messages"][0]["content"]


def test_persistent_failure_is_reported_not_shown(bundle) -> None:
    """Two strikes and the caller keeps the template. Never show unvalidated prose."""
    bad = _beginner_prose(bundle) + "\n  It fell because the Fed raised rates."
    client = FakeClient([bad, bad])
    result = generate_report(bundle, client=client)

    assert not result.usable
    assert result.attempts == 2
    assert any(i.kind == "causal_language" for i in result.validation.errors)


def test_dropped_counterevidence_is_caught_and_repaired(bundle) -> None:
    if not bundle.counterevidence:
        pytest.skip("this bundle has no counterevidence")
    stripped = _beginner_prose(bundle)
    for item in bundle.counterevidence:
        stripped = stripped.replace(item.label, "something else")
    client = FakeClient([stripped, _beginner_prose(bundle)])
    result = generate_report(bundle, client=client)
    assert result.usable
    assert result.attempts == 2


def test_advice_language_is_caught(bundle) -> None:
    bad = _beginner_prose(bundle) + "\n  SPY looks undervalued at these levels."
    client = FakeClient([bad, bad])
    result = generate_report(bundle, client=client)
    assert not result.usable
    assert any(i.kind == "verdict_language" for i in result.validation.errors)


def test_empty_response_does_not_crash(bundle) -> None:
    client = FakeClient(["", ""])
    result = generate_report(bundle, client=client)
    assert not result.usable


def _status_error(cls, status: int, message: str):
    """Build a real SDK status error; they require genuine request objects."""
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls(message, response=response, body=None)


def test_api_errors_become_llm_unavailable(bundle) -> None:
    """Every failure mode must route to the template, not to a traceback."""
    import anthropic
    import httpx2

    errors = [
        anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
        _status_error(anthropic.AuthenticationError, 401, "bad key"),
        _status_error(anthropic.NotFoundError, 404, "no such model"),
        _status_error(anthropic.RateLimitError, 429, "slow down"),
        _status_error(anthropic.InternalServerError, 500, "server error"),
    ]
    for error in errors:
        with pytest.raises(LLMUnavailable):
            generate_report(bundle, client=FakeClient([], error=error))


def test_refusal_is_handled(bundle) -> None:
    class Refusing(FakeMessages):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            return SimpleNamespace(
                content=[], stop_reason="refusal",
                usage=SimpleNamespace(input_tokens=10, output_tokens=0),
            )

    client = FakeClient([])
    client.messages = Refusing([])
    with pytest.raises(LLMUnavailable, match="declined"):
        generate_report(bundle, client=client)


def test_max_attempts_is_validated(bundle) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        generate_report(bundle, client=FakeClient([]), max_attempts=0)


# --- template vs LLM -------------------------------------------------------

def test_both_paths_carry_the_same_sections_and_evidence(bundle) -> None:
    """The LLM changes the wording, never the evidence or the structure."""
    template = render_report(bundle)
    llm = _beginner_prose(bundle)
    for section in REQUIRED_SECTIONS:
        assert section in template
        assert section in llm

    from research_cli.synthesis.validator import validate_report

    assert validate_report(template, bundle).ok
    assert validate_report(llm, bundle).ok
