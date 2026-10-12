"""Scripted demo agents: canned answers built by reading the prompt.

Used by `--provider mock`, so the debate layout can be seen offline without
spending any model calls, and by the tests, whose hooks make one kind of
answer misbehave on purpose. The answers cite fact ids that really appear
in the prompt, so they pass the checks; the CLI labels the result as demo
output, because no model wrote it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

_FACT = re.compile(r"^([A-Z]\d+): ", re.MULTILINE)
_PERSONA = re.compile(r"You are the ([A-Za-z\- ]+?)\.")

STANCES = {"Value reader": "bear", "Momentum reader": "bull",
           "Big-picture reader": "neutral", "Skeptic": "bear"}


def fact_ids(prompt: str) -> list[str]:
    return _FACT.findall(prompt)


def scripted(
    claims_for: Callable[[str, list[str]], list[dict]] | None = None,
    judge_scores: dict[str, float] | None = None,
    opening_override: Callable[[str], str | None] | None = None,
) -> Callable[[str, str | None], str]:
    """Build a responder. Hooks let a test change one kind of answer."""

    def default_claims(persona: str, ids: list[str]) -> list[dict]:
        own = [i for i in ids if not i.startswith("S")] or ids
        scen = [i for i in ids if i.startswith("S")]
        return [
            {"text": f"The {persona} sees the figures in {own[0]} as the main point.",
             "evidence_ids": [own[0]]},
            {"text": "The scenario facts show a wide range of past outcomes.",
             "evidence_ids": scen[:2] or [own[0]]},
        ]

    def respond(prompt: str, system: str | None) -> str:
        persona_match = _PERSONA.search(prompt)
        persona = persona_match.group(1) if persona_match else "agent"
        ids = fact_ids(prompt)
        if "Make your opening case" in prompt:
            if opening_override is not None:
                override = opening_override(persona)
                if override is not None:
                    return override
            return json.dumps({
                "stance": STANCES.get(persona, "neutral"),
                "headline": f"The {persona} reads the evidence its own way.",
                "claims": (claims_for or default_claims)(persona, ids),
                "falsifiers": ["A long run of results that point the other way."],
                "confidence_words": "moderate",
                "proposed_scenarios": [{"name": "strait closure", "proxy": "oil"}],
            })
        if "Challenge that one claim" in prompt:
            target = re.search(r"id (\w+\.c\d+)", prompt)
            return json.dumps({"target_claim_id": target.group(1) if target else "",
                               "challenge": "That claim leaves out the worst past outcomes.",
                               "evidence_ids": ids[:1]})
        if "Answer the challenge" in prompt:
            return json.dumps({"rebuttal": "The worst outcomes are real and were counted.",
                               "evidence_ids": ids[:1], "concede": False})
        if "You judge a debate" in prompt:
            cases = re.findall(r"^CASE (\w+)", prompt, re.MULTILINE)
            scores = {c: (judge_scores or {}).get(c, 0.5) for c in cases}
            return json.dumps({"scores": scores,
                               "summaries": {c: f"This is the {c} view in plain words." for c in cases}})
        return "{}"

    return respond
