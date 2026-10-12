"""Parse and police agent output. Nothing an agent says is used unchecked.

Each parser returns `(result, problems)`. `problems` is what gets quoted back
on the single repair attempt. `result` is built from the parts that passed,
so after the repair a partly-good answer keeps its good claims and drops the
rest -- and a case with no valid claim left is no case at all.

The checks on every sentence:

* every cited id exists in the facts that agent was shown;
* every number traces to the facts that sentence cites (or, for text that
  cites nothing, to the agent's whole slice);
* no causal, verdict or certainty phrasing.
"""

from __future__ import annotations

import json
from typing import Any

from ..scenarios.library import PROXY_SYMBOLS, SCENARIOS
from ..synthesis.validator_stance import allowed_numbers, banned_phrases, trace_numbers
from .models import AgentCase, Challenge, Claim, EvidenceItem, ProposedScenario, Rebuttal

__all__ = [
    "check_sentence",
    "extract_json",
    "map_proposal",
    "parse_challenge",
    "parse_judge",
    "parse_opening",
    "parse_rebuttal",
]

_STANCES = ("bull", "bear", "neutral")
_MAX_CLAIMS = 4
_MAX_TEXT = 400


def extract_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in `text`, tolerating code fences and chatter."""
    start = text.find("{")
    while start != -1:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            return value if isinstance(value, dict) else None
        except ValueError:
            start = text.find("{", start + 1)
    return None


def check_sentence(
    text: Any, cited: list[EvidenceItem], where: str, max_len: int = _MAX_TEXT
) -> list[str]:
    """Problems with one sentence of agent prose, given the facts it cites."""
    if not isinstance(text, str) or not text.strip():
        return [f"{where}: missing text"]
    problems = []
    if len(text) > max_len:
        problems.append(f"{where}: longer than {max_len} characters")
    for kind, label, excerpt in banned_phrases(text):
        problems.append(f"{where}: {kind.replace('_', ' ')} {label!r} in \"{excerpt}\"")
    for number in trace_numbers(text, allowed_numbers(item.text for item in cited)):
        problems.append(f"{where}: number {number} is not in the facts it cites")
    return problems


def _ids(raw: Any, facts: dict[str, EvidenceItem], where: str) -> tuple[tuple[str, ...], list[str]]:
    if not isinstance(raw, list) or not raw:
        return (), [f"{where}: cites no fact ids"]
    ids = tuple(str(i).strip() for i in raw)
    unknown = [i for i in ids if i not in facts]
    if unknown:
        return ids, [f"{where}: cites unknown fact id(s) {', '.join(unknown)}"]
    return ids, []


def map_proposal(name: str, proxy: str, by: str) -> ProposedScenario:
    """Map an agent's extra "what if" onto a measured library scenario, or none."""
    if proxy in PROXY_SYMBOLS or proxy == "self":
        match = next((s for s in SCENARIOS if s.trigger and s.trigger.proxy == proxy), None)
        if match is not None:
            return ProposedScenario(name, by, match.scenario_id)
    lowered = name.lower()
    match = next((s for s in SCENARIOS if s.trigger and any(k in lowered for k in s.keywords)), None)
    return ProposedScenario(name, by, match.scenario_id if match else None)


def parse_opening(
    text: str, persona_id: str, persona_label: str, facts: dict[str, EvidenceItem]
) -> tuple[AgentCase | None, list[str], list[ProposedScenario]]:
    data = extract_json(text)
    if data is None:
        return None, ["the answer was not a JSON object"], []
    problems: list[str] = []
    slice_items = list(facts.values())

    stance = str(data.get("stance", "")).strip().lower()
    if stance not in _STANCES:
        problems.append(f"stance must be one of {', '.join(_STANCES)}")
    headline = data.get("headline")
    headline_problems = check_sentence(headline, slice_items, "headline")
    problems += headline_problems

    claims: list[Claim] = []
    dropped = 0
    raw_claims = data.get("claims") if isinstance(data.get("claims"), list) else []
    if not raw_claims:
        problems.append("no claims were given")
    for index, raw in enumerate(raw_claims[:_MAX_CLAIMS], start=1):
        raw = raw if isinstance(raw, dict) else {}
        where = f"claim {index}"
        ids, id_problems = _ids(raw.get("evidence_ids"), facts, where)
        claim_problems = id_problems or check_sentence(
            raw.get("text"), [facts[i] for i in ids], where
        )
        if claim_problems:
            dropped += 1
            problems += claim_problems
        else:
            claims.append(Claim(f"{persona_id}.c{index}", raw["text"].strip(), ids))

    falsifiers = []
    raw_falsifiers = data.get("falsifiers") if isinstance(data.get("falsifiers"), list) else []
    for index, raw in enumerate(raw_falsifiers[:3], start=1):
        issue = check_sentence(raw, slice_items, f"falsifier {index}")
        problems += issue
        if not issue:
            falsifiers.append(raw.strip())
    if not falsifiers:
        problems.append("state at least one falsifier: what would prove this case wrong")

    proposals = []
    raw_proposals = data.get("proposed_scenarios")
    for raw in (raw_proposals if isinstance(raw_proposals, list) else [])[:2]:
        if isinstance(raw, dict) and isinstance(raw.get("name"), str):
            name = raw["name"].strip()[:60]
            if name and not check_sentence(name, [], "proposal"):
                proposals.append(map_proposal(name, str(raw.get("proxy", "none")), persona_id))

    confidence = str(data.get("confidence_words", "")).strip().lower()
    if stance not in _STANCES or headline_problems or not claims:
        return None, problems, proposals
    case = AgentCase(
        persona=persona_id, persona_label=persona_label, stance=stance,  # type: ignore[arg-type]
        headline=headline.strip(), claims=claims, falsifiers=falsifiers,
        confidence_words=confidence if confidence in ("low", "moderate", "high") else "unstated",
        dropped_claims=dropped,
    )
    return case, problems, proposals


def parse_challenge(
    text: str, by: str, target: AgentCase, facts: dict[str, EvidenceItem]
) -> tuple[Challenge | None, list[str]]:
    data = extract_json(text)
    if data is None:
        return None, ["the answer was not a JSON object"]
    ids, problems = _ids(data.get("evidence_ids"), facts, "challenge")
    problems = problems or check_sentence(data.get("challenge"), [facts[i] for i in ids], "challenge")
    claim_id = str(data.get("target_claim_id", ""))
    if claim_id not in {c.claim_id for c in target.claims}:
        claim_id = target.claims[0].claim_id
    if problems:
        return None, problems
    return Challenge(by, target.persona, claim_id, data["challenge"].strip(), ids), []


def parse_rebuttal(
    text: str, by: str, facts: dict[str, EvidenceItem]
) -> tuple[Rebuttal | None, list[str]]:
    data = extract_json(text)
    if data is None:
        return None, ["the answer was not a JSON object"]
    ids, problems = _ids(data.get("evidence_ids"), facts, "rebuttal")
    problems = problems or check_sentence(data.get("rebuttal"), [facts[i] for i in ids], "rebuttal")
    if problems:
        return None, problems
    return Rebuttal(by, data["rebuttal"].strip(), ids, bool(data.get("concede", False))), []


def parse_judge(
    text: str, cases: list[AgentCase], evidence: dict[str, EvidenceItem]
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    """Judge scores (clipped to 0-1) and summaries that passed every check."""
    data = extract_json(text)
    if data is None:
        return {}, {}, ["the judge's answer was not a JSON object"]
    scores, summaries, problems = {}, {}, []
    raw_scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    raw_summaries = data.get("summaries") if isinstance(data.get("summaries"), dict) else {}
    for case in cases:
        try:
            scores[case.persona] = min(max(float(raw_scores[case.persona]), 0.0), 1.0)
        except (KeyError, TypeError, ValueError):
            problems.append(f"no usable score for {case.persona}")
        cited = [evidence[i] for i in case.cited_ids() if i in evidence]
        issue = check_sentence(
            raw_summaries.get(case.persona), cited, f"summary of {case.persona}", max_len=900
        )
        if issue:
            problems += issue
        else:
            summaries[case.persona] = raw_summaries[case.persona].strip()
    return scores, summaries, problems
