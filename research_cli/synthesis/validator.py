"""Check that a report only says what the evidence bundle supports.

In Phase 1 this guards a string template, where it should never fire --
which is exactly the point. A validator that cannot pass code known to be
correct is a validator that will be switched off the first time it
complains. Getting it green against the template is what earns the right to
trust it in Phase 2, when a language model writes the prose and the failure
mode becomes a confidently invented number.

Four checks:

**Numeric traceability.** Every figure in the prose must trace to a field in
the bundle. Two kinds of text are excluded first, because they are somebody
else's words that the report is quoting and attributing: the user's own
question, and third-party headline titles. A report that echoes "is AAPL a
buy" is not making a recommendation, and a headline that says "raises price
target" is a fact about the headline. The report is stripped of dates, rules and list markers, the
remaining numbers are extracted, and each is matched against the bundle's
flattened numeric fields -- allowing for the formatting a report legitimately
does: rounding to a couple of decimals, scaling a 0-1 rate to a percentage,
taking an absolute value, and counting the length of a list. Anything left
unmatched is reported with the surrounding text so a human can see it.

**Causal language.** "Caused", "because", "due to", "drove", "triggered",
"led to", "resulted in" are forbidden outright. The tool observes
coincidence and cannot observe causation, so the vocabulary is constrained
rather than left to the writer's judgement.

**Verdict language.** "Buy", "sell", "hold", "recommend", "should invest"
and the like. This tool is not licensed to give advice and does not.

**Structure.** Required sections present; counterevidence surfaced whenever
the bundle holds any.

Severity: `error` means the report should not be shown as-is. `warning`
means a human should look. The CLI treats errors as blocking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..evidence.schema import EvidenceBundle

__all__ = [
    "CAUSAL_PATTERNS",
    "VERDICT_PATTERNS",
    "ValidationIssue",
    "ValidationReport",
    "extract_numbers",
    "validate_report",
]

Severity = Literal["error", "warning"]

#: Phrases asserting causation. Word-bounded so "becausexyz" is not matched
#: and, more importantly, so discussing the words in a limitations section
#: still trips the check -- if a report wants to say "this is not a cause",
#: it must phrase it without the forbidden verb.
CAUSAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcaused?\b", "caused"),
    (r"\bcausing\b", "causing"),
    (r"\bbecause\b", "because"),
    (r"\bdue to\b", "due to"),
    (r"\bdrove\b", "drove"),
    (r"\bdriven by\b", "driven by"),
    (r"\btriggered\b", "triggered"),
    (r"\bled to\b", "led to"),
    (r"\bresulted in\b", "resulted in"),
    (r"\bas a result of\b", "as a result of"),
    (r"\bthanks to\b", "thanks to"),
    (r"\bthe reason (?:for|was|is)\b", "the reason for"),
)

#: Phrases that would turn a research note into a recommendation.
VERDICT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\byou should (?:buy|sell|hold|invest|avoid)\b", "you should buy/sell/hold"),
    (r"\bwe recommend\b", "we recommend"),
    (r"\bi recommend\b", "i recommend"),
    (r"\brecommend (?:buying|selling|holding)\b", "recommend buying/selling"),
    (r"\b(?:strong|clear)\s+(?:buy|sell)\b", "strong buy/sell"),
    (r"\bis a (?:buy|sell)\b", "is a buy/sell"),
    (r"\b(?:buy|sell)\s+rating\b", "buy/sell rating"),
    (r"\bgood (?:time )?to (?:buy|sell)\b", "good time to buy/sell"),
    (r"\bworth (?:buying|selling)\b", "worth buying/selling"),
    (r"\bundervalued\b", "undervalued"),
    (r"\bovervalued\b", "overvalued"),
    (r"\bprice target\b", "price target"),
)

#: Numbers a report may always use: they are structural, not claims.
_ALWAYS_ALLOWED = {0.0, 1.0, 2.0, 100.0}

#: Durations inside fixed labels ("12-month range", "52-week high"). The
#: number is part of the label's name, not a measurement, so it is removed
#: before extraction rather than having to appear in the bundle.
_DURATION_LABEL = re.compile(
    r"\b\d{1,3}\s*-\s*(?:day|week|month|quarter|year)s?\b", re.IGNORECASE
)

_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
#: Written dates ("Monday 21 September 2026", "September 15, 2026"). Without
#: this the day and year read as unsupported numeric claims.
_WRITTEN_DATE = re.compile(
    rf"\b(?:\d{{1,2}}\s+(?:{_MONTH_NAMES})|(?:{_MONTH_NAMES})\s+\d{{1,2}})"
    rf"(?:,)?(?:\s+\d{{4}})?\b",
    re.IGNORECASE,
)
_RULE_LINE = re.compile(r"^[=\-]{3,}$", re.MULTILINE)
_LIST_MARKER = re.compile(r"^(\s*)\d+\.\s", re.MULTILINE)
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a report."""

    severity: Severity
    kind: str
    message: str
    excerpt: str = ""

    def __str__(self) -> str:
        suffix = f"  ...{self.excerpt}..." if self.excerpt else ""
        return f"[{self.severity}] {self.kind}: {self.message}{suffix}"


@dataclass
class ValidationReport:
    """The outcome of validating one report against one bundle."""

    issues: list[ValidationIssue] = field(default_factory=list)
    numbers_checked: int = 0
    numbers_matched: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when nothing blocking was found."""
        return not self.errors

    def summary(self) -> str:
        traced = (
            f"{self.numbers_matched}/{self.numbers_checked} numbers traced to the evidence"
            if self.numbers_checked
            else "no numbers to trace"
        )
        if self.ok and not self.warnings:
            return f"PASS -- {traced}, no issues."
        return (
            f"{'PASS' if self.ok else 'FAIL'} -- {traced}; "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)."
        )


def extract_numbers(text: str) -> list[tuple[float, str, str]]:
    """Pull every numeric claim out of report prose, with its context.

    Dates, horizontal rules and list markers are removed first: none of them
    is a claim about the data, and leaving them in would produce noise that
    trains a reader to ignore the validator.

    Returns (value, token, excerpt) triples. The raw token is kept because
    the precision a number was *printed* at sets the rounding tolerance, and
    reading that off the surrounding excerpt picks up whichever number came
    first in the sentence instead.

    A percentage is returned as written -- "80%" gives 80.0 -- and the
    matcher is responsible for also trying 0.80, since the bundle stores
    rates as fractions.
    """
    cleaned = _ISO_DATE.sub(" ", text)
    cleaned = _WRITTEN_DATE.sub(" ", cleaned)
    cleaned = _DURATION_LABEL.sub(" ", cleaned)
    cleaned = _RULE_LINE.sub(" ", cleaned)
    cleaned = _LIST_MARKER.sub(r"\1", cleaned)

    results: list[tuple[float, str, str]] = []
    for match in _NUMBER.finditer(cleaned):
        token = match.group(0)
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        start, stop = max(0, match.start() - 30), min(len(cleaned), match.end() + 30)
        excerpt = " ".join(cleaned[start:stop].split())
        results.append((value, token, excerpt))
    return results


def _remove_quoted(text: str, phrase: str, placeholder: str) -> str:
    """Remove `phrase` from `text` even after the renderer has wrapped it.

    A plain `str.replace` fails here: the report word-wraps, so a headline
    title arrives with newlines and indentation inserted between its words.
    Matching on whitespace-flexible word boundaries is what makes the
    exclusion actually hold.
    """
    words = phrase.split()
    if not words:
        return text
    pattern = r"\s+".join(re.escape(word) for word in words)
    return re.sub(pattern, placeholder, text)


def _bundle_strings(bundle: EvidenceBundle) -> list[str]:
    """Every free-text field in the bundle, for embedded-number extraction."""
    texts: list[str] = [bundle.period_description, bundle.ticker, bundle.query]
    texts.extend(bundle.limitations)
    texts.extend(bundle.warnings)
    texts.extend(bundle.data_sources)
    if bundle.observation is not None:
        texts.append(bundle.observation.plain_summary)
    for factor in bundle.factors:
        texts.extend([factor.what_happened, factor.condition_description, factor.label])
        if factor.note:
            texts.append(factor.note)
    for item in bundle.counterevidence:
        texts.extend([item.label, item.detail])
    for release in bundle.macro_releases:
        texts.extend([release.description, release.label])
    for peer in bundle.peers:
        texts.extend([peer.label, peer.metric])
    if bundle.fundamentals is not None:
        texts.extend(
            [bundle.fundamentals.company_name or "", bundle.fundamentals.sector or ""]
        )
    return [t for t in texts if t]


def _allowed_values(bundle: EvidenceBundle) -> set[float]:
    """Every value a report may legitimately print.

    Bundle fields, plus the transformations a report performs on them: a
    0-1 rate rendered as a percentage, an absolute value where the report
    writes "fell 2.1%" for a -2.1 field, and the lengths of the lists it
    counts.
    """
    values = set(_ALWAYS_ALLOWED)
    for value in bundle.numeric_fields().values():
        values.update({value, abs(value), value * 100.0, abs(value) * 100.0, value / 100.0})
        # Large figures are printed in human units -- a market cap of
        # 4.9e12 appears as "$4.9 trillion" -- so the scaled forms are
        # legitimate renderings of the same field.
        for scale in (1e3, 1e6, 1e9, 1e12):
            values.update({value / scale, abs(value) / scale})

    # Numbers that live inside bundle *strings* -- percentile bin edges in a
    # condition description, a beta in a what-happened sentence, the "500"
    # in "S&P 500". They are in the bundle, just not as numeric fields, and
    # a report that quotes or paraphrases those sentences is fully supported.
    for text in _bundle_strings(bundle):
        for match in _NUMBER.finditer(text):
            try:
                embedded = float(match.group(0).replace(",", ""))
            except ValueError:
                continue
            values.update({embedded, abs(embedded), embedded * 100.0, embedded / 100.0})

    counts = [
        len(bundle.factors), len(bundle.counterevidence), len(bundle.macro_releases),
        len(bundle.limitations), len(bundle.warnings), len(bundle.data_sources),
        len(bundle.peers), len(bundle.external_factors()), len(bundle.mechanical_factors()),
        len(bundle.supported_factors()),
    ]
    values.update(float(c) for c in counts)
    # Enumeration of any list the report walks.
    values.update(float(i) for c in counts for i in range(1, c + 1))
    return values


def _decimals(token: str) -> int:
    """Decimal places in the printed token itself, for the rounding tolerance.

    Measured on the token, never on the surrounding text: an excerpt often
    holds several numbers at different precisions, and taking the first
    would apply one number's tolerance to another.
    """
    _, _, fraction = token.partition(".")
    return len(fraction)


def _matches(value: float, allowed: set[float], decimals: int) -> bool:
    """Whether a printed number is consistent with some allowed value.

    A report rounds: 1.5503 prints as "1.55". So a match means some allowed
    value rounds to the printed one at the precision it was printed with.
    """
    tolerance = 0.5 * (10.0 ** -decimals) + 1e-9
    return any(abs(value - candidate) <= tolerance for candidate in allowed)


def validate_report(
    report: str,
    bundle: EvidenceBundle,
    check_numbers: bool = True,
    forbid_verdicts: bool = True,
    required_sections: tuple[str, ...] | None = None,
) -> ValidationReport:
    """Validate a rendered report against the bundle it was built from.

    Parameters
    ----------
    report : the rendered text.
    bundle : the bundle it should be derived from.
    check_numbers : trace every figure. Disable only to isolate the other
        checks while debugging.
    forbid_verdicts : flag buy/sell/hold language. Always on in practice.
    required_sections : headings that must appear. Defaults to the
        attribution report's sections.

    Returns a `ValidationReport`; never raises for report content, since a
    validator that crashes on bad input cannot report on bad input.
    """
    from .template_report import REQUIRED_SECTIONS

    result = ValidationReport()
    sections = required_sections if required_sections is not None else REQUIRED_SECTIONS
    upper = report.upper()

    for section in sections:
        if section.upper() not in upper:
            result.issues.append(
                ValidationIssue(
                    "error", "missing_section", f"required section {section!r} is absent"
                )
            )

    # The report echoes the user's question back to them. If they asked
    # "is AAPL a buy", that phrase is *their* words, not a recommendation the
    # tool is making, so it is removed before the language checks. Leaving it
    # in would flag every report that quotes a bluntly worded question.
    scannable = report
    if bundle.query:
        scannable = _remove_quoted(scannable, bundle.query, " [user question] ")
    # Third-party headline titles are quoted verbatim and attributed to
    # their publisher. "Wells Fargo raises price target" is a fact about
    # what a headline says, not this tool recommending anything, and its
    # numbers belong to the headline, not to the data. Publisher names are
    # removed too -- "24/7 Wall St." is a masthead, not a measurement.
    for item in bundle.news:
        scannable = _remove_quoted(scannable, item.title, " [quoted headline] ")
        scannable = _remove_quoted(scannable, item.publisher, " [publisher] ")

    lowered = scannable.lower()
    for pattern, label in CAUSAL_PATTERNS:
        for match in re.finditer(pattern, lowered):
            start, stop = max(0, match.start() - 40), min(len(scannable), match.end() + 40)
            result.issues.append(
                ValidationIssue(
                    "error",
                    "causal_language",
                    f"asserts causation with {label!r}; this tool reports coincidence only",
                    " ".join(scannable[start:stop].split()),
                )
            )

    if forbid_verdicts:
        for pattern, label in VERDICT_PATTERNS:
            for match in re.finditer(pattern, lowered):
                start, stop = max(0, match.start() - 40), min(len(scannable), match.end() + 40)
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "verdict_language",
                        f"gives investment advice with {label!r}",
                        " ".join(scannable[start:stop].split()),
                    )
                )

    if bundle.counterevidence:
        labels_present = sum(
            1 for item in bundle.counterevidence if item.label.lower()[:40] in lowered
        )
        if labels_present == 0:
            result.issues.append(
                ValidationIssue(
                    "error",
                    "missing_counterevidence",
                    f"the bundle holds {len(bundle.counterevidence)} piece(s) of "
                    "counterevidence and the report surfaces none",
                )
            )
        elif labels_present < len(bundle.counterevidence):
            result.issues.append(
                ValidationIssue(
                    "warning",
                    "partial_counterevidence",
                    f"only {labels_present} of {len(bundle.counterevidence)} pieces of "
                    "counterevidence appear in the report",
                )
            )

    if check_numbers:
        allowed = _allowed_values(bundle)
        for value, token, excerpt in extract_numbers(scannable):
            result.numbers_checked += 1
            if _matches(value, allowed, _decimals(token)):
                result.numbers_matched += 1
            else:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "unsupported_number",
                        f"{value:g} does not trace to any field in the evidence bundle",
                        excerpt,
                    )
                )
    return result
