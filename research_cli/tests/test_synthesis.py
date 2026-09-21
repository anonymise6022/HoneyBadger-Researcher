"""Tests for the template renderer and the validator that guards it.

The validator tests matter more than the template ones. In Phase 1 the
validator guards a string template and should never fire; the point of these
tests is to prove it *would* fire, so that it can be trusted in Phase 2 when
a language model writes the prose.
"""

from __future__ import annotations

from datetime import date

import pytest

from research_cli.evidence.schema import Counterevidence
from research_cli.evidence.why_moved import build_why_moved_bundle
from research_cli.query_parser import parse_query
from research_cli.synthesis.template_report import REQUIRED_SECTIONS, render_report
from research_cli.synthesis.validator import extract_numbers, validate_report

TODAY = date(2026, 9, 21)


@pytest.fixture(scope="module")
def report(spy_bundle) -> str:
    return render_report(spy_bundle)


# --- the template ----------------------------------------------------------

def test_report_has_every_required_section(report: str) -> None:
    for section in REQUIRED_SECTIONS:
        assert section in report


def test_report_quotes_the_original_question(spy_bundle, report: str) -> None:
    assert spy_bundle.query in report


def test_report_states_the_move_before_listing_any_factors(report: str) -> None:
    """Most moves are noise; leading with explanations implies otherwise."""
    assert report.index("SUMMARY") < report.index("KEY EVIDENCE")


def test_report_wraps_without_joining_words(report: str) -> None:
    """Regression: the wrapper dropped the space at every line break."""
    for artefact in ("forreal", "past4years", "ishigh", "coincidedwith"):
        assert artefact not in report
    assert all(len(line) <= 80 for line in report.split("\n"))


def test_report_does_not_lowercase_acronyms(report: str) -> None:
    """Regression: 'High-yield bonds (HYG)' rendered as '(hyg)'."""
    for mangled in ("(hyg)", "(vix)", "(uup)"):
        assert mangled not in report


def test_snapshot_bundle_is_refused_by_the_attribution_renderer(spy_bundle) -> None:
    snapshot = spy_bundle.model_copy(update={"question_type": "snapshot"})
    with pytest.raises(ValueError, match="attribution"):
        render_report(snapshot)


def test_report_ends_with_the_not_advice_statement(report: str) -> None:
    assert "not investment advice" in report.lower()


# --- the validator on good input -------------------------------------------

def test_template_output_passes_validation(spy_bundle, report: str) -> None:
    """A validator that cannot pass known-correct code will be switched off."""
    result = validate_report(report, spy_bundle)
    assert result.ok, "\n".join(str(i) for i in result.issues)
    assert result.numbers_matched == result.numbers_checked


@pytest.mark.parametrize(
    "query",
    ["why did SPY fall today", "why did AAPL fall today", "why did EURUSD rally today",
     "why did TSLA jump this week", "why did KO decline today"],
)
def test_every_ticker_type_validates_clean(query: str) -> None:
    bundle = build_why_moved_bundle(parse_query(query, today=TODAY), source="mock")
    result = validate_report(render_report(bundle), bundle)
    assert result.ok, "\n".join(str(i) for i in result.issues)


def test_dates_are_not_read_as_numeric_claims(report: str) -> None:
    """'Monday 21 September 2026' must not surface 21 and 2026 as claims."""
    numbers = [value for value, _, _ in extract_numbers(report)]
    assert 2026 not in numbers


# --- the validator on bad input --------------------------------------------

def test_invented_number_is_caught(spy_bundle, report: str) -> None:
    result = validate_report(report + "\n  SPY fell 7.83% on heavy selling.", spy_bundle)
    assert not result.ok
    assert any(i.kind == "unsupported_number" for i in result.errors)


def test_altered_number_is_caught(spy_bundle, report: str) -> None:
    """A figure quietly changed from the bundle must not slip through."""
    swing = f"{spy_bundle.observation.daily_vol_pct:.2f}%"
    tampered = report.replace(swing, "9.99%")
    assert swing in report
    result = validate_report(tampered, spy_bundle)
    assert not result.ok
    assert any(i.kind == "unsupported_number" for i in result.errors)


@pytest.mark.parametrize(
    "sentence",
    [
        "The selloff was caused by rising yields.",
        "It dropped because inflation disappointed.",
        "The move was due to the Fed.",
        "Rising rates drove the decline.",
        "The print triggered a selloff.",
        "Weak data led to the drop.",
        "This resulted in heavy selling.",
    ],
)
def test_causal_language_is_caught(spy_bundle, report: str, sentence: str) -> None:
    result = validate_report(f"{report}\n  {sentence}", spy_bundle)
    assert any(i.kind == "causal_language" for i in result.errors), sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "SPY is a buy at these levels.",
        "We recommend buying here.",
        "You should sell before earnings.",
        "Our price target is 800.",
        "The index looks undervalued.",
        "This is a strong buy.",
    ],
)
def test_verdict_language_is_caught(spy_bundle, report: str, sentence: str) -> None:
    result = validate_report(f"{report}\n  {sentence}", spy_bundle)
    assert any(i.kind == "verdict_language" for i in result.errors), sentence


def test_missing_section_is_caught(spy_bundle, report: str) -> None:
    result = validate_report(report.replace("COMPETING EXPLANATIONS", "OTHER"), spy_bundle)
    assert any(i.kind == "missing_section" for i in result.errors)


def test_dropped_counterevidence_is_caught(spy_bundle) -> None:
    """The failure mode of a research tool is a confident story."""
    bundle = spy_bundle.model_copy(
        update={
            "counterevidence": [
                Counterevidence(label="Volume was light", detail="Only 0.4x average.")
            ]
        }
    )
    stripped = "\n".join(
        line for line in render_report(bundle).split("\n") if "Volume was light" not in line
    )
    result = validate_report(stripped, bundle)
    assert any(i.kind == "missing_counterevidence" for i in result.errors)


def test_partial_counterevidence_is_a_warning_not_an_error(spy_bundle) -> None:
    bundle = spy_bundle.model_copy(
        update={
            "counterevidence": [
                Counterevidence(label="First point", detail="a"),
                Counterevidence(label="Second point", detail="b"),
            ]
        }
    )
    result = validate_report(render_report(bundle).replace("Second point", "x"), bundle)
    assert any(i.kind == "partial_counterevidence" for i in result.warnings)
    assert result.ok, "a partial omission should warn, not block"


def test_validator_never_raises_on_arbitrary_text(spy_bundle) -> None:
    """A validator that crashes on bad input cannot report on bad input."""
    for text in ("", "   ", "no numbers here at all", "###", "1e999", "nan", "∞ 3.4.5.6"):
        assert isinstance(validate_report(text, spy_bundle).ok, bool)


def test_rounding_is_tolerated_but_wrong_digits_are_not(spy_bundle) -> None:
    """1.5503 may print as 1.55; it may not print as 1.56."""
    close = spy_bundle.observation.close
    good = validate_report(
        f"{render_report(spy_bundle)}\n  Close was {close:.1f}.", spy_bundle
    )
    assert good.ok
    bad = validate_report(
        f"{render_report(spy_bundle)}\n  Close was {close + 7.77:.2f}.", spy_bundle
    )
    assert not bad.ok


def test_percentage_and_fraction_forms_both_trace(spy_bundle, report: str) -> None:
    """Hit rates are stored as 0.68 and printed as 68%; both must validate."""
    factor = next(
        (f for f in spy_bundle.factors if f.historical_hit_rate is not None), None
    )
    if factor is None:
        pytest.skip("no scored factor in this bundle")
    result = validate_report(
        f"{report}\n  The rate was {factor.historical_hit_rate:.0%} "
        f"({factor.historical_hit_rate:.2f} as a fraction).",
        spy_bundle,
    )
    assert result.ok, "\n".join(str(i) for i in result.issues)
