"""Assemble the evidence for "is X going to go up" -- without predicting it.

Nobody can answer that question, and this tool does not try. It answers the
nearest questions that *can* be answered from data: what is this thing (the
snapshot), and what has it done before in each of several kinds of market
(the scenarios). Every scenario number is a frequency from history and is
labelled that way.

The outlook bundle is the snapshot bundle with the scenario pack attached.
A failure in the scenario layer costs the scenarios and adds a warning; it
never costs the snapshot, which stands on its own.
"""

from __future__ import annotations

from ..data.price_ingest import PriceDataError, Source, UnknownSymbolError
from ..query_parser import ParsedQuery
from ..scenarios.engine import DEFAULT_SCENARIO_YEARS, build_scenarios
from .schema import EvidenceBundle
from .snapshot import build_snapshot_bundle

__all__ = ["INDEX_SYMBOLS", "build_outlook_bundle"]

#: Funds a beginner is likely to ask about. A synthetic fallback can invent a
#: sector for these, so the name check backs up the missing-sector check.
INDEX_SYMBOLS = frozenset({"SPY", "QQQ", "DIA", "IWM", "VOO", "VTI", "IVV", "GLD", "USO", "TLT"})


def build_outlook_bundle(
    parsed: ParsedQuery,
    source: Source = "auto",
    scenario_years: float = DEFAULT_SCENARIO_YEARS,
    include_news: bool = True,
) -> EvidenceBundle:
    """Snapshot evidence plus historical scenarios for one symbol.

    Raises whatever `build_snapshot_bundle` raises when the symbol itself
    cannot be priced. Scenario failures are recorded, not raised.
    """
    snapshot = build_snapshot_bundle(parsed, source=source, include_news=include_news)
    fundamentals = snapshot.fundamentals
    is_fund = (
        parsed.is_fx
        or parsed.ticker.upper() in INDEX_SYMBOLS
        or fundamentals is None
        or fundamentals.sector is None
    )

    warnings = list(snapshot.warnings)
    limitations = list(snapshot.limitations)
    data_sources = list(snapshot.data_sources)
    try:
        pack = build_scenarios(
            parsed.yf_symbol, parsed.ticker, parsed.end, source=source,
            years=scenario_years, is_fund=is_fund,
        )
    except (PriceDataError, UnknownSymbolError, ValueError) as exc:
        warnings.append(f"historical scenarios could not be measured ({exc})")
        return snapshot.model_copy(update={"question_type": "outlook", "warnings": warnings})

    warnings.extend(pack.warnings)
    limitations.extend(pack.limitations)
    data_sources.extend(pack.data_sources)
    return snapshot.model_copy(
        update={
            "question_type": "outlook",
            "scenarios": pack.results,
            "scenario_baseline": pack.baseline,
            "warnings": warnings,
            "limitations": limitations,
            "data_sources": data_sources,
            "is_synthetic": snapshot.is_synthetic or pack.synthetic,
        }
    )
