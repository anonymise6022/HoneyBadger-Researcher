"""A terminal research tool for beginner retail investors.

Answers two kinds of question about a ticker, in plain language, using only
evidence it can show you:

    "why did SPY fall today"        -> attribution: what coincided with the move
    "is AAPL good to invest"        -> snapshot: fundamentals and peer context

The design rule that shapes everything here: **the tool never asserts a
cause and never gives a verdict.** Markets do not come with labels, and a
retail investor asking "why did my stock drop" is best served by an honest
list of what else happened that day, ranked by how often each factor has
historically coincided with moves like this one -- with the sample size
attached, and with the evidence that points the other way shown too.

Pipeline:

    query_parser.py    natural-language query -> structured request
                       (deterministic regex/keyword, never an LLM)
    data/              price, macro and fundamental ingestion, with a
                       synthetic fallback so nothing needs an API key
    evidence/          assemble a typed EvidenceBundle: observations,
                       candidate factors, hit rates, confidence, counterevidence
    synthesis/         turn the bundle into prose, then verify every number
                       in that prose traces back to a bundle field
    backtest/          walk-forward strategy testing, no lookahead by
                       construction
    display/           rich terminal rendering

Nothing is investment advice. See `synthesis.template_report` for how that
is enforced rather than merely stated.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
