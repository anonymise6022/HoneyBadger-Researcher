"""The options desk: live chains, greeks, and a paper account that obeys them.

Three modules, deliberately separable:

    greeks.py   Black-Scholes prices and sensitivities, pure standard
                library, in the units a broker screen uses.
    chain.py    listed contracts at one expiry -- real quotes when the
                market can be reached, a modelled chain when it cannot.
    paper.py    an account that fills across the spread, charges
                commission, collateralises shorts and settles at expiry.

Nothing here recommends a trade or scores one. The desk shows what a
contract costs, what it is sensitive to, and what happens to the account if
it is bought or sold -- which is the same standing rule as the rest of the
tool, applied to a screen where the temptation to do otherwise is strongest.
"""

from __future__ import annotations

__all__ = ["chain", "greeks", "paper"]
