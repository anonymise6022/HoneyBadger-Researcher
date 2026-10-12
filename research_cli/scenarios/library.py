"""The fixed library of "what if" scenarios, each tied to something measurable.

A scenario is only given a number when it has a **trigger** -- a stated rule
on a price series anyone can check, such as "oil up more than 30% over three
months". The thresholds are conventional, round and written into the label
rather than tuned: "a bear market is 20% below the high" is a definition
people already use, and a threshold chosen to make the results look
interesting would be the same free-parameter problem the hit-rate code
avoids with quintiles.

Scenarios with no honest trigger -- a war, a new regulation -- get none.
They still carry named historical **episodes**, whose returns are facts,
and a `watch_for` list. They are never given a frequency, because price
data cannot say how often wars start.

Episodes are hand-picked examples and are labelled as such. They illustrate;
the trigger-based statistics are the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

__all__ = [
    "PROXY_SYMBOLS",
    "SCENARIOS",
    "Episode",
    "ScenarioSpec",
    "Trigger",
    "scenario_by_id",
]

#: Proxy id -> vendor symbol. "self" means the symbol being asked about.
PROXY_SYMBOLS: dict[str, str] = {
    "oil": "CL=F",
    "vix": "^VIX",
    "rates": "^TNX",
    "market": "SPY",
}

TriggerKind = str  # "change_pct" | "change_points" | "level_below" | "drawdown_pct" | "shock_day"


@dataclass(frozen=True)
class Trigger:
    """A rule on one series that says "this situation is happening now".

    kind:
        change_pct     percent change over `window` sessions beyond `threshold`
        change_points  change in level (e.g. yield points) beyond `threshold`
        level_below    the level is below `threshold`
        drawdown_pct   percent below the `window`-session high, at or past `threshold`
        shock_day      the symbol's own daily move, in multiples of its trailing
                       `window`-session volatility, at or past `threshold`, on a
                       day the market fell less than 1%
    """

    proxy: str
    kind: TriggerKind
    window: int
    threshold: float
    description: str


@dataclass(frozen=True)
class Episode:
    label: str
    start: date
    end: date


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    label: str
    plain_label: str
    description: str
    trigger: Trigger | None
    episodes: tuple[Episode, ...] = ()
    watch_for: tuple[str, ...] = ()
    applies_to_funds: bool = True
    keywords: tuple[str, ...] = field(default_factory=tuple)


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "supply_shock", "Supply shock", "Oil and supplies suddenly get expensive",
        "A sudden jump in the cost of energy or key parts, such as an oil spike or a "
        "chip shortage.",
        Trigger("oil", "change_pct", 63, 30.0, "oil up more than 30% over three months"),
        (
            Episode("2008 oil spike", date(2008, 1, 2), date(2008, 7, 11)),
            Episode("2021 chip shortage", date(2021, 1, 4), date(2021, 12, 31)),
            Episode("2022 energy price spike", date(2022, 2, 24), date(2022, 6, 14)),
        ),
        ("oil and gas prices", "shipping delays", "shortages of parts the company needs"),
        keywords=("supply", "oil", "shortage", "strait", "shipping", "energy", "chip"),
    ),
    ScenarioSpec(
        "war_geopolitical", "War or geopolitical crisis", "A war or big fight between countries",
        "A war or a sharp political crisis between countries.",
        None,
        (
            Episode("2003 Iraq invasion", date(2003, 3, 20), date(2003, 4, 30)),
            Episode("2022 Russia invades Ukraine", date(2022, 2, 24), date(2022, 3, 31)),
            Episode("2023 Israel-Hamas war begins", date(2023, 10, 9), date(2023, 10, 31)),
        ),
        ("news of military conflict", "sanctions", "sudden jumps in the VIX fear index"),
        keywords=("war", "invasion", "conflict", "sanction", "geopolitic", "attack"),
    ),
    ScenarioSpec(
        "rate_shock", "Interest-rate shock", "Borrowing money suddenly costs a lot more",
        "Interest rates rise fast, which makes borrowing dearer and future profits worth "
        "less today.",
        Trigger("rates", "change_points", 126, 1.0,
                "10-year Treasury yield up more than 1 percentage point in six months"),
        (
            Episode("2013 taper tantrum", date(2013, 5, 2), date(2013, 9, 5)),
            Episode("2022 rate hikes", date(2022, 1, 3), date(2022, 10, 21)),
        ),
        ("central bank meetings", "inflation reports", "the 10-year Treasury yield"),
        keywords=("rate", "fed", "yield", "inflation", "interest"),
    ),
    ScenarioSpec(
        "recession_crash", "Recession or market crash", "The whole market falls a lot",
        "A broad fall in the stock market, usually alongside a weaker economy.",
        Trigger("market", "drawdown_pct", 252, -20.0,
                "S&P 500 (SPY) more than 20% below its high of the past year"),
        (
            Episode("2008 financial crisis", date(2008, 9, 15), date(2009, 3, 9)),
            Episode("2020 Covid crash", date(2020, 2, 19), date(2020, 3, 23)),
            Episode("2022 bear market", date(2022, 1, 3), date(2022, 10, 12)),
        ),
        ("rising unemployment", "falling company profits", "credit markets under stress"),
        keywords=("recession", "crash", "bear", "depression", "unemployment", "crisis"),
    ),
    ScenarioSpec(
        "sector_regulation", "New rules for the industry", "New laws change how the business works",
        "Governments change the rules for this kind of business, for example new laws, "
        "fines or a break-up case.",
        None,
        (),
        ("proposed laws aimed at this industry", "regulator investigations or lawsuits",
         "statements from officials about this sector"),
        keywords=("regulat", "law", "antitrust", "ban", "fine", "government", "tariff"),
    ),
    ScenarioSpec(
        "company_shock", "Company-specific shock", "Bad news about just this company",
        "A sudden fall that hits this company but not the market, the kind a weak earnings "
        "report or a product failure brings.",
        Trigger("self", "shock_day", 20, -3.0,
                "a one-day drop of more than 3 times its usual daily move while the "
                "S&P 500 fell less than 1%"),
        (),
        ("earnings report dates", "product launches or recalls", "management changes"),
        applies_to_funds=False,
        keywords=("earnings", "miss", "competitor", "recall", "lawsuit", "ceo", "product"),
    ),
    ScenarioSpec(
        "melt_up", "Strong market rally", "The whole market zooms up",
        "A fast, broad rise in the stock market.",
        Trigger("market", "change_pct", 63, 15.0, "S&P 500 (SPY) up more than 15% in three months"),
        (
            Episode("2020-21 rebound", date(2020, 3, 24), date(2021, 2, 12)),
            Episode("2023-24 rally", date(2023, 10, 27), date(2024, 3, 28)),
        ),
        ("very fast price rises across the market", "lots of first-time buyers piling in"),
        keywords=("rally", "boom", "melt", "bubble", "moon", "surge"),
    ),
    ScenarioSpec(
        "quiet_market", "Nothing much happens", "A calm, boring stretch",
        "Markets stay calm and nothing big happens.",
        Trigger("vix", "level_below", 1, 15.0, "the VIX fear index below 15 (a calm market)"),
        (),
        ("the VIX staying low", "few big news days"),
        keywords=("calm", "quiet", "nothing", "boring", "sideways"),
    ),
)


def scenario_by_id(scenario_id: str) -> ScenarioSpec | None:
    return next((s for s in SCENARIOS if s.scenario_id == scenario_id), None)
