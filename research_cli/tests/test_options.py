"""The options desk: pricing, the chain, and the account's arithmetic.

The pricing tests check against values that can be looked up rather than
against this code's own output, which is the only way a pricing test means
anything. The account tests are mostly about money moving in the right
direction by the right amount -- the failures that make a paper account
flattering rather than useful are all of that kind, and none of them are
visible by reading the screen.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from research_cli.options import chain as chain_module
from research_cli.options.greeks import (
    CONTRACT_SIZE,
    black_scholes_price,
    greeks,
    implied_volatility,
    intrinsic_value,
)
from research_cli.options.paper import (
    Account,
    OrderError,
    Position,
    STARTING_CASH,
    load_account,
    mark_price,
    save_account,
    short_requirement,
)

# Textbook case: S=100, K=100, T=1, r=5%, sigma=20%.
_TEXTBOOK = dict(spot=100.0, strike=100.0, years=1.0, rate=0.05, vol=0.20)


class TestPricing:
    def test_call_and_put_match_published_values(self) -> None:
        call = black_scholes_price(kind="call", **_TEXTBOOK)
        put = black_scholes_price(kind="put", **_TEXTBOOK)
        assert call == pytest.approx(10.450584, abs=1e-6)
        assert put == pytest.approx(5.573526, abs=1e-6)

    def test_put_call_parity_holds(self) -> None:
        call = black_scholes_price(kind="call", **_TEXTBOOK)
        put = black_scholes_price(kind="put", **_TEXTBOOK)
        forward = 100.0 - 100.0 * math.exp(-0.05)
        assert call - put == pytest.approx(forward, abs=1e-9)

    def test_greeks_match_published_values(self) -> None:
        result = greeks(kind="call", **_TEXTBOOK)
        assert result.delta == pytest.approx(0.636831, abs=1e-6)
        assert result.gamma == pytest.approx(0.018762, abs=1e-6)

    def test_theta_is_per_day_and_vega_per_point(self) -> None:
        """The units are the whole point; annualized figures read as nonsense
        on a screen that quotes a one-month contract."""
        result = greeks(kind="call", **_TEXTBOOK)
        assert result.theta == pytest.approx(-6.414028 / 365.0, abs=1e-6)
        assert result.vega == pytest.approx(37.524035 / 100.0, abs=1e-6)
        assert result.rho == pytest.approx(53.232482 / 100.0, abs=1e-6)

    def test_per_contract_scales_by_the_multiplier(self) -> None:
        one = greeks(kind="call", **_TEXTBOOK)
        hundred = one.per_contract()
        assert hundred.price == pytest.approx(one.price * CONTRACT_SIZE)
        assert hundred.delta == pytest.approx(one.delta * CONTRACT_SIZE)

    def test_expired_contract_is_worth_its_intrinsic_value(self) -> None:
        assert black_scholes_price(110, 100, 0.0, 0.05, 0.2, "call") == 10.0
        assert intrinsic_value(110, 100, "put") == 0.0
        result = greeks(110, 100, 0.0, 0.05, 0.2, "call")
        assert result.delta == 1.0
        assert result.gamma == 0.0 and result.vega == 0.0

    @pytest.mark.parametrize("strike", [80, 95, 100, 105, 130])
    @pytest.mark.parametrize("years", [7 / 365, 30 / 365, 0.5, 2.0])
    @pytest.mark.parametrize("vol", [0.1, 0.25, 0.8])
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_implied_volatility_recovers_the_input(
        self, strike: float, years: float, vol: float, kind: str
    ) -> None:
        price = black_scholes_price(100, strike, years, 0.04, vol, kind)
        if price < 0.01:
            pytest.skip("below the tick -- no quote would exist")
        recovered = implied_volatility(price, 100, strike, years, 0.04, kind)
        if recovered is None:
            # Legitimate: see the vega test below.
            assert greeks(100, strike, years, 0.04, vol, kind).vega < 0.005
        else:
            assert recovered == pytest.approx(vol, abs=1e-4)

    def test_unmeasurable_implied_volatility_is_none_not_a_guess(self) -> None:
        """A deep in-the-money contract a day out prices the same at 10% vol
        as at 70%, so the quote pins nothing down and no figure should be
        printed beside it."""
        price = black_scholes_price(100, 60, 1 / 365, 0.04, 0.2, "call")
        assert implied_volatility(price, 100, 60, 1 / 365, 0.04, "call") is None

    def test_a_quote_below_intrinsic_has_no_implied_volatility(self) -> None:
        assert implied_volatility(1.0, 100, 80, 0.5, 0.04, "call") is None


class TestChain:
    def test_modelled_chain_is_stable_between_calls(self) -> None:
        """A screen that redraws different open interest every refresh looks
        broken, and Python's salted hash makes that the default."""
        expiry = date.today() + timedelta(days=30)
        first = chain_module.fetch_chain("SPY", expiry, source="mock")
        second = chain_module.fetch_chain("SPY", expiry, source="mock")
        assert [row.open_interest for row in first.rows] == [
            row.open_interest for row in second.rows
        ]

    def test_modelled_chain_brackets_the_underlying(self) -> None:
        expiry = date.today() + timedelta(days=30)
        result = chain_module.fetch_chain("SPY", expiry, source="mock")
        strikes = sorted({row.strike for row in result.rows})
        assert strikes[0] < result.spot < strikes[-1]
        assert result.synthetic and result.warnings

    def test_every_book_is_the_right_way_round(self) -> None:
        expiry = date.today() + timedelta(days=45)
        result = chain_module.fetch_chain("AAPL", expiry, source="mock")
        for row in result.rows:
            assert row.ask > 0
            if row.bid is not None:
                assert row.bid <= row.ask

    def test_fills_cross_the_spread(self) -> None:
        row = chain_module.ContractQuote(
            kind="call", strike=100.0, bid=1.05, ask=1.35, last=1.2,
            vendor_iv=0.3, volume=10, open_interest=100,
        )
        assert row.fill_price("buy") == 1.35
        assert row.fill_price("sell") == 1.05
        assert row.mid == pytest.approx(1.20)
        assert row.spread_pct == pytest.approx(0.25)

    def test_a_one_sided_book_has_no_midpoint(self) -> None:
        """Treating a missing bid as zero halves the apparent price of every
        illiquid contract on the screen."""
        row = chain_module.ContractQuote(
            kind="put", strike=50.0, bid=None, ask=0.05, last=None,
            vendor_iv=0.9, volume=0, open_interest=3,
        )
        assert row.mid is None
        assert row.fill_price("sell") is None
        assert row.fill_price("buy") == 0.05

    def test_vendor_implied_volatility_is_used_when_plausible(self) -> None:
        row = chain_module.ContractQuote(
            kind="call", strike=100.0, bid=1.05, ask=1.35, last=1.2,
            vendor_iv=0.3123, volume=10, open_interest=100,
        )
        value, origin = row.implied_vol(spot=100.0, years=0.25)
        assert value == pytest.approx(0.3123)
        assert origin == "vendor"

    def test_a_placeholder_implied_volatility_is_re_solved(self) -> None:
        row = chain_module.ContractQuote(
            kind="call", strike=100.0, bid=1.05, ask=1.35, last=1.2,
            vendor_iv=1e-05, volume=10, open_interest=100,
        )
        value, origin = row.implied_vol(spot=100.0, years=0.25)
        assert origin == "solved from the midpoint"
        assert 0.01 < value < 1.0

    def test_expiry_is_measured_to_the_new_york_close(self) -> None:
        """Not to midnight: on the last day that is most of the time value."""
        expiry = date.today() + timedelta(days=1)
        result = chain_module.fetch_chain("SPY", expiry, source="mock")
        morning = datetime.combine(expiry, datetime.min.time(), tzinfo=timezone.utc)
        hours = result.years_to_expiry(morning) * 365 * 24
        assert 15 < hours < 22

    def test_find_tolerates_a_float_round_trip(self) -> None:
        expiry = date.today() + timedelta(days=30)
        result = chain_module.fetch_chain("SPY", expiry, source="mock")
        target = result.rows[0]
        assert result.find(target.kind, target.strike + 1e-9) is target


def _account() -> Account:
    return Account()


class TestOrders:
    def test_buying_pays_the_ask_and_the_commission(self) -> None:
        account = _account()
        position, ticket = account.open_position(
            "AAPL", "2030-01-18", 190.0, "call", "buy", 2, fill_price=1.35, spot=185.0
        )
        assert ticket.premium == pytest.approx(270.0)
        assert ticket.commission == pytest.approx(1.30)
        assert account.cash == pytest.approx(STARTING_CASH - 271.30)
        assert position.contracts == 2
        assert position.collateral == 0.0
        # A long option ties up no collateral, so buying power is just cash.
        assert account.buying_power == pytest.approx(account.cash)

    def test_selling_receives_the_bid_and_ties_up_collateral(self) -> None:
        account = _account()
        _, ticket = account.open_position(
            "AAPL", "2030-01-18", 180.0, "put", "sell", 1, fill_price=2.50, spot=185.0
        )
        assert ticket.cash_effect == pytest.approx(250.0 - 0.65)
        assert account.cash == pytest.approx(STARTING_CASH + 249.35)
        # Cash secured: the account must be able to buy 100 shares at 180.
        assert ticket.collateral == pytest.approx(18_000.0)
        assert account.buying_power == pytest.approx(account.cash - 18_000.0)

    def test_an_uncovered_call_is_held_to_regulation_t(self) -> None:
        requirement = short_requirement("call", strike=200.0, contracts=1, spot=185.0, premium_per_share=1.0)
        # 20% of spot, less the 15 points out of the money, plus the premium.
        assert requirement == pytest.approx((0.20 * 185.0 - 15.0 + 1.0) * 100)
        far = short_requirement("call", strike=400.0, contracts=1, spot=185.0, premium_per_share=0.05)
        # Far out of the money the 10% floor binds instead of going negative.
        assert far == pytest.approx((0.10 * 185.0 + 0.05) * 100)

    def test_an_order_beyond_buying_power_is_refused(self) -> None:
        account = _account()
        with pytest.raises(OrderError) as raised:
            account.open_position(
                "SPY", "2030-01-18", 600.0, "put", "sell", 5, fill_price=8.0, spot=590.0
            )
        assert "buying power" in str(raised.value)
        assert account.positions == []
        assert account.cash == pytest.approx(STARTING_CASH)

    def test_an_empty_book_cannot_be_traded(self) -> None:
        account = _account()
        with pytest.raises(OrderError):
            account.open_position(
                "AAPL", "2030-01-18", 190.0, "call", "buy", 1, fill_price=None, spot=185.0
            )

    def test_closing_realizes_the_move_less_both_commissions(self) -> None:
        account = _account()
        account.open_position(
            "AAPL", "2030-01-18", 190.0, "call", "buy", 1, fill_price=1.00, spot=185.0
        )
        outcome = account.close_position("p1", fill_price=1.60)
        # 60 cents a share on 100 shares, less 65 cents of commission each way.
        assert outcome["realized"] == pytest.approx(60.0 - 0.65)
        assert account.realized == pytest.approx(60.0 - 1.30)
        assert account.positions == []
        assert account.cash == pytest.approx(STARTING_CASH + 60.0 - 1.30)

    def test_a_partial_close_leaves_the_rest_open(self) -> None:
        """Half the line goes, and half the collateral comes back with it."""
        account = _account()
        account.open_position(
            "KO", "2030-01-18", 80.0, "put", "sell", 2, fill_price=2.00, spot=85.0
        )
        assert account.collateral_held == pytest.approx(16_000.0)
        account.close_position("p1", fill_price=1.00, contracts=1)
        remaining = account.positions[0]
        assert remaining.contracts == -1
        assert remaining.collateral == pytest.approx(8_000.0)
        assert account.collateral_held == pytest.approx(8_000.0)

    def test_realized_plus_unrealized_reconciles_with_equity(self) -> None:
        """The one identity the blotter has to satisfy. It is where a missing
        commission or a released collateral shows up."""
        account = _account()
        account.open_position(
            "AAPL", "2030-01-18", 190.0, "call", "buy", 2, fill_price=1.35, spot=185.0
        )
        account.open_position(
            "AAPL", "2030-01-18", 170.0, "put", "sell", 1, fill_price=2.10, spot=185.0
        )
        account.close_position("p1", fill_price=1.80, contracts=1)

        marks = {"p1": 2.00, "p2": 2.40}
        unrealized = sum(
            position.unrealized(marks[position.id]) for position in account.positions
        )
        assert account.realized + unrealized == pytest.approx(
            account.equity(marks) - account.starting_cash
        )


class TestExpiry:
    def _expired(self, kind: str, strike: float, contracts: int) -> Account:
        account = _account()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        account.sequence = 1
        account.positions.append(
            Position(
                id="p1", symbol="AAPL", expiry=yesterday, strike=strike, kind=kind,
                contracts=contracts, average_price=1.00,
                opened_at=datetime.now(timezone.utc).isoformat(),
                collateral=strike * 100 if contracts < 0 and kind == "put" else 0.0,
            )
        )
        account.cash -= 100.0 * contracts  # what the fill would have cost
        return account

    def test_a_worthless_option_expires_and_leaves_the_book(self) -> None:
        account = self._expired("call", 200.0, 1)
        events = account.settle_expired(lambda symbol, when: 185.0)
        assert account.positions == []
        assert events[0]["outcome"] == "expired worthless"
        assert events[0]["realized"] == pytest.approx(-100.0)

    def test_a_long_in_the_money_call_is_exercised(self) -> None:
        account = self._expired("call", 180.0, 1)
        cash_before = account.cash
        events = account.settle_expired(lambda symbol, when: 185.0)
        assert account.cash == pytest.approx(cash_before + 500.0)
        assert "exercised" in events[0]["outcome"]
        assert events[0]["realized"] == pytest.approx(400.0)

    def test_a_short_in_the_money_put_is_assigned(self) -> None:
        account = self._expired("put", 190.0, -1)
        cash_before = account.cash
        events = account.settle_expired(lambda symbol, when: 185.0)
        assert account.cash == pytest.approx(cash_before - 500.0)
        assert "assigned" in events[0]["outcome"]
        # Collateral goes with the position, so buying power comes back.
        assert account.collateral_held == 0.0

    def test_settlement_waits_for_a_price_rather_than_guessing(self) -> None:
        account = self._expired("call", 180.0, 1)
        assert account.settle_expired(lambda symbol, when: None) == []
        assert len(account.positions) == 1

    def test_an_unexpired_position_is_left_alone(self) -> None:
        account = _account()
        account.open_position(
            "AAPL", (date.today() + timedelta(days=30)).isoformat(), 190.0,
            "call", "buy", 1, fill_price=1.00, spot=185.0,
        )
        assert account.settle_expired(lambda symbol, when: 999.0) == []
        assert len(account.positions) == 1


class TestMarking:
    def _position(self, contracts: int) -> Position:
        return Position(
            id="p1", symbol="AAPL", expiry="2030-01-18", strike=190.0, kind="call",
            contracts=contracts, average_price=1.00, opened_at="2030-01-01T00:00:00+00:00",
            collateral=0.0,
        )

    def test_a_long_marks_at_the_bid_and_a_short_at_the_ask(self) -> None:
        assert mark_price(self._position(1), bid=1.05, ask=1.35) == 1.05
        assert mark_price(self._position(-1), bid=1.05, ask=1.35) == 1.35

    def test_intrinsic_value_is_the_floor_when_the_book_is_empty(self) -> None:
        """An in-the-money contract nobody is bidding for is not worthless,
        and a short one is not free to buy back."""
        assert mark_price(self._position(1), bid=None, ask=None, intrinsic=4.0) == 4.0
        assert mark_price(self._position(-1), bid=None, ask=None, intrinsic=4.0) == 4.0


class TestPersistence:
    def test_an_account_survives_a_round_trip(self, tmp_path) -> None:
        account = _account()
        account.open_position(
            "AAPL", "2030-01-18", 190.0, "call", "buy", 2, fill_price=1.35, spot=185.0
        )
        path = save_account(account, tmp_path / "paper.json")
        restored = load_account(path)
        assert restored.cash == pytest.approx(account.cash)
        assert restored.positions == account.positions
        assert restored.sequence == account.sequence

    def test_a_corrupt_file_starts_a_fresh_account_rather_than_crashing(self, tmp_path) -> None:
        path = tmp_path / "paper.json"
        path.write_text("{not json at all", encoding="utf-8")
        assert load_account(path).cash == pytest.approx(STARTING_CASH)

    def test_a_reset_returns_the_opening_balance(self) -> None:
        account = _account()
        account.open_position(
            "AAPL", "2030-01-18", 190.0, "call", "buy", 1, fill_price=1.35, spot=185.0
        )
        account.reset()
        assert account.cash == pytest.approx(STARTING_CASH)
        assert account.positions == [] and account.ledger == []
