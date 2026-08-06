"""Tests for the structured trade-plan protocol (plan.py)."""

from __future__ import annotations

import pytest

from trade_tools.plan import TradePlan


def buy_plan(**over) -> TradePlan:
    base = dict(
        symbol="000725.SZ",
        decision="buy",
        signal_date="2024-01-02",
        signal_close=10.0,
        stop_price=9.5,
        target_price=11.0,
        max_hold_days=10,
        position_pct=0.1,
        account_equity=100_000.0,
    )
    base.update(over)
    return TradePlan(**base)


class TestValidate:
    def test_valid_buy_passes(self) -> None:
        assert buy_plan().validate() == []

    def test_missing_stop_and_target(self) -> None:
        p = buy_plan(stop_price=None, target_price=None)
        reasons = p.validate()
        assert any("stop_price" in r and "target_price" in r for r in reasons)

    def test_buy_missing_position_pct(self) -> None:
        p = buy_plan(position_pct=None)
        assert any("position_pct" in r for r in p.validate())

    def test_buy_missing_hold_days(self) -> None:
        p = buy_plan(max_hold_days=None)
        assert any("max_hold_days" in r for r in p.validate())

    def test_low_rr_blocked(self) -> None:
        p = buy_plan(stop_price=9.9, target_price=10.1)
        assert any("G7" in r for r in p.validate())

    def test_stop_at_or_above_entry(self) -> None:
        p = buy_plan(stop_price=10.5, target_price=11.0)
        assert any("stop_price 必须 < 入场价" in r for r in p.validate())

    def test_risk_over_1pct(self) -> None:
        p = buy_plan(position_pct=1.0)  # 10% stop distance at full size
        assert any("G1" in r for r in p.validate())

    def test_position_pct_out_of_range(self) -> None:
        p = buy_plan(position_pct=1.5)
        assert any("position_pct" in r for r in p.validate())

    def test_bad_decision(self) -> None:
        p = buy_plan(decision="maybe")
        assert any("decision" in r for r in p.validate())

    def test_empty_symbol(self) -> None:
        p = buy_plan(symbol="")
        assert any("symbol" in r for r in p.validate())


class TestNonBuy:
    def test_no_buy_without_risk_params_is_ok(self) -> None:
        p = buy_plan(
            decision="no_buy",
            stop_price=None,
            target_price=None,
            position_pct=None,
            max_hold_days=None,
        )
        assert p.validate() == []


class TestDerived:
    def test_entry_price_defaults_to_signal_close(self) -> None:
        p = buy_plan()
        assert p.entry_price == 10.0

    def test_entry_price_uses_trigger(self) -> None:
        p = buy_plan(entry_ref=10.2)
        assert p.entry_price == 10.2

    def test_rr(self) -> None:
        assert buy_plan().rr == pytest.approx(2.0)

    def test_position_value_from_equity(self) -> None:
        assert buy_plan().position_value == 10_000.0
