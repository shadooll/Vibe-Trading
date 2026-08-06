"""Tests for the G1-G8 decision gate (gate.py)."""

from __future__ import annotations

from trade_tools.gate import GateContext, gate
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


class TestGate:
    def test_valid_buy_passes(self) -> None:
        assert gate(buy_plan(), GateContext()).passed

    def test_out_of_watchlist_blocked(self) -> None:
        v = gate(buy_plan(), GateContext(in_watchlist=False))
        assert v.blocked and any("G8" in x for x in v.violations)

    def test_chi_next_blocked(self) -> None:
        v = gate(buy_plan(symbol="300750.SZ"), GateContext())
        assert v.blocked and any("禁买" in x for x in v.violations)

    def test_star_board_blocked(self) -> None:
        v = gate(buy_plan(symbol="688111.SH"), GateContext())
        assert v.blocked and any("禁买" in x for x in v.violations)

    def test_bse_blocked(self) -> None:
        v = gate(buy_plan(symbol="832566.BJ"), GateContext())
        assert v.blocked and any("禁买" in x for x in v.violations)

    def test_low_rr_blocked(self) -> None:
        v = gate(buy_plan(stop_price=9.9, target_price=10.1), GateContext())
        assert v.blocked and any("G7" in x for x in v.violations)

    def test_risk_over_1pct_blocked(self) -> None:
        v = gate(buy_plan(position_pct=1.0), GateContext())
        assert v.blocked and any("G1" in x for x in v.violations)

    def test_daily_frequency_blocked(self) -> None:
        v = gate(buy_plan(), GateContext(daily_trades=1))
        assert v.blocked and any("G2" in x for x in v.violations)

    def test_weekly_frequency_blocked(self) -> None:
        v = gate(buy_plan(), GateContext(week_trades=3))
        assert v.blocked and any("G3" in x for x in v.violations)

    def test_daily_circuit_blocked(self) -> None:
        v = gate(buy_plan(), GateContext(daily_loss_pct=0.02))
        assert v.blocked and any("G4" in x for x in v.violations)

    def test_consecutive_losses_blocked(self) -> None:
        v = gate(buy_plan(), GateContext(consecutive_losses=3))
        assert v.blocked and any("G5" in x for x in v.violations)

    def test_no_buy_always_passes(self) -> None:
        v = gate(buy_plan(decision="no_buy"), GateContext())
        assert v.passed
