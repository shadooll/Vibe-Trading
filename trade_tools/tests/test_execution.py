"""Tests for the A-share paper-trade execution simulator (execution.py)."""

from __future__ import annotations

import pytest

from helpers import make_scenario
from trade_tools.execution import ExecutionSimulator
from trade_tools.plan import TradePlan

SIGNAL_DATE = "2024-01-02"


def buy_plan(**over) -> TradePlan:
    base = dict(
        symbol="000725.SZ",
        decision="buy",
        signal_date=SIGNAL_DATE,
        signal_close=10.0,
        stop_price=9.5,
        target_price=11.0,
        max_hold_days=10,
        position_pct=0.1,
        account_equity=100_000.0,
    )
    base.update(over)
    return TradePlan(**base)


def run(after_signal: list, **plan_over):
    df = make_scenario(10.0, after_signal)
    return ExecutionSimulator(df).settle(buy_plan(**plan_over))


class TestEntry:
    def test_happy_target_round_trip(self) -> None:
        r = run(
            [
                [10.20, 10.50, 10.00, 10.40, 1_000_000],  # 01-03 entry
                [10.60, 11.00, 10.50, 10.90, 1_200_000],  # 01-04 target hit
                [11.00, 11.10, 10.90, 11.00, 1_000_000],  # 01-05 sell at open
            ]
        )
        assert r.filled
        assert r.exit_reason == "target"
        assert r.entry_price == pytest.approx(10.21)  # 10.20 + 1 tick slippage
        assert r.exit_price == pytest.approx(10.99)  # 11.00 - 1 tick slippage
        assert r.gross_ret > 0
        assert r.net_ret < r.gross_ret  # 费用吃收益

    def test_gap_up_abandon(self) -> None:
        r = run([[10.40, 10.60, 10.30, 10.50, 1_000_000]])  # open > 10.0*1.03
        assert not r.filled
        assert r.exit_reason == "gap_up_abandon"

    def test_gap_down_below_stop_abandon(self) -> None:
        r = run([[9.30, 9.50, 9.20, 9.40, 1_000_000]])  # open < stop 9.5
        assert not r.filled
        assert r.exit_reason == "gap_down_stop_abandon"

    def test_limit_up_no_fill(self) -> None:
        r = run([[11.00, 11.00, 11.00, 11.00, 100_000]])  # one-price at +10%
        assert not r.filled
        assert r.exit_reason == "limit_up_no_fill"


class TestExit:
    def test_stop_hit_sells_next_open(self) -> None:
        r = run(
            [
                [10.20, 10.30, 9.40, 9.60, 1_000_000],  # 01-03 entry + stop hit
                [9.70, 9.80, 9.60, 9.70, 1_000_000],  # 01-04 sell at open
            ]
        )
        assert r.filled
        assert r.exit_reason == "stop"
        assert r.entry_price == pytest.approx(10.21)
        assert r.exit_price == pytest.approx(9.49)  # min(open, stop) - 1 tick

    def test_gap_through_stop_fills_at_open(self) -> None:
        r = run(
            [
                [10.20, 10.30, 9.40, 9.60, 1_000_000],  # 01-03 stop hit
                [9.20, 9.40, 9.10, 9.30, 1_000_000],  # 01-04 open < stop
            ]
        )
        assert r.exit_reason == "stop"
        assert r.exit_price == pytest.approx(9.19)  # fill at open 9.20 - 1 tick

    def test_limit_down_locked_defers_to_next_day(self) -> None:
        r = run(
            [
                [10.20, 10.30, 9.40, 9.60, 1_000_000],  # 01-03 stop hit
                [8.64, 8.64, 8.64, 8.64, 50_000],  # 01-04 locked limit-down
                [8.90, 9.00, 8.80, 8.95, 1_000_000],  # 01-05 sell at open
            ]
        )
        assert r.exit_reason == "stop"
        assert r.exit_price == pytest.approx(8.89)  # 8.90 - 1 tick

    def test_suspension_carries_forward(self) -> None:
        r = run(
            [
                [10.20, 10.30, 9.40, 9.60, 1_000_000],  # 01-03 stop hit
                None,  # 01-04 suspended
                [9.80, 9.90, 9.70, 9.85, 1_000_000],  # 01-05 resume
            ]
        )
        assert r.exit_reason == "stop"
        assert r.exit_price == pytest.approx(9.49)  # min(9.80, stop 9.5) - 1 tick

    def test_time_stop_after_max_hold(self) -> None:
        r = run(
            [
                [10.20, 10.30, 10.00, 10.20, 1_000_000],  # held 1
                [10.20, 10.40, 10.10, 10.30, 1_000_000],  # held 2
                [10.30, 10.50, 10.20, 10.40, 1_000_000],  # held 3 = max_hold
                [10.40, 10.60, 10.30, 10.50, 1_000_000],  # 01-06 sell at open
            ],
            max_hold_days=3,
        )
        assert r.exit_reason == "time_stop"
        assert r.hold_days == 3
        assert r.exit_price == pytest.approx(10.39)


class TestFailClosed:
    def test_no_buy_is_not_filled(self) -> None:
        r = run([[10.20, 10.50, 10.00, 10.40, 1_000_000]], decision="no_buy")
        assert not r.filled
        assert r.exit_reason == "decision=no_buy"

    def test_protocol_violation_fails_closed(self) -> None:
        r = run(
            [[10.20, 10.50, 10.00, 10.40, 1_000_000]],
            target_price=None,
            max_hold_days=None,
        )
        assert not r.filled
        assert r.exit_reason == "protocol_violation"

    def test_no_bar_after_signal(self) -> None:
        from helpers import make_bars

        df = make_bars(
            [[10.0, 10.0, 10.0, 10.0, 1_000_000]], start="2024-01-02"
        )  # signal-day bar only, nothing after it
        r = ExecutionSimulator(df).settle(buy_plan())
        assert not r.filled
        assert r.exit_reason == "no_bar_after_signal"
