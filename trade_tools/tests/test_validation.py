"""Tests for the Track B validation pipeline (validation.py)."""

from __future__ import annotations

import pandas as pd

from helpers import make_bars
from trade_tools.execution import ExecutionSimulator
from trade_tools.pit import Point
from trade_tools.validation import (
    make_buy_hold_plan,
    make_r1r4_plan,
    paired_deltas,
    report,
    run_points,
)

BASE_VOL = 1_000_000.0


def _df(n=270, decision_idx=200, vol=1_000_000.0, spike: bool = True) -> pd.DataFrame:
    """稳步上升 270 根 bar；第 ``decision_idx`` 根放量（触发 R2 用）。

    返回归一化帧（date -> DatetimeIndex），与 rules/execution 内部口径一致。
    """
    closes = [10.0 + i * 0.02 for i in range(n)]
    rows, prev = [], closes[0]
    for i, c in enumerate(closes):
        v = vol * 2 if (spike and i == decision_idx) else vol
        rows.append([prev, c + 0.02, c - 0.02, c, v])
        prev = c
    df = make_bars(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def _point(df: pd.DataFrame, regime: str = "趋势", symbol: str = "000725.SZ") -> Point:
    return Point(symbol=symbol, date=str(df.index[200].date()), regime=regime)


class TestBuyHoldBaseline:
    def test_plan_is_hold_only_full_size(self) -> None:
        df = _df()
        plan = make_buy_hold_plan(_point(df), df, 100_000.0)
        assert plan.hold_only and plan.max_hold_days == 60
        assert plan.position_pct == 1.0
        assert plan.validate() == []

    def test_settles_time_stop_at_60(self) -> None:
        df = _df()
        r = ExecutionSimulator(df).settle(make_buy_hold_plan(_point(df), df, 100_000.0))
        assert r.filled
        assert r.exit_reason == "time_stop"
        assert r.hold_days == 60
        assert r.net_ret > 0  # 稳步上升 → 基准赚钱


class TestR1R4Arm:
    def test_triggered_plan_has_stop_target_position(self) -> None:
        df = _df()  # 第 200 根放量突破 → R2
        plan = make_r1r4_plan(_point(df), df, 100_000.0)
        assert plan is not None
        assert plan.metadata["reason"] == "R2"
        assert plan.stop_price < plan.entry_price < plan.target_price
        assert 0 < plan.position_pct <= 1.0
        assert plan.validate() == []

    def test_no_trigger_returns_none(self) -> None:
        df = _df(spike=False)  # 没有放量 → R1/R2 都不触发
        assert make_r1r4_plan(_point(df), df, 100_000.0) is None


class TestRunPoints:
    def test_runs_maker_across_points(self) -> None:
        df = _df()
        bars_by_symbol = {"000725.SZ": df}
        points = [_point(df), _point(df, regime="震荡")]
        results = run_points(points, bars_by_symbol, make_buy_hold_plan, 100_000.0)
        assert len(results) == 2
        assert all(r.filled for r in results)
        assert all(r.exit_reason == "time_stop" for r in results)
        assert all(r.net_ret > 0 for r in results)

    def test_missing_symbol_zeroed(self) -> None:
        df = _df()
        points = [_point(df)]
        results = run_points(points, {}, make_buy_hold_plan, 100_000.0)
        assert results[0].decision == "no_buy"
        assert results[0].net_ret == 0.0


class TestReport:
    def test_paired_deltas_and_report_structure(self) -> None:
        df = _df()
        bars = {"000725.SZ": df}
        pts = [_point(df), _point(df, symbol="000725.SZ")]
        agent = run_points(pts, bars, make_r1r4_plan, 100_000.0)  # R2 触发
        baseline = run_points(pts, bars, make_buy_hold_plan, 100_000.0)
        deltas = paired_deltas(agent, baseline)
        assert len(deltas) == 2
        assert all(isinstance(d, float) for d in deltas)

        rep = report(deltas)
        assert rep["n"] == 2
        assert set(rep["summary"]) >= {"mean", "median", "win_rate", "n"}
        assert len(rep["bootstrap_ci"]) == 2
        assert rep["gate"]["verdict"] in ("pass", "abandon", "insufficient")
