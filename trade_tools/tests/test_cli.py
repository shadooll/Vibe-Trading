"""Tests for the unified CLI (cli.py).

Each subcommand's pure path is covered offline (no network): check (pass/block/
protocol violation), settle (fill/abandon/no bar/missing column), pack --csv
(JSON on disk + no lookahead), sample (determinism + spacing), power (row
count), verdict (three tiers). The network path of ``pack`` without ``--csv``
is deliberately not exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from helpers import make_bars, make_scenario
from trade_tools.cli import main


# ── 数据构造工具 ───────────────────────────────────────────────────────────────
def buy_plan(**over) -> dict:
    """一份合法的 buy 计划（JSON dict），可用 **over 覆盖/扩展。"""
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
    return base


def write_json(tmp_path: Path, name: str, data: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def write_csv(tmp_path: Path, name: str, df: pd.DataFrame) -> str:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def rising_bars(
    n: int = 320, start_price: float = 10.0, start="2021-01-04"
) -> pd.DataFrame:
    """单调上行日线（open/high/low/close/volume），用于造趋势行情。"""
    closes = [start_price + i * 0.02 for i in range(n)]
    rows = [
        [round(c - 0.01, 4), round(c * 1.002, 4), round(c * 0.998, 4), round(c, 4), 2e7]
        for c in closes
    ]
    return make_bars(rows, start=start)


def trading_days(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """两个日期之间的交易日数（含端点）；make_bars 日历 = 纯工作日。"""
    lo, hi = (a, b) if a <= b else (b, a)
    return len(pd.bdate_range(lo, hi))


# ── check ─────────────────────────────────────────────────────────────────────
class TestCheck:
    def test_valid_buy_passes(self, capsys, tmp_path) -> None:
        plan = write_json(tmp_path, "plan.json", buy_plan())
        assert main(["check", plan]) == 0
        out = capsys.readouterr().out
        assert "通过" in out
        assert "000725.SZ" in out
        assert "G1-G8" in out

    def test_g8_watchlist_blocked(self, capsys, tmp_path) -> None:
        plan = write_json(tmp_path, "plan.json", buy_plan(gate={"in_watchlist": False}))
        assert main(["check", plan]) == 0
        out = capsys.readouterr().out
        assert "拦截" in out
        assert "G8" in out

    def test_protocol_violation_printed(self, capsys, tmp_path) -> None:
        plan = write_json(
            tmp_path,
            "plan.json",
            buy_plan(target_price=None, max_hold_days=None, position_pct=None),
        )
        assert main(["check", plan]) == 0
        out = capsys.readouterr().out
        assert "协议违规" in out
        assert "stop_price" in out and "target_price" in out
        assert "max_hold_days" in out

    def test_bad_json_friendly_error(self, capsys, tmp_path) -> None:
        path = tmp_path / "plan.json"
        path.write_text("{ not json", encoding="utf-8")
        assert main(["check", str(path)]) == 2
        assert "错误" in capsys.readouterr().out


# ── settle ────────────────────────────────────────────────────────────────────
class TestSettle:
    def test_filled_target_round_trip(self, capsys, tmp_path) -> None:
        plan = write_json(tmp_path, "plan.json", buy_plan())
        csv_path = write_csv(
            tmp_path,
            "ohlcv.csv",
            make_scenario(
                10.0,
                [
                    [10.20, 10.50, 10.00, 10.40, 1_000_000],
                    [10.60, 11.00, 10.50, 10.90, 1_200_000],
                    [11.00, 11.10, 10.90, 11.00, 1_000_000],
                ],
            ),
        )
        assert main(["settle", plan, csv_path]) == 0
        out = capsys.readouterr().out
        assert "filled=True" in out
        assert "exit_reason=target" in out
        assert "entry=2024-01-03 @ 10.21" in out
        assert "exit=2024-01-05 @ 10.99" in out
        assert "net_ret=+" in out
        assert "cost=" in out
        assert "hold_days=2" in out

    def test_gap_up_abandon(self, capsys, tmp_path) -> None:
        plan = write_json(tmp_path, "plan.json", buy_plan())
        csv_path = write_csv(
            tmp_path,
            "ohlcv.csv",
            make_scenario(10.0, [[10.40, 10.60, 10.30, 10.50, 1_000_000]]),
        )
        assert main(["settle", plan, csv_path]) == 0
        out = capsys.readouterr().out
        assert "filled=False" in out
        assert "exit_reason=gap_up_abandon" in out

    def test_no_bar_after_signal(self, capsys, tmp_path) -> None:
        plan = write_json(tmp_path, "plan.json", buy_plan())
        csv_path = write_csv(
            tmp_path,
            "ohlcv.csv",
            make_bars([[10.0, 10.0, 10.0, 10.0, 1_000_000]], start="2024-01-02"),
        )
        assert main(["settle", plan, csv_path]) == 0
        assert "exit_reason=no_bar_after_signal" in capsys.readouterr().out

    def test_missing_column_friendly_error(self, capsys, tmp_path) -> None:
        plan = write_json(tmp_path, "plan.json", buy_plan())
        bad = tmp_path / "bad.csv"
        bad.write_text(
            "date,open,high,low,close\n2024-01-02,10,10.5,9.8,10.2\n", encoding="utf-8"
        )
        assert main(["settle", plan, str(bad)]) == 2
        out = capsys.readouterr().out
        assert "缺列" in out
        assert "volume" in out


# ── pack --csv（离线）─────────────────────────────────────────────────────────
class TestPack:
    def test_offline_pack_writes_json_no_lookahead(self, capsys, tmp_path) -> None:
        df = rising_bars(320)
        as_of = str(pd.Timestamp(df["date"].iloc[299]).date())
        csv_path = write_csv(tmp_path, "ohlcv.csv", df)
        out_path = tmp_path / "pack.json"
        argv = ["pack", "600519.SH", as_of, "--csv", csv_path, "--out", str(out_path)]
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert "数据包 600519.SH" in out
        assert "regime=趋势" in out
        assert "已落盘" in out
        pack = json.loads(out_path.read_text(encoding="utf-8"))
        assert pack["ok"] is True
        assert pack["n_bars"] == 300  # 只含 <= as_of 的 300 根，后 20 根被截断
        assert pack["bars"][-1]["date"] == as_of

    def test_insufficient_bars_error(self, capsys, tmp_path) -> None:
        df = rising_bars(100)
        as_of = str(pd.Timestamp(df["date"].iloc[-1]).date())
        csv_path = write_csv(tmp_path, "ohlcv.csv", df)
        assert main(["pack", "600519.SH", as_of, "--csv", csv_path]) == 1
        out = capsys.readouterr().out
        assert "失败" in out
        assert "MA200" in out


# ── sample（纯本地）───────────────────────────────────────────────────────────
class TestSample:
    def _universe_dir(self, tmp_path: Path, frames: dict[str, pd.DataFrame]) -> str:
        d = tmp_path / "universe"
        d.mkdir()
        for symbol, df in frames.items():
            df.to_csv(d / f"{symbol}.csv", index=False)
        return str(d)

    def _argv(
        self, universe_dir: str, start: str, end: str, per_regime: str = "3"
    ) -> list[str]:
        return [
            "sample",
            "--universe-dir",
            universe_dir,
            "--start",
            start,
            "--end",
            end,
            "--per-regime",
            per_regime,
            "--seed",
            "42",
        ]

    def test_deterministic_and_format(self, capsys, tmp_path) -> None:
        frames = {
            "600001.SH": rising_bars(400),
            "600002.SH": rising_bars(400, start_price=20.0),
        }
        d = self._universe_dir(tmp_path, frames)
        start = str(frames["600001.SH"]["date"].iloc[250])
        end = str(frames["600001.SH"]["date"].iloc[399])
        argv = self._argv(d, start, end)
        assert main(argv) == 0
        out1 = capsys.readouterr().out
        assert main(argv) == 0
        out2 = capsys.readouterr().out
        assert out1 == out2
        lines = out1.strip().splitlines()
        assert lines
        for line in lines:
            symbol, date, regime = line.split(",")
            assert symbol in frames
            assert len(date.split("-")) == 3
            assert regime in ("趋势", "震荡", "熊市")

    def test_spacing_within_stratum(self, capsys, tmp_path) -> None:
        frames = {"600001.SH": rising_bars(400)}
        d = self._universe_dir(tmp_path, frames)
        start = str(frames["600001.SH"]["date"].iloc[250])
        end = str(frames["600001.SH"]["date"].iloc[399])
        assert main(self._argv(d, start, end)) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        dates = sorted(pd.Timestamp(line.split(",")[1]) for line in lines)
        assert len(dates) >= 2
        for a, b in zip(dates, dates[1:]):
            assert trading_days(a, b) >= 60

    def test_empty_window_prints_nothing(self, capsys, tmp_path) -> None:
        frames = {"600001.SH": rising_bars(400)}
        d = self._universe_dir(tmp_path, frames)
        # 窗口只含首日 -> 范围内无足够历史判定 regime，返回 0 且不打印任何点。
        early = str(pd.Timestamp(frames["600001.SH"]["date"].iloc[0]))
        argv = [
            "sample",
            "--universe-dir",
            d,
            "--start",
            early,
            "--end",
            early,
            "--per-regime",
            "3",
            "--seed",
            "42",
        ]
        assert main(argv) == 0
        assert capsys.readouterr().out.strip() == ""


# ── power ─────────────────────────────────────────────────────────────────────
class TestPower:
    def test_row_count_and_values(self, capsys) -> None:
        assert main(["power", "--delta", "0.02", "0.03", "0.05"]) == 0
        out = capsys.readouterr().out
        data_lines = [ln for ln in out.splitlines() if ln.startswith("  delta")]
        assert len(data_lines) == 3
        assert "619" in out  # required_n(0.02, 0.2) = 619
        assert "99" in out  # required_n(0.05, 0.2) = 99

    def test_default_deltas(self, capsys) -> None:
        assert main(["power"]) == 0
        out = capsys.readouterr().out
        assert len([ln for ln in out.splitlines() if ln.startswith("  delta")]) == 3


# ── verdict ───────────────────────────────────────────────────────────────────
class TestVerdict:
    def test_pass(self, capsys) -> None:
        assert main(["verdict", "0.03", "0.09", "0.05"]) == 0
        out = capsys.readouterr().out
        assert "[通过]" in out
        assert "Phase 4" in out

    def test_abandon(self, capsys) -> None:
        assert main(["verdict", "--", "-0.05", "-0.01", "-0.03"]) == 0
        assert "[放弃]" in capsys.readouterr().out

    def test_insufficient(self, capsys) -> None:
        assert main(["verdict", "--", "-0.01", "0.04", "0.015"]) == 0
        out = capsys.readouterr().out
        assert "[证据不足]" in out
        assert "跨零" in out

    def test_below_cost_floor_insufficient(self, capsys) -> None:
        assert main(["verdict", "0.005", "0.03", "0.01", "--cost-floor", "0.02"]) == 0
        assert "[证据不足]" in capsys.readouterr().out
