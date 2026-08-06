"""Tests for the R1-R4 deterministic entry engine (rules.py)."""

from __future__ import annotations

import pytest
import pandas as pd

from helpers import make_bars
from trade_tools.pit import REGIME_BEAR, REGIME_RANGE, REGIME_TREND, REGIME_UNKNOWN
from trade_tools.rules import (
    r1_signal,
    r2_signal,
    r3_signal,
    r4_signal,
    r1r4_signal,
)

BASE_VOL = 1_000_000.0


def _series(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """从收盘价序列构造 OHLCV（open=prev close，振幅 ±0.02）。"""
    if volumes is None:
        volumes = [BASE_VOL] * len(closes)
    rows, prev = [], closes[0]
    for c, v in zip(closes, volumes):
        rows.append([prev, c + 0.02, c - 0.02, c, v])
        prev = c
    return make_bars(rows)


def _uptrend(n: int = 220, start: float = 10.0, step: float = 0.02) -> list[float]:
    return [start + i * step for i in range(n)]


class TestR2:
    def test_breakout_with_volume_surge_triggers(self) -> None:
        closes = _uptrend()  # 最高 14.38
        bars = _series(
            closes + [closes[-1] + 0.02],
            volumes=[BASE_VOL] * len(closes) + [2 * BASE_VOL],
        )
        sig = r2_signal(bars)
        assert sig is not None and sig["reason"] == "R2"

    def test_new_high_without_volume_surge_no_trigger(self) -> None:
        closes = _uptrend()
        bars = _series(closes + [closes[-1] + 0.02])  # 量没放大
        assert r2_signal(bars) is None

    def test_not_new_high_no_trigger(self) -> None:
        closes = _uptrend()
        bars = _series(
            closes + [closes[-1] - 0.5],
            volumes=[BASE_VOL] * len(closes) + [2 * BASE_VOL],
        )
        assert r2_signal(bars) is None


class TestR4:
    def test_pullback_shrink_up_day_triggers(self) -> None:
        closes = [10.0] * 30 + [9.90, 9.80, 9.70, 9.60, 9.50, 9.60]
        volumes = [BASE_VOL] * 30 + [BASE_VOL] * 5 + [BASE_VOL * 0.4]
        sig = r4_signal(_series(closes, volumes))
        assert sig is not None and sig["reason"] == "R4"

    def test_no_shrink_no_trigger(self) -> None:
        closes = [10.0] * 30 + [9.90, 9.80, 9.70, 9.60, 9.50, 9.60]
        sig = r4_signal(_series(closes))  # 量没缩
        assert sig is None

    def test_no_pullback_no_trigger(self) -> None:
        closes = [10.0] * 35 + [10.0, 10.1]
        sig = r4_signal(
            _series(closes, [BASE_VOL] * 35 + [BASE_VOL * 0.4, BASE_VOL * 0.4])
        )
        assert sig is None


class TestR3:
    def test_oversold_recovery_triggers(self) -> None:
        # 急跌后强反弹：RSI 先破 30 再站回
        closes = [10.0] * 40 + [9.0, 8.2, 7.6, 7.1, 6.8, 8.5]
        sig = r3_signal(_series(closes))
        assert sig is not None and sig["reason"] == "R3"

    def test_steady_rise_no_trigger(self) -> None:
        assert r3_signal(_series(_uptrend())) is None


class TestR1:
    def test_pullback_to_ma20_bounce_triggers(self) -> None:
        closes = _uptrend()  # 顶端 14.38
        # 回踩：14.38 → 14.32 → 14.26 → 反弹 14.34（不破 20 日线、缩量、收阳站回）
        tail = [closes[-1], closes[-1] - 0.06, closes[-1] - 0.12, closes[-1] - 0.04]
        volumes = [BASE_VOL] * len(closes) + [BASE_VOL * 0.3] * len(tail)
        sig = r1_signal(_series(closes + tail, volumes))
        assert sig is not None and sig["reason"] == "R1"

    def test_breakout_no_pullback_no_trigger(self) -> None:
        closes = _uptrend() + [14.5]  # 创新高不是回踩
        assert r1_signal(_series(closes)) is None

    def test_price_far_above_ma20_not_pullback(self) -> None:
        """Pilot 实测：价高 MA20 15% 也触发 R1 是假触发——回踩须贴近线。"""
        # 长期横盘 10 元（MA20≈10），近 6 根快速拉到 12，今天 11.8 小回
        closes = [10.0] * 200 + [10.5, 11.0, 11.5, 12.0, 12.0, 11.8]
        # 近 5 日低点 ≈10.98 远高于 MA20×1.03≈10.38 → 不贴近 20 日线 → 不触发
        assert r1_signal(_series(closes)) is None


class TestStopTarget:
    def test_stop_below_entry_target_above(self) -> None:
        closes = _uptrend()
        bars = _series(
            closes + [closes[-1] + 0.02],
            volumes=[BASE_VOL] * len(closes) + [2 * BASE_VOL],
        )
        sig = r2_signal(bars)
        assert sig is not None
        e, s, t = sig["entry_ref"], sig["stop"], sig["target"]
        assert s < e < t
        assert t == pytest.approx(e + 2 * (e - s), rel=1e-6)


class TestDispatch:
    def test_trend_uses_r1_r2(self) -> None:
        closes = _uptrend()
        bars = _series(
            closes + [closes[-1] + 0.02],
            volumes=[BASE_VOL] * len(closes) + [2 * BASE_VOL],
        )
        assert r1r4_signal(bars, REGIME_TREND) is not None  # R2 触发

    def test_range_uses_r3_r4(self) -> None:
        closes = [10.0] * 30 + [9.90, 9.80, 9.70, 9.60, 9.50, 9.60]
        volumes = [BASE_VOL] * 30 + [BASE_VOL] * 5 + [BASE_VOL * 0.4]
        assert r1r4_signal(_series(closes, volumes), REGIME_RANGE) is not None  # R4

    def test_bear_treated_as_range(self) -> None:
        closes = [10.0] * 30 + [9.90, 9.80, 9.70, 9.60, 9.50, 9.60]
        volumes = [BASE_VOL] * 30 + [BASE_VOL] * 5 + [BASE_VOL * 0.4]
        assert r1r4_signal(_series(closes, volumes), REGIME_BEAR) is not None  # R4

    def test_unknown_no_trigger(self) -> None:
        closes = _uptrend()
        bars = _series(
            closes + [closes[-1] + 0.02],
            volumes=[BASE_VOL] * len(closes) + [2 * BASE_VOL],
        )
        assert r1r4_signal(bars, REGIME_UNKNOWN) is None
