"""Tests for PIT decision-point sampling and as-of data packs (pit.py)."""

from __future__ import annotations

import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from helpers import make_bars
from trade_tools.execution import _norm_df
from trade_tools.pit import (
    REGIME_BEAR,
    REGIME_RANGE,
    REGIME_TREND,
    REGIME_UNKNOWN,
    UniverseFilter,
    _board_ok,
    build_data_pack,
    compute_snapshot,
    regime_at,
    regime_stratum,
    sample_points,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── 合成行情工具（纯本地，不依赖网络）──────────────────────────────────────────
def linspace(start: float, stop: float, n: int) -> list[float]:
    if n <= 1:
        return [stop]
    step = (stop - start) / (n - 1)
    return [start + step * i for i in range(n)]


def spike(closes: list[float], pos: int, pct: float) -> list[float]:
    """把第 ``pos`` 根 bar 改成比前一根涨/跌 ``pct``（用于清掉 ST 近似判定）。"""
    out = list(closes)
    out[pos] = out[pos - 1] * (1 + pct)
    return out


def make_series(
    closes: list[float], start="2015-01-05", volume: float = 2e7
) -> pd.DataFrame:
    """从收盘价序列构造规范化的 OHLCV（DatetimeIndex，与 pit 的输入契约一致）。"""
    rows: list[tuple] = []
    date = _dt.date.fromisoformat(start)
    prev = closes[0]
    for close in closes:
        open_ = prev
        high = max(open_, close) * 1.002
        low = min(open_, close) * 0.998
        rows.append((str(date), open_, high, low, close, volume))
        prev = close
        date += _dt.timedelta(days=1)
        while date.weekday() >= 5:
            date += _dt.timedelta(days=1)
    raw = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    return _norm_df(raw)


def trend_series(n: int = 1200) -> list[float]:
    return spike(list(linspace(5, 11, n)), -40, 0.06)


def bear_series(n: int = 1200) -> list[float]:
    return spike(list(linspace(30, 15, n)), -40, -0.06)


def range_series(
    n_flat: int = 500, n_decline: int = 100, n_rise: int = 120, rise_end: float = 13.8
):
    """平高 → 下跌 → 反弹：反弹段维持 价<MA200 且 MA60 向上（震荡）。"""
    return spike(
        [20.0] * n_flat
        + list(linspace(20, 12, n_decline))
        + list(linspace(12, rise_end, n_rise)),
        -30,
        0.07,
    )


def unknown_series(n: int = 300) -> list[float]:
    """冲高后回落：价仍 > MA200 但 MA60 向下 -> 未知（回落尾部 ~26 根为未知）。"""
    return list(linspace(5, 12.5, n - 60)) + list(linspace(12.5, 11.0, 60))


def trading_day_distance(
    dates: pd.DatetimeIndex, a: pd.Timestamp, b: pd.Timestamp
) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return int(((dates >= lo) & (dates <= hi)).sum())


def dense_universe() -> dict[str, pd.DataFrame]:
    """趋势 x2 + 熊市 x2 + 震荡 x3（震荡窗口错开），全部主板代码。"""
    universe: dict[str, pd.DataFrame] = {}
    for i in range(2):
        universe[f"60000{i}.SH"] = make_series(trend_series())
        universe[f"60010{i}.SH"] = make_series(bear_series())
    for i, n_flat in enumerate([500, 650, 800]):
        universe[f"60020{i}.SH"] = make_series(range_series(n_flat=n_flat))
    return universe


# ── regime_at / regime_stratum ────────────────────────────────────────────────
class TestRegime:
    def test_trend(self) -> None:
        df = make_series(trend_series())
        assert regime_at(df["close"], df.index[-1]) == REGIME_TREND

    def test_bear_maps_to_range(self) -> None:
        df = make_series(bear_series())
        assert regime_at(df["close"], df.index[-1]) == REGIME_RANGE

    def test_unknown(self) -> None:
        df = make_series(unknown_series())
        assert regime_at(df["close"], df.index[-1]) == REGIME_UNKNOWN

    def test_insufficient_data_is_unknown(self) -> None:
        df = make_series(list(linspace(5, 11, 50)))
        assert regime_at(df["close"], df.index[-1]) == REGIME_UNKNOWN

    def test_empty_is_unknown(self) -> None:
        close = pd.Series(dtype="float64")
        assert regime_at(close, "2020-01-01") == REGIME_UNKNOWN

    def test_is_point_in_time(self) -> None:
        # 先涨 300 根后崩 60 根：在上涨段末端判趋势，用全量数据则判震荡。
        closes = list(linspace(5, 11, 300)) + list(linspace(11, 6, 60))
        df = make_series(closes, start="2015-01-05")
        assert regime_at(df["close"], df.index[290]) == REGIME_TREND
        assert regime_at(df["close"], df.index[-1]) == REGIME_RANGE

    def test_stratum_labels(self) -> None:
        trend = make_series(trend_series())
        bear = make_series(bear_series())
        rng = make_series(range_series())
        unk = make_series(unknown_series())
        assert regime_stratum(trend["close"], trend.index[-1]) == REGIME_TREND
        assert regime_stratum(bear["close"], bear.index[-1]) == REGIME_BEAR
        assert regime_stratum(rng["close"], rng.index[-1]) == REGIME_RANGE
        assert regime_stratum(unk["close"], unk.index[-1]) == REGIME_UNKNOWN

    def test_stratum_consistent_with_regime_at(self) -> None:
        for closes in (trend_series(), bear_series(), range_series(), unknown_series()):
            df = make_series(closes)
            at = regime_at(df["close"], df.index[-1])
            st = regime_stratum(df["close"], df.index[-1])
            if at == REGIME_RANGE:
                assert st in (REGIME_RANGE, REGIME_BEAR)
            else:
                assert st == at


# ── UniverseFilter ────────────────────────────────────────────────────────────
class TestUniverseFilter:
    def test_board_allow_main_board(self) -> None:
        assert _board_ok("600519.SH")
        assert _board_ok("000725.SZ")
        assert _board_ok("002594.SZ")  # 中小板已并入主板
        assert not _board_ok("300750.SZ")  # 创业板
        assert not _board_ok("688111.SH")  # 科创板
        assert not _board_ok("832566.BJ")  # 北交所
        assert not _board_ok("510300.SH")  # ETF
        assert not _board_ok("000300.SH")  # 指数
        assert not _board_ok("600519")  # 无后缀

    def test_eligible_main_board_stock_passes(self) -> None:
        df = make_series(trend_series(), start="2015-01-05")
        verdict = UniverseFilter().judge("600519.SH", df, df.index[-1])
        assert verdict.eligible, verdict.failures
        assert verdict.failures == ()

    def test_listing_age_insufficient(self) -> None:
        df = make_series(trend_series(n=500), start="2015-01-05")  # ~2 年
        verdict = UniverseFilter().judge("600519.SH", df, df.index[-1])
        assert not verdict.eligible
        assert any("上市不满" in x for x in verdict.failures)

    def test_low_turnover_fails(self) -> None:
        df = make_series(trend_series(), start="2013-01-05", volume=1e4)
        verdict = UniverseFilter().judge("600519.SH", df, df.index[-1])
        assert any("成交额" in x for x in verdict.failures)

    def test_volume_multiplier_applied(self) -> None:
        # volume 按"手"计：不乘 100 成交额不足，乘 100 换算成股后达标。
        df = make_series(trend_series(), start="2013-01-05", volume=2e6)
        as_of = df.index[-1]
        v_default = UniverseFilter().judge("600519.SH", df, as_of)
        assert any("成交额" in x for x in v_default.failures)
        v_lots = UniverseFilter(volume_multiplier=100.0).judge("600519.SH", df, as_of)
        assert v_lots.eligible, v_lots.failures

    def test_st_calm_series_flagged(self) -> None:
        df = make_series([10.0] * 300, start="2013-01-05")
        verdict = UniverseFilter().judge("600519.SH", df, df.index[-1])
        assert any("ST" in x for x in verdict.failures)

    def test_st_check_disabled(self) -> None:
        df = make_series([10.0] * 300, start="2013-01-05")
        verdict = UniverseFilter(st_check=None).judge("600519.SH", df, df.index[-1])
        assert not any("ST" in x for x in verdict.failures)

    def test_st_heuristic_is_point_in_time(self) -> None:
        # 早段平静（近 60 日无 >5.5% 波动 -> 疑似 ST），晚段出现 >5.5% 大波动。
        calm = [10.0] * 120
        violent = spike([10.0 + i * 0.02 for i in range(120)], -40, 0.07)
        df = make_series(calm + violent, start="2013-01-05")
        early = df.index[150]  # 大波动（全局 index 200）还没进入近 60 日窗口
        late = df.index[-1]
        f = UniverseFilter()
        v_early = f.judge("600519.SH", df.iloc[:151], early)
        v_late = f.judge("600519.SH", df, late)
        assert any("ST" in x for x in v_early.failures)
        assert not any("ST" in x for x in v_late.failures)

    def test_suspension_gap_fails(self) -> None:
        df = make_series(trend_series(), start="2013-01-05")
        as_of = df.index[-1]
        short = df.iloc[:-8]  # 最近 bar 距 as_of 8 个自然日
        verdict = UniverseFilter().judge("600519.SH", short, as_of)
        assert any("停牌" in x for x in verdict.failures)

    def test_long_suspension_window_fails(self) -> None:
        # 最后 bar 就是 as_of，但近 30 日窗口只有 ~2 根 bar。
        df = _norm_df(
            make_bars(
                [[10.0, 10.0, 10.0, 10.0, 2e7]] * 40
                + [None] * 25
                + [[10.5, 10.5, 10.5, 10.5, 2e7]],
                start="2013-01-05",
            )
        )
        verdict = UniverseFilter().judge("600519.SH", df, df.index[-1])
        assert any("30 日" in x and "bar" in x for x in verdict.failures)

    def test_as_of_slices_future_data(self) -> None:
        # judge 内部做 <= as_of 防御截断：传全量 df + 中间 as_of，仍只看 <= as_of。
        df = make_series(trend_series(), start="2013-01-05")
        as_of = df.index[700]
        verdict = UniverseFilter().judge("600519.SH", df, as_of)
        assert verdict.checks["earliest_bar_date"] == str(
            pd.Timestamp(df.index[0]).date()
        )


# ── sample_points ─────────────────────────────────────────────────────────────
class TestSamplePoints:
    def test_deterministic_with_seed(self) -> None:
        u = dense_universe()
        a = sample_points(u, "2017-01-01", "2019-12-31", 3, seed=42)
        b = sample_points(u, "2017-01-01", "2019-12-31", 3, seed=42)
        assert a == b
        assert a == sorted(a, key=lambda p: (p.date, p.symbol))

    def test_layers_filled(self) -> None:
        u = dense_universe()
        pts = sample_points(u, "2017-01-01", "2019-12-31", 3, seed=3)
        counts = Counter(p.regime for p in pts)
        assert counts == {REGIME_TREND: 3, REGIME_RANGE: 3, REGIME_BEAR: 3}
        assert all(p.regime != REGIME_UNKNOWN for p in pts)

    def test_spacing_within_stratum(self) -> None:
        u = dense_universe()
        pts = sample_points(u, "2017-01-01", "2019-12-31", 3, seed=7)
        calendar = make_series(trend_series()).index  # 所有合成标的共享该交易日历
        by_regime: dict[str, list] = {}
        for p in pts:
            by_regime.setdefault(p.regime, []).append(p)
        assert set(by_regime) == {REGIME_TREND, REGIME_RANGE, REGIME_BEAR}
        for regime, group in by_regime.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    d = trading_day_distance(
                        calendar,
                        pd.Timestamp(group[i].date),
                        pd.Timestamp(group[j].date),
                    )
                    assert d >= 60, (regime, group[i], group[j], d)

    def test_unknown_regime_not_sampled(self) -> None:
        df = make_series(unknown_series())
        # 只取回落段尾部（确认全是未知），整个池进不了任何层。
        pts = sample_points({"600999.SH": df}, df.index[-20], df.index[-1], 5, seed=1)
        assert pts == []

    def test_market_bars_drives_regime_labels(self) -> None:
        """Pilot 实测修正：regime 用大盘判（手册环①），不是个股。

        个股全程强劲上升（个股自身 regime=趋势），但传大盘序列后，点标签必须
        反映大盘状态——对每个点，大盘在决策日的 regime == 点标签。
        """
        u = {"000001.SZ": make_series(trend_series())}  # 个股上升
        market = make_series(range_series())["close"]  # 大盘 震荡→熊市→震荡
        pts = sample_points(
            u, "2015-06-01", "2017-03-01", 4, seed=5, market_bars=market
        )
        assert pts, "大盘震荡/熊市段应能抽到点"
        for p in pts:
            assert regime_stratum(market, p.date) == p.regime
        # 个股自身是上升趋势，若没传 market_bars 会全标"趋势"——大盘震荡段不应有趋势标签。
        assert all(p.regime != REGIME_TREND for p in pts)

    def test_no_points_after_delisting(self) -> None:
        full = make_series(trend_series(), start="2015-01-05")
        live = full.iloc[:800]  # 800 根后无数据（退市/长期停牌）
        u = {"600001.SH": live, "600002.SH": full}
        pts = sample_points(u, "2017-01-01", "2019-12-31", 3, seed=11)
        assert pts
        last_live = pd.Timestamp(live.index[-1])
        assert all(
            p.symbol != "600001.SH" or pd.Timestamp(p.date) <= last_live for p in pts
        )

    def test_universe_filter_applied(self) -> None:
        good = make_series(trend_series())  # 有 +6% 日 -> 非 ST
        bad = make_series(list(linspace(5, 11, 1200)))  # 无大波动 -> 疑似 ST
        u = {"600001.SH": good, "600002.SH": bad}
        window_start = good.index[-60]  # 只取最后 60 根：good 已过 +6% 日，bad 仍平静
        f = UniverseFilter()
        pts = sample_points(
            u, window_start, good.index[-1], 3, seed=9, universe_filter=f
        )
        assert pts and all(p.symbol == "600001.SH" for p in pts)
        # 关掉 ST 判定后 bad 能过宇宙过滤 —— 证明是 ST 近似在排除它。
        assert any("ST" in x for x in f.judge("600002.SH", bad, bad.index[-1]).failures)
        assert (
            UniverseFilter(st_check=lambda df: False)
            .judge("600002.SH", bad, bad.index[-1])
            .eligible
        )

    def test_sparse_pool_returns_fewer(self) -> None:
        df = make_series(trend_series(n=400), start="2015-01-05")
        pts = sample_points({"600001.SH": df}, df.index[200], df.index[-1], 10, seed=1)
        assert 0 < len(pts) <= 5
        assert all(p.regime == REGIME_TREND for p in pts)

    def test_invalid_args(self) -> None:
        df = make_series(trend_series())
        u = {"600001.SH": df}
        with pytest.raises(ValueError):
            sample_points(u, "2018-01-01", "2019-01-01", 0)
        with pytest.raises(ValueError):
            sample_points(u, "2018-01-01", "2019-01-01", 3, min_spacing=0)
        with pytest.raises(ValueError):
            sample_points(u, "2019-01-01", "2018-01-01", 3)


# ── compute_snapshot / build_data_pack ────────────────────────────────────────
class TestSnapshot:
    def test_flat_series_values(self) -> None:
        df = make_series([10.0] * 220, start="2015-01-05")
        s = compute_snapshot(df)
        assert s["close"] == pytest.approx(10.0)
        assert s["ma20"] == pytest.approx(10.0)
        assert s["ma60"] == pytest.approx(10.0)
        assert s["ma200"] == pytest.approx(10.0)
        assert s["ma60_up"] is False
        assert s["rsi"] == pytest.approx(100.0)  # 无涨跌 -> daily_check 口径
        assert s["vol_ratio"] == pytest.approx(1.0)
        assert s["new_high20"] is False
        assert s["ret20"] == pytest.approx(0.0)

    def test_matches_daily_check_compute_state(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "export_trade_data"))
        try:
            import daily_check
        except ImportError:
            pytest.skip("daily_check 不可导入")
        for closes in (trend_series(), bear_series(), range_series(), unknown_series()):
            df = make_series(closes, start="2015-01-05")
            expected = daily_check.compute_state(df)
            got = compute_snapshot(df)
            assert set(got) == set(expected)
            for key in expected:
                if isinstance(expected[key], bool):
                    assert got[key] == expected[key], key
                else:
                    assert got[key] == pytest.approx(expected[key]), key


class TestBuildDataPack:
    def test_offline_pack_json_round_trip(self) -> None:
        df = make_series(trend_series(), start="2013-01-05")
        pack = build_data_pack("600519.SH", df.index[-1], frame=df, lookback=len(df))
        assert pack["ok"] is True
        assert pack["source"] == "injected"
        assert pack["n_bars"] == len(df)
        assert pack["regime"] == regime_at(df["close"], df.index[-1])
        assert pack["snapshot"]["close"] == pytest.approx(float(df["close"].iloc[-1]))
        assert json.loads(json.dumps(pack)) == pack

    def test_only_uses_data_up_to_as_of(self) -> None:
        closes = list(linspace(5, 11, 300)) + list(linspace(11, 6, 60))
        df = make_series(closes, start="2015-01-05")
        as_of = df.index[290]
        pack = build_data_pack("600519.SH", as_of, frame=df)
        assert pack["ok"] is True
        assert pack["bars"][-1]["date"] == str(pd.Timestamp(as_of).date())
        assert pack["n_bars"] == 291
        assert pack["regime"] == REGIME_TREND  # 用 <=as_of 数据判，不看后面的大跌
        assert pack["snapshot"]["close"] == pytest.approx(float(df["close"].iloc[290]))

    def test_insufficient_bars_returns_error_pack(self) -> None:
        df = make_series(list(linspace(5, 11, 100)), start="2015-01-05")
        pack = build_data_pack("600519.SH", df.index[-1], frame=df)
        assert pack["ok"] is False
        assert "MA200" in pack["reason"]
