"""Tests for cost sensitivity (spec §6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engines.china_a import ChinaAEngine


def _frame(n=120, start="2023-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    # A sine so the strategy trades both ways.
    close = 10 + 2 * np.sin(np.linspace(0, 8 * np.pi, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
            "pre_close": np.roll(close, 1),
        },
        index=idx,
    )


class _Loader:
    def __init__(self, frame):
        self._frame = frame

    def fetch(self, *a, **k):
        return {"000001.SZ": self._frame.copy()}


class _CrossEngine:
    """Causal MA-cross signal: long when fast > slow, flat otherwise."""

    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            fast = df["close"].rolling(5).mean()
            slow = df["close"].rolling(20).mean()
            sig = (fast > slow).astype(float).where(slow.notna(), 0.0)
            out[sym] = sig
        return out


def _run(multipliers, tmp_path):
    engine = ChinaAEngine({"initial_cash": 100_000})
    config = {
        "codes": ["000001.SZ"],
        "start_date": "2023-01-01",
        "end_date": "2023-04-30",
        "causality_check": "off",  # small fixture; MA cross is causal anyway
    }
    if multipliers is not None:
        config["cost_sensitivity"] = {"multipliers": multipliers}
    return engine.run_backtest(config, _Loader(_frame()), _CrossEngine(), tmp_path)


def test_cost_sensitivity_present(tmp_path):
    m = _run([0.5, 1.0, 2.0, 5.0], tmp_path)
    cs = m.get("cost_sensitivity")
    assert cs is not None
    assert cs["authoritative"] is False
    for k in ("0.5", "1.0", "2.0", "5.0"):
        assert k in cs, cs.keys()
        assert "total_return" in cs[k]
        assert "n_trades" in cs[k]


def test_m1_matches_main_run(tmp_path):
    m = _run([1.0], tmp_path)
    cs = m["cost_sensitivity"]["1.0"]
    # Fresh instance at scale 1.0 reproduces the main run's numbers.
    assert cs["total_return"] == pytest.approx(m["total_return"], rel=1e-9)
    assert cs["n_trades"] == m["trade_count"]


def test_higher_cost_not_better(tmp_path):
    m = _run([1.0, 2.0, 5.0], tmp_path)
    cs = m["cost_sensitivity"]
    # Net return is monotonically non-increasing as cost scales up.
    assert cs["2.0"]["total_return"] <= cs["1.0"]["total_return"] + 1e-9
    assert cs["5.0"]["total_return"] <= cs["2.0"]["total_return"] + 1e-9
    # Commission itself grows with the multiplier.
    assert cs["5.0"]["total_commission"] >= cs["1.0"]["total_commission"]


def test_no_config_zero_overhead(tmp_path):
    m = _run(None, tmp_path)
    assert "cost_sensitivity" not in m


def test_artifacts_unaffected(tmp_path):
    import json as _json
    m = _run([1.0, 5.0], tmp_path)
    # run_card + metrics.csv describe the MAIN run, not a sensitivity tier.
    card = _json.loads((tmp_path / "run_card.json").read_text(encoding="utf-8"))
    assert card["metrics"]["total_return"] == pytest.approx(m["total_return"], rel=1e-9)


# ─── Crypto funding-path identity (review H4 explicit requirement) ───


def _crypto_frame(n=48, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="8h")  # funding slots at 00/08/16
    close = 100 + 5 * np.sin(np.linspace(0, 4 * np.pi, n))
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1_000.0,
        },
        index=idx,
    )


class _CryptoLoader:
    def __init__(self, frame):
        self._frame = frame

    def fetch(self, *a, **k):
        return {"BTC-USDT": self._frame.copy()}


def test_crypto_funding_m1_bit_identical(tmp_path):
    """m=1.0 must reproduce the main run on a funding-bearing crypto run.

    A fresh instance replays funding from clean dedup state, so the equity path
    (including funding deductions) matches the main run bit-for-bit; an in-place
    rerun would silently skip the replay (review H4).
    """
    from backtest.engines.crypto import CryptoEngine

    engine = CryptoEngine({
        "initial_cash": 100_000, "leverage": 2.0,
        "maker_rate": 0.0002, "taker_rate": 0.0005, "funding_rate": 0.0001,
    })
    config = {
        "codes": ["BTC-USDT"], "start_date": "2024-01-01", "end_date": "2024-01-20",
        "interval": "8h", "causality_check": "off",
        "cost_sensitivity": {"multipliers": [1.0]},
    }
    m = engine.run_backtest(config, _CryptoLoader(_crypto_frame()), _CrossEngine(), tmp_path)
    cs = m["cost_sensitivity"]["1.0"]
    assert cs["total_return"] == pytest.approx(m["total_return"], rel=1e-12)
    assert cs["n_trades"] == m["trade_count"]


# ─── HK market-kwarg identity (review: GlobalEquityEngine constructor arg) ───


class _HKLoader:
    def __init__(self, frame):
        self._frame = frame

    def fetch(self, *a, **k):
        return {"0700.HK": self._frame.copy()}


def test_hk_market_m1_bit_identical(tmp_path):
    """m=1.0 must reproduce an HK run whose costs come from the market kwarg.

    GlobalEquityEngine takes a positional ``market`` arg (default "us"); the
    runner builds HK runs with ``market="hk"``. A fresh sensitivity instance
    constructed as ``cls(config)`` silently defaults to "us" and drops the HK
    stamp-tax/levy/settlement stack — so m=1.0 would diverge from the main run.
    The fresh instance must inherit the live instance's market.
    """
    from backtest.engines.global_equity import GlobalEquityEngine

    engine = GlobalEquityEngine({"initial_cash": 1_000_000}, market="hk")
    config = {
        "codes": ["0700.HK"], "start_date": "2023-01-01", "end_date": "2023-04-30",
        "causality_check": "off",
        "cost_sensitivity": {"multipliers": [1.0, 2.0]},
    }
    m = engine.run_backtest(config, _HKLoader(_frame()), _CrossEngine(), tmp_path)
    cs = m["cost_sensitivity"]
    # m=1.0 reproduces the main run (HK cost stack intact on the fresh instance).
    assert cs["1.0"]["total_return"] == pytest.approx(m["total_return"], rel=1e-9)
    assert cs["1.0"]["n_trades"] == m["trade_count"]
    # HK run actually charged commission (stamp tax etc.) — non-zero, and grows with the multiplier.
    assert cs["1.0"]["total_commission"] > 0
    assert cs["2.0"]["total_commission"] >= cs["1.0"]["total_commission"]
