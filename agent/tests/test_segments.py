"""Tests for the three-way split + unified score + trade floor (spec §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    MIN_TOTAL_TRADES,
    MIN_VALID_TRADES,
    calc_metrics,
    calc_segment_metrics,
    calc_unified_score,
    validation_floor,
)
from backtest.models import TradeRecord
from backtest.runner import BacktestConfigSchema


def _equity(n=300, start="2023-01-01", drift=0.001):
    idx = pd.date_range(start, periods=n, freq="D")
    rets = np.full(n, drift)
    eq = 1000.0 * np.cumprod(1 + rets)
    return pd.Series(eq, index=idx)


def _trade(entry, exit_, pnl=10.0):
    return TradeRecord(
        symbol="AAA", direction=1, entry_price=100.0, exit_price=101.0,
        entry_time=pd.Timestamp(entry), exit_time=pd.Timestamp(exit_),
        size=1.0, leverage=1.0, pnl=pnl, pnl_pct=1.0, exit_reason="signal",
        holding_bars=1, commission=0.1, entry_margin=100.0, exit_margin=101.0,
    )


def _many_trades(n, idx):
    step = max(1, len(idx) // (n + 1))
    return [_trade(idx[i], idx[min(i + step, len(idx) - 1)]) for i in range(0, len(idx) - 1, step)][:n]


# ─── Schema validation ───


def _cfg(**kw):
    base = {
        "codes": ["AAA"], "start_date": "2023-01-01", "end_date": "2023-12-31",
        "source": "tushare",
    }
    base.update(kw)
    return base


def test_schema_accepts_valid_split():
    s = BacktestConfigSchema(**_cfg(train_end="2023-06-01", valid_end="2023-09-01"))
    assert s.train_end == "2023-06-01"
    assert s.valid_end == "2023-09-01"


def test_schema_rejects_unordered_split():
    with pytest.raises(Exception):
        BacktestConfigSchema(**_cfg(train_end="2023-09-01", valid_end="2023-06-01"))
    with pytest.raises(Exception):
        BacktestConfigSchema(**_cfg(train_end="2023-01-01", valid_end="2023-06-01"))  # te == start
    with pytest.raises(Exception):
        BacktestConfigSchema(**_cfg(train_end="2023-06-01", valid_end="2024-01-01"))  # ve > end


def test_schema_requires_both_boundaries():
    with pytest.raises(Exception):
        BacktestConfigSchema(**_cfg(train_end="2023-06-01"))
    with pytest.raises(Exception):
        BacktestConfigSchema(**_cfg(valid_end="2023-09-01"))


def test_schema_allows_valid_end_equal_end():
    # Empty test segment is allowed (marked insufficient="test", not rejected).
    s = BacktestConfigSchema(**_cfg(train_end="2023-06-01", valid_end="2023-12-31"))
    assert s.valid_end == "2023-12-31"


# ─── Segment metrics ───


def test_segments_split_by_index():
    eq = _equity(300)
    trades = _many_trades(60, eq.index)
    seg = calc_segment_metrics(eq, trades, "2023-05-01", "2023-08-01")
    assert set(seg) == {"train", "valid", "test"}
    for name in ("train", "valid", "test"):
        assert seg[name]["sharpe"] is not None
        assert seg[name]["n_trades"] >= 0


def test_segment_trade_attribution_by_exit():
    eq = _equity(300)
    te = pd.Timestamp("2023-05-01")
    ve = pd.Timestamp("2023-08-01")
    # One trade that ENTERS in train but EXITS in valid.
    cross = _trade("2023-04-15", "2023-06-15")
    seg = calc_segment_metrics(eq, [cross], te, ve)
    assert seg["train"]["n_trades"] == 0  # exit not in train
    assert seg["valid"]["n_trades"] == 1  # exit lands in valid


def test_unified_score_formula():
    segments = {
        "train": {"sharpe": 2.0, "n_trades": 100},
        "valid": {"sharpe": 1.0, "n_trades": 100},
        "test": {"sharpe": 0.5, "n_trades": 10},
    }
    # valid − 0.5·max(0, train − valid) = 1.0 − 0.5·(2.0−1.0) = 0.5
    assert calc_unified_score(segments) == pytest.approx(0.5)


def test_unified_score_no_penalty_when_valid_beats_train():
    segments = {"train": {"sharpe": 0.5}, "valid": {"sharpe": 1.5}, "test": {"sharpe": 1.0}}
    assert calc_unified_score(segments) == pytest.approx(1.5)


def test_unified_score_null_without_split():
    assert calc_unified_score({"train": {}, "valid": {}}) is None


def test_floor_valid_too_few_trades():
    seg = {
        "train": {"sharpe": 1.0, "n_trades": 100},
        "valid": {"sharpe": 1.0, "n_trades": MIN_VALID_TRADES - 1},
        "test": {"sharpe": 1.0, "n_trades": 10},
    }
    assert validation_floor(seg, 200) == "valid"


def test_floor_overall_too_few_trades():
    seg = {
        "train": {"sharpe": 1.0, "n_trades": 60},
        "valid": {"sharpe": 1.0, "n_trades": MIN_VALID_TRADES},
        "test": {"sharpe": 1.0, "n_trades": 5},
    }
    assert validation_floor(seg, MIN_TOTAL_TRADES - 1) == "overall"


def test_floor_empty_test_segment():
    seg = {
        "train": {"sharpe": 1.0, "n_trades": 60},
        "valid": {"sharpe": 1.0, "n_trades": MIN_VALID_TRADES},
        "test": {"sharpe": None, "n_trades": 0},
    }
    assert validation_floor(seg, 200) == "test"


# ─── calc_metrics integration ───


def test_calc_metrics_adds_segments_with_split():
    eq = _equity(300)
    trades = _many_trades(80, eq.index)
    m = calc_metrics(eq, trades, 1000.0, 252, train_end="2023-05-01", valid_end="2023-08-01")
    assert "segments" in m
    assert "unified_score" in m


def test_calc_metrics_legacy_without_split_is_unchanged():
    eq = _equity(300)
    trades = _many_trades(80, eq.index)
    m = calc_metrics(eq, trades, 1000.0, 252)
    assert "segments" not in m
    assert "unified_score" not in m
    assert "validation_insufficient" not in m


def test_calc_metrics_valid_floor_voids_unified_score():
    eq = _equity(300)
    # Few trades in valid → floor hit → unified_score voided.
    idx = eq.index
    trades = [_trade(idx[0], idx[1]), _trade(idx[10], idx[20])]
    m = calc_metrics(eq, trades, 1000.0, 252, train_end="2023-05-01", valid_end="2023-08-01")
    assert m.get("validation_insufficient") in ("valid", "overall")
    assert m["unified_score"] is None
