"""Tests for the 2b data boundary (spec §4.4): as-of isolation on the agent path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engines.china_a import ChinaAEngine


def _frame(n=120, start="2023-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    close = 10 + 2 * np.sin(np.linspace(0, 8 * np.pi, n))
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1_000_000.0,
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
    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            fast = df["close"].rolling(5).mean()
            slow = df["close"].rolling(20).mean()
            out[sym] = (fast > slow).astype(float).where(slow.notna(), 0.0)
        return out


def _run(config_extra, tmp_path, monkeypatch, search_id=None):
    # Point the runtime root at tmp_path so the ledger never touches the real
    # ~/.vibe-trading, and set/clear the search marker.
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "home"))
    if search_id is None:
        monkeypatch.delenv("VIBE_TRADING_SEARCH_ID", raising=False)
    else:
        monkeypatch.setenv("VIBE_TRADING_SEARCH_ID", search_id)
    from src.config.accessor import reset_env_config
    reset_env_config()

    engine = ChinaAEngine({"initial_cash": 100_000})
    config = {
        "codes": ["000001.SZ"],
        "start_date": "2023-01-01",
        "end_date": "2023-04-30",
        "causality_check": "off",
    }
    config.update(config_extra)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    m = engine.run_backtest(config, _Loader(_frame()), _CrossEngine(), run_dir)
    return m, run_dir


def _read_ohlcv(run_dir):
    return pd.read_csv(run_dir / "artifacts" / "ohlcv_000001.SZ.csv", index_col=0, parse_dates=True)


def test_agent_path_trims_ohlcv_beyond_valid_end(tmp_path, monkeypatch):
    _, run_dir = _run(
        {"train_end": "2023-02-15", "valid_end": "2023-03-15"},
        tmp_path, monkeypatch, search_id="agent-search-1",
    )
    ohlcv = _read_ohlcv(run_dir)
    assert ohlcv.index.max() <= pd.Timestamp("2023-03-15")
    # Some rows were actually trimmed (test segment existed).
    assert len(ohlcv) < 120


def test_non_agent_path_keeps_full_ohlcv(tmp_path, monkeypatch):
    # No search marker → CLI/legacy behaviour: full OHLCV preserved.
    _, run_dir = _run(
        {"train_end": "2023-02-15", "valid_end": "2023-03-15"},
        tmp_path, monkeypatch, search_id=None,
    )
    ohlcv = _read_ohlcv(run_dir)
    assert len(ohlcv) == 120
    assert ohlcv.index.max() > pd.Timestamp("2023-03-15")


def test_no_split_keeps_full_ohlcv(tmp_path, monkeypatch):
    _, run_dir = _run({}, tmp_path, monkeypatch, search_id="agent-search-1")
    ohlcv = _read_ohlcv(run_dir)
    assert len(ohlcv) == 120


def test_result_artifacts_stay_whole_on_agent_path(tmp_path, monkeypatch):
    # The engine still consumed the full data → test-segment metrics exist;
    # only the raw OHLCV the agent could peek at is clipped.
    m, run_dir = _run(
        {"train_end": "2023-02-15", "valid_end": "2023-03-15"},
        tmp_path, monkeypatch, search_id="agent-search-1",
    )
    assert m["segments"]["test"]["sharpe"] is not None  # test segment computed
    equity = pd.read_csv(run_dir / "artifacts" / "equity.csv", index_col=0, parse_dates=True)
    assert equity.index.max() > pd.Timestamp("2023-03-15")  # result not clipped


def test_enrichment_as_of_prefers_valid_end():
    from backtest.engines.base import _enrichment_as_of
    assert _enrichment_as_of({"valid_end": "2023-03-15", "end_date": "2023-04-30"}) == "2023-03-15"
    assert _enrichment_as_of({"end_date": "2023-04-30"}) == "2023-04-30"
