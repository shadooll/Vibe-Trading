"""Tests for Phase 3a OOS test-segment physical isolation (spec §8.1).

On the agent search path with a three-way split, the runner persists the rows
past ``valid_end`` to a holdout dir under the real runtime root
(``oos_holdout/<run_id>/``) and records a ``data_isolation`` block on the run
card. The in-memory data_map fed to the engine stays whole (test metrics still
computed); only the agent-visible pack is isolated. Non-agent / no-split runs
are untouched.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtest.runner import _isolate_test_segment


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


def _set_home_and_search(tmp_path, monkeypatch, search_id):
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "home"))
    if search_id is None:
        monkeypatch.delenv("VIBE_TRADING_SEARCH_ID", raising=False)
    else:
        monkeypatch.setenv("VIBE_TRADING_SEARCH_ID", search_id)
    from src.config.accessor import reset_env_config
    reset_env_config()


def test_isolation_splits_and_records_on_agent_path(tmp_path, monkeypatch):
    _set_home_and_search(tmp_path, monkeypatch, "agent-search-1")
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    frame = _frame()
    data_map = {"000001.SZ": frame}
    config = {"valid_end": "2023-03-15", "end_date": "2023-04-30"}

    block = _isolate_test_segment(data_map, config, run_dir)

    assert block is not None
    assert block["enabled"] is True
    assert block["boundary"] == "2023-03-15"
    assert block["symbols"] == ["000001.SZ"]
    assert block["unblinded_at"]  # non-empty ISO timestamp
    # Holdout file written under the REAL runtime root, outside the sandbox view.
    holdout = tmp_path / "home" / "oos_holdout" / "run_x" / "ohlcv_test_000001.SZ.csv"
    assert holdout.exists()
    held = pd.read_csv(holdout, index_col=0, parse_dates=True)
    assert held.index.min() > pd.Timestamp("2023-03-15")
    assert held.index.max() == frame.index.max()
    # The in-memory data_map is NOT mutated (engine still gets the full series).
    assert len(data_map["000001.SZ"]) == len(frame)


def test_isolation_skipped_without_search_marker(tmp_path, monkeypatch):
    _set_home_and_search(tmp_path, monkeypatch, None)
    run_dir = tmp_path / "run_y"
    run_dir.mkdir()
    block = _isolate_test_segment({"000001.SZ": _frame()}, {"valid_end": "2023-03-15"}, run_dir)
    assert block is None
    assert not (tmp_path / "home" / "oos_holdout").exists()


def test_isolation_skipped_without_split(tmp_path, monkeypatch):
    _set_home_and_search(tmp_path, monkeypatch, "agent-search-1")
    run_dir = tmp_path / "run_z"
    run_dir.mkdir()
    block = _isolate_test_segment({"000001.SZ": _frame()}, {}, run_dir)
    assert block is None


def test_isolation_skipped_when_nothing_past_boundary(tmp_path, monkeypatch):
    _set_home_and_search(tmp_path, monkeypatch, "agent-search-1")
    run_dir = tmp_path / "run_w"
    run_dir.mkdir()
    # valid_end beyond the last bar -> no test segment to hold out.
    block = _isolate_test_segment({"000001.SZ": _frame()}, {"valid_end": "2023-12-31"}, run_dir)
    assert block is None


def test_isolation_fail_closed_on_unwritable_holdout(tmp_path, monkeypatch):
    _set_home_and_search(tmp_path, monkeypatch, "agent-search-1")
    run_dir = tmp_path / "run_v"
    run_dir.mkdir()
    # Point the runtime root at a path whose parent is a FILE so mkdir fails.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("VIBE_TRADING_HOME", str(blocker / "home"))
    from src.config.accessor import reset_env_config
    reset_env_config()
    with pytest.raises(OSError):
        _isolate_test_segment({"000001.SZ": _frame()}, {"valid_end": "2023-03-15"}, run_dir)


def test_run_card_mounts_data_isolation(tmp_path):
    from backtest.run_card import write_run_card
    metrics = {"final_value": 1100.0, "sharpe": 1.2, "trade_count": 40}
    config = {
        "codes": ["000001.SZ"],
        "_data_isolation": {
            "enabled": True,
            "boundary": "2023-03-15",
            "holdout_path": "oos_holdout/run_x",
            "symbols": ["000001.SZ"],
            "unblinded_at": "2026-08-11T00:00:00+00:00",
            "unblind_reason": "engine requires the test segment in-memory",
        },
    }
    card = write_run_card(tmp_path, config, metrics)
    assert card["schema_version"] == "0.3"
    assert card["data_isolation"]["enabled"] is True
    assert card["data_isolation"]["boundary"] == "2023-03-15"
    on_disk = json.loads((tmp_path / "run_card.json").read_text(encoding="utf-8"))
    assert on_disk["data_isolation"]["holdout_path"] == "oos_holdout/run_x"
    md = (tmp_path / "run_card.md").read_text(encoding="utf-8")
    assert "Data Isolation" in md
    assert "unblinded_at" in md


def test_run_card_no_isolation_block_when_absent(tmp_path):
    from backtest.run_card import write_run_card
    card = write_run_card(tmp_path, {"codes": ["000001.SZ"]}, {"sharpe": 1.0})
    assert "data_isolation" not in card
