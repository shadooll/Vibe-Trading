"""Tests for run_card nested-block mounting + NaN guard (spec §7, review H5)."""

from __future__ import annotations

import json

from backtest.run_card import write_run_card


def _base_metrics():
    return {
        "final_value": 1100.0,
        "total_return": 0.1,
        "sharpe": 1.2,
        "trade_count": 50,
    }


def test_nested_blocks_mounted(tmp_path):
    metrics = _base_metrics()
    metrics["segments"] = {
        "train": {"sharpe": 1.5, "total_return": 0.2, "max_drawdown": -0.1, "n_trades": 30},
        "valid": {"sharpe": 1.0, "total_return": 0.1, "max_drawdown": -0.05, "n_trades": 20},
        "test": {"sharpe": 0.5, "total_return": 0.05, "max_drawdown": -0.02, "n_trades": 5},
    }
    metrics["attribution"] = {"beta": 0.8, "alpha_annual": 0.05, "r_squared": 0.4}
    metrics["cost_sensitivity"] = {"authoritative": False, "1.0": {"total_return": 0.1}}
    metrics["dsr"] = {"DSR": 0.97, "verdict": "significant", "n_trials": 10}
    card = write_run_card(tmp_path, {"codes": ["AAA"]}, metrics)
    for key in ("segments", "attribution", "cost_sensitivity", "dsr"):
        assert key in card, key
    # And persisted to disk.
    on_disk = json.loads((tmp_path / "run_card.json").read_text(encoding="utf-8"))
    assert on_disk["dsr"]["verdict"] == "significant"
    assert on_disk["segments"]["valid"]["sharpe"] == 1.0


def test_schema_version_is_03(tmp_path):
    card = write_run_card(tmp_path, {"codes": ["AAA"]}, _base_metrics())
    assert card["schema_version"] == "0.3"


def test_nan_inf_in_nested_blocks_serialised_as_null(tmp_path):
    """NaN/Inf inside nested blocks must become null, NOT crash allow_nan=False."""
    metrics = _base_metrics()
    metrics["segments"] = {
        "train": {"sharpe": float("nan"), "total_return": float("inf"), "n_trades": 30},
        "valid": {"sharpe": float("-inf"), "total_return": 0.1, "n_trades": 20},
    }
    metrics["dsr"] = {"DSR": float("nan"), "verdict": "unavailable"}
    metrics["cost_sensitivity"] = {"authoritative": False, "5.0": {"total_return": float("-inf")}}
    metrics["attribution"] = None  # no benchmark → null block, skipped
    write_run_card(tmp_path, {"codes": ["AAA"]}, metrics)
    # Strict JSON written (no bare NaN/Infinity token).
    raw = (tmp_path / "run_card.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)  # raises if invalid
    assert parsed["segments"]["train"]["sharpe"] is None
    assert parsed["segments"]["train"]["total_return"] is None
    assert parsed["segments"]["valid"]["sharpe"] is None
    assert parsed["dsr"]["DSR"] is None
    assert parsed["cost_sensitivity"]["5.0"]["total_return"] is None
    # attribution None block is not mounted.
    assert "attribution" not in parsed


def test_markdown_escapes_html_in_agent_strings(tmp_path):
    metrics = _base_metrics()
    metrics["dsr"] = {"DSR": 0.9, "verdict": "weak", "reason": "<img src=x onerror=alert(1)>"}
    write_run_card(
        tmp_path,
        {"codes": ["<script>alert(1)</script>"]},
        metrics,
        data_sources=["<b>tushare</b>"],
    )
    md = (tmp_path / "run_card.md").read_text(encoding="utf-8")
    assert "<img" not in md
    assert "<script>" not in md
    assert "<b>" not in md
    assert "&lt;" in md  # escaped
