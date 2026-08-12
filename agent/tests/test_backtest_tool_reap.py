"""Tests for the C′ trusted-write reaper in backtest_tool (spec §9.12).

On the agent search path the agent-running subprocess only RECORDS the trial
record and OOS-isolation metadata — it must not write the protected ledger or
holdout. ``_reap_protected_writes`` runs in the trusted server process after the
subprocess exits to: append the trial ledger (first-trial boundary lock, §9.9),
persist the OOS holdout, and compute the DSR on the now-complete ledger, then
merge the results back into the run card.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.tools.backtest_tool import _reap_protected_writes


def _write_run(tmp_path: Path, card: dict) -> Path:
    """Build a run dir under the runtime root's runs/ with a run card."""
    run_dir = tmp_path / "home" / "runs" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_card.json").write_text(
        json.dumps(card, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def _write_equity(run_dir: Path, values=(1_000_000.0, 1_010_000.0, 1_005_000.0, 1_020_000.0)) -> None:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    pd.DataFrame({"equity": list(values)}).to_csv(artifacts / "equity.csv", index=False)


def _base_card() -> dict:
    return {
        "schema_version": "0.3",
        "metrics": {"ledger_status": "server_pending", "sharpe": 1.1, "trade_count": 5},
        "backtest": {"valid_end": "2024-06-30", "interval": "1D", "start_date": "2024-01-01"},
        "data_sources": ["tushare"],
        "_trial_record": {
            "ts": "2026-08-12T00:00:00Z",
            "search_id": "subprocess-claim",  # server must override this
            "strategy_hash": "abc",
            "config_hash": "def",
            "run_dir": "run_1",
            "interval": "1D",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "train_end": "2024-03-31",
            "valid_end": "2024-06-30",
            "sharpe": 1.1,
            "valid_sharpe": 0.9,
            "n_trades": 5,
            "exit_reason_counts": {"signal": 5},
        },
    }


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "home"))
    from src.config.accessor import reset_env_config

    reset_env_config()
    return tmp_path / "home"


def test_reap_appends_ledger_and_computes_dsr(home, tmp_path, monkeypatch):
    """Server reaps the trial record: ledger appended under the SERVER's search
    id (not the subprocess's claim), DSR computed, card rewritten."""
    import backtest.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "persist_test_holdout",
        lambda config, run_id: {"holdout_status": "persisted"},
    )
    run_dir = _write_run(tmp_path, _base_card())
    _write_equity(run_dir)

    _reap_protected_writes(run_dir, {"valid_end": "2024-06-30"}, "server-sess-1")

    # Ledger written under the runtime root, keyed by the SERVER search id.
    from backtest.trials import read_trials

    trials = read_trials(root=home)
    assert len(trials) == 1
    assert trials[0]["search_id"] == "server-sess-1"  # server-authoritative
    assert trials[0]["valid_sharpe"] == 0.9

    # Card rewritten: handoff stripped, status flipped, DSR mounted.
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    assert "_trial_record" not in card
    assert card["metrics"]["ledger_status"] == "server_written"
    assert card["dsr"]["search_id"] == "server-sess-1"


def test_reap_persists_holdout_via_server(home, tmp_path, monkeypatch):
    """A server_pending isolation block is persisted by the server and its
    metadata merged back into the card's data_isolation."""
    captured: dict = {}

    import backtest.runner as runner_mod

    def fake_persist(config, run_id):
        captured["run_id"] = run_id
        return {
            "holdout_status": "persisted",
            "symbols": ["000001.SZ"],
            "test_row_counts": {"000001.SZ": 30},
        }

    monkeypatch.setattr(runner_mod, "persist_test_holdout", fake_persist)

    card = _base_card()
    card["data_isolation"] = {
        "enabled": True,
        "boundary": "2024-06-30",
        "holdout_path": "oos_holdout/run_1",
        "holdout_status": "server_pending",
        "symbols": ["000001.SZ"],
        "test_row_counts": {"000001.SZ": 30},
    }
    run_dir = _write_run(tmp_path, card)
    _write_equity(run_dir)

    _reap_protected_writes(run_dir, {"valid_end": "2024-06-30"}, "server-sess-1")

    assert captured["run_id"] == "run_1"
    out = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    assert out["data_isolation"]["holdout_status"] == "persisted"


def test_reap_no_search_marker_is_noop(home, tmp_path, monkeypatch):
    """A non-search run (no ledger_status server_pending, no pending holdout)
    is left entirely untouched."""
    called = []
    import backtest.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "persist_test_holdout",
        lambda *a, **k: called.append(1) or {},
    )
    card = {"schema_version": "0.3", "metrics": {"sharpe": 1.0}}
    run_dir = _write_run(tmp_path, card)

    _reap_protected_writes(run_dir, {}, None)

    assert called == []
    # Card unchanged (no dsr, no ledger_status munging).
    out = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    assert "dsr" not in out
    assert out["metrics"] == {"sharpe": 1.0}


def test_reap_ledger_failure_marks_dsr_non_authoritative(home, tmp_path, monkeypatch):
    """Fail-closed (review H6): if the server-side ledger append fails, the DSR
    is still computed but forced non-authoritative, and the run does not crash."""
    import backtest.runner as runner_mod

    monkeypatch.setattr(
        runner_mod, "persist_test_holdout", lambda c, r: {"holdout_status": "persisted"}
    )
    import backtest.trials as trials_mod

    def boom(record, root=None):
        raise OSError("disk full")

    monkeypatch.setattr(trials_mod, "append_trial_with_sanction", boom)

    run_dir = _write_run(tmp_path, _base_card())
    _write_equity(run_dir)

    # Must not raise.
    _reap_protected_writes(run_dir, {"valid_end": "2024-06-30"}, "server-sess-1")

    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    assert card["metrics"]["ledger_status"] == "failed"
    assert card["dsr"]["authoritative"] is False
    assert "ledger write failed" in card["dsr"]["reason"]
