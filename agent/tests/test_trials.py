"""Tests for the trial ledger (D2 search accounting)."""

from __future__ import annotations

import json
import threading

import pandas as pd

from backtest.trials import (
    append_trial,
    config_hash,
    ledger_path,
    read_trials,
    read_trials_verbose,
    sanitize_search_id,
)


def _rec(search_id="s1", strategy_hash="aaa", sharpe=1.0, **kw):
    rec = {
        "ts": "2026-08-09T12:00:00Z",
        "search_id": search_id,
        "strategy_hash": strategy_hash,
        "config_hash": "cfg",
        "run_dir": "/tmp/run",
        "sharpe": sharpe,
        "n_trades": 42,
    }
    rec.update(kw)
    return rec


def test_sanitize_search_id_charset():
    assert sanitize_search_id("agent-run-abc:01.2_x") == "agent-run-abc:01.2_x"
    assert sanitize_search_id("bad id") is None  # space
    assert sanitize_search_id("a/b") is None  # path sep
    assert sanitize_search_id("x" * 65) is None  # too long
    assert sanitize_search_id("ctrl\nchar") is None
    assert sanitize_search_id("") is None
    assert sanitize_search_id(None) is None
    assert sanitize_search_id(123) is None


def test_ledger_path_under_root(tmp_path):
    assert ledger_path(tmp_path) == tmp_path / "backtest_trials.jsonl"


def test_append_and_read_roundtrip(tmp_path):
    append_trial(_rec("s1", "aaa", 1.0), root=tmp_path)
    append_trial(_rec("s1", "bbb", 2.0), root=tmp_path)
    append_trial(_rec("s2", "ccc", 3.0), root=tmp_path)

    all_rows = read_trials(root=tmp_path)
    assert len(all_rows) == 3

    s1 = read_trials(search_id="s1", root=tmp_path)
    assert len(s1) == 2
    assert {r["strategy_hash"] for r in s1} == {"aaa", "bbb"}


def test_read_filters_by_search_id_and_not_strategy(tmp_path):
    append_trial(_rec("s1", "aaa"), root=tmp_path)
    append_trial(_rec("s1", "aaa"), root=tmp_path)
    append_trial(_rec("s1", "bbb"), root=tmp_path)
    rows = read_trials(search_id="s1", root=tmp_path)
    # Same search, different strategy hashes do NOT mix into separate groups.
    assert len(rows) == 3


def test_tolerant_reader_skips_corrupt_lines(tmp_path):
    append_trial(_rec("s1", "aaa"), root=tmp_path)
    path = ledger_path(tmp_path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("this is not json\n")
        fh.write('{"broken": \n')
    append_trial(_rec("s1", "bbb"), root=tmp_path)

    rows, corrupt = read_trials_verbose(search_id="s1", root=tmp_path)
    assert len(rows) == 2  # intact rows still read
    assert corrupt == 2  # two bad lines counted


def test_read_missing_file_returns_empty(tmp_path):
    assert read_trials(root=tmp_path) == []


def test_concurrent_appends_no_interleaving(tmp_path):
    def worker(n):
        for i in range(20):
            append_trial(_rec(f"s{n}", f"h{n}-{i}"), root=tmp_path)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    path = ledger_path(tmp_path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 80
    for ln in lines:  # every line is intact JSON (no interleaved half-lines)
        json.loads(ln)


def test_config_hash_matches_run_card(tmp_path):
    from backtest.run_card import _file_hash, _json_hash

    cfg = {"codes": ["AAA"], "start_date": "2024-01-01", "end_date": "2024-06-01"}
    # In-memory source.
    assert config_hash(cfg, None) == _json_hash(cfg)
    # File source when present.
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    assert config_hash(cfg, cfg_file) == _file_hash(cfg_file)


# --- P0 (spec §9.9): valid_end cross-trial boundary lock ---


def test_sanction_first_trial_sets_authority_no_mismatch(tmp_path):
    from backtest.trials import append_trial_with_sanction

    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=1.0), root=tmp_path)
    rows = read_trials(search_id="s1", root=tmp_path)
    assert len(rows) == 1
    # The first trial IS the authority — never self-flagged.
    assert "valid_end_mismatch" not in rows[0]
    assert "sanctioned_valid_end" not in rows[0]


def test_sanction_same_boundary_all_accepted(tmp_path):
    from backtest.trials import append_trial_with_sanction

    for sharpe in (1.0, 1.5, 0.8):
        append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=sharpe), root=tmp_path)
    rows = read_trials(search_id="s1", root=tmp_path)
    assert len(rows) == 3
    assert all("valid_end_mismatch" not in r for r in rows)


def test_sanction_drift_flagged_and_still_recorded(tmp_path):
    from backtest.trials import append_trial_with_sanction

    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=1.0), root=tmp_path)
    # Agent shifts the OOS boundary to pull more good performance into valid.
    append_trial_with_sanction(_rec("s1", valid_end="2024-12-31", valid_sharpe=2.5), root=tmp_path)
    rows = read_trials(search_id="s1", root=tmp_path)
    assert len(rows) == 2  # honest record — the drift trial is still written
    assert "valid_end_mismatch" not in rows[0]
    assert rows[1]["valid_end_mismatch"] is True
    assert rows[1]["sanctioned_valid_end"] == "2024-06-30"
    assert rows[1]["valid_end"] == "2024-12-31"  # original value kept


def test_sanction_authority_is_first_non_none(tmp_path):
    from backtest.trials import append_trial_with_sanction

    # First trial has no OOS split (valid_end None) — not the authority.
    append_trial_with_sanction(_rec("s1", valid_end=None, valid_sharpe=None), root=tmp_path)
    # Second trial sets the boundary → becomes authority.
    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=1.0), root=tmp_path)
    # Third trial with same boundary → accepted (authority came from 2nd, not 1st).
    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=0.9), root=tmp_path)
    rows = read_trials(search_id="s1", root=tmp_path)
    assert all("valid_end_mismatch" not in r for r in rows)


def test_sanction_scoped_per_search_id(tmp_path):
    from backtest.trials import append_trial_with_sanction

    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=1.0), root=tmp_path)
    # A DIFFERENT search can set its own boundary — not affected by s1's authority.
    append_trial_with_sanction(_rec("s2", valid_end="2024-12-31", valid_sharpe=1.2), root=tmp_path)
    s2 = read_trials(search_id="s2", root=tmp_path)
    assert "valid_end_mismatch" not in s2[0]


def test_sanction_concurrent_single_authority(tmp_path):
    from backtest.trials import append_trial_with_sanction

    # Two threads race with different boundaries; the lock must make exactly one
    # the authority, the other flagged (no both-become-authority race).
    def worker(boundary):
        append_trial_with_sanction(_rec("s1", valid_end=boundary, valid_sharpe=1.0), root=tmp_path)

    threads = [
        threading.Thread(target=worker, args=("2024-06-30",)),
        threading.Thread(target=worker, args=("2024-12-31",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = read_trials(search_id="s1", root=tmp_path)
    assert len(rows) == 2
    mismatched = [r for r in rows if r.get("valid_end_mismatch")]
    accepted = [r for r in rows if not r.get("valid_end_mismatch")]
    assert len(accepted) == 1  # exactly one authority
    assert len(mismatched) == 1  # exactly one flagged
    assert mismatched[0]["sanctioned_valid_end"] == accepted[0]["valid_end"]


def test_run_dsr_excludes_boundary_drift(tmp_path, monkeypatch):
    from backtest.trials import append_trial_with_sanction, run_dsr

    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=1.0), root=tmp_path)
    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=1.1), root=tmp_path)
    # Drift trial with an inflated valid_sharpe that must NOT reach the DSR pool.
    append_trial_with_sanction(_rec("s1", valid_end="2024-12-31", valid_sharpe=99.0), root=tmp_path)

    captured = {}

    def fake_dsr(trial_scores, daily_returns, bars_per_year, mode):
        captured["trial_scores"] = trial_scores
        return {"DSR": 0.5, "verdict": "weak"}

    monkeypatch.setattr("backtest.validation.deflated_sharpe_ratio", fake_dsr)
    equity = pd.Series([100.0, 101.0, 102.0, 101.5])
    out = run_dsr("s1", equity, 252, root=tmp_path)

    # The 99.0 drift trial is excluded; only the two sanctioned scores pool.
    assert captured["trial_scores"] == [1.0, 1.1]
    assert out["n_excluded_boundary_drift"] == 1


def test_run_dsr_no_drift_no_exclusion_key(tmp_path, monkeypatch):
    from backtest.trials import append_trial_with_sanction, run_dsr

    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=1.0), root=tmp_path)
    append_trial_with_sanction(_rec("s1", valid_end="2024-06-30", valid_sharpe=0.9), root=tmp_path)

    monkeypatch.setattr(
        "backtest.validation.deflated_sharpe_ratio",
        lambda trial_scores, daily_returns, bars_per_year, mode: {"DSR": 0.9, "verdict": "significant"},
    )
    equity = pd.Series([100.0, 101.0, 102.0])
    out = run_dsr("s1", equity, 252, root=tmp_path)
    assert "n_excluded_boundary_drift" not in out
