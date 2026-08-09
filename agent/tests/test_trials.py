"""Tests for the trial ledger (D2 search accounting)."""

from __future__ import annotations

import json
import os
import threading

import pytest

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
