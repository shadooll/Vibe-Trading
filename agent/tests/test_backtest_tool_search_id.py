"""Tests for BacktestTool search-id injection (review S-H1).

The server/API path must activate search accounting (trial ledger / DSR / OOS
isolation) by injecting the real session id, not rely on an env var that no
code writes. These tests pin the injection chain: constructor-injected session
id → run_backtest's extra_env → subprocess VIBE_TRADING_SEARCH_ID.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.tools.backtest_tool import BacktestTool, run_backtest


def _make_run_dir(tmp_path: Path) -> Path:
    # safe_run_dir only accepts run dirs under the runtime root's runs/ dir, so
    # build the fixture there (VIBE_TRADING_HOME is pointed at tmp by the caller).
    run_dir = tmp_path / "home" / "runs" / "run_1"
    (run_dir / "code").mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"source": "tushare", "codes": ["000001.SZ"]}), encoding="utf-8"
    )
    (run_dir / "code" / "signal_engine.py").write_text("# stub\n", encoding="utf-8")
    return run_dir


class _FakeResult:
    success = True
    exit_code = 0
    stdout = ""
    stderr = ""
    artifacts: dict = {}


def _capture_extra_env(monkeypatch, tmp_path):
    """Mock Runner.execute to capture the extra_env it is called with."""
    captured: dict = {}

    def fake_execute(self, entry_script, run_path, **kwargs):
        captured["extra_env"] = kwargs.get("extra_env") or {}
        return _FakeResult()

    monkeypatch.setattr("src.tools.backtest_tool.Runner.execute", fake_execute)
    # Keep the runtime root inside tmp so get_runtime_root() is hermetic.
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "home"))
    from src.config.accessor import reset_env_config
    reset_env_config()
    return captured


def test_injected_session_id_reaches_subprocess_env(tmp_path, monkeypatch):
    captured = _capture_extra_env(monkeypatch, tmp_path)
    run_dir = _make_run_dir(tmp_path)

    tool = BacktestTool(default_session_id="sess-abc-123")
    tool.execute(run_dir=str(run_dir))

    assert captured["extra_env"].get("VIBE_TRADING_SEARCH_ID") == "sess-abc-123"


def test_injected_session_id_sanitised(tmp_path, monkeypatch):
    captured = _capture_extra_env(monkeypatch, tmp_path)
    run_dir = _make_run_dir(tmp_path)

    # A forged/invalid id (path separator) must not survive.
    tool = BacktestTool(default_session_id="../evil")
    tool.execute(run_dir=str(run_dir))

    assert "VIBE_TRADING_SEARCH_ID" not in captured["extra_env"]


def test_no_injected_id_falls_back_to_env_chain(tmp_path, monkeypatch):
    captured = _capture_extra_env(monkeypatch, tmp_path)
    run_dir = _make_run_dir(tmp_path)

    # CLI/legacy: no constructor injection, no env set → no search id (the
    # fail-closed "not part of a search" path that isolation/ledger rely on).
    monkeypatch.delenv("VIBE_GOAL_SESSION_ID", raising=False)
    monkeypatch.delenv("VIBE_TRADING_SEARCH_ID", raising=False)
    from src.config.accessor import reset_env_config
    reset_env_config()

    run_backtest(str(run_dir))  # no search_session_id
    assert "VIBE_TRADING_SEARCH_ID" not in captured["extra_env"]


def test_env_chain_still_honoured_when_no_injection(tmp_path, monkeypatch):
    captured = _capture_extra_env(monkeypatch, tmp_path)
    run_dir = _make_run_dir(tmp_path)

    # Pre-set VIBE_TRADING_SEARCH_ID (the legacy server-set path) is still used
    # when no constructor injection is present.
    monkeypatch.setenv("VIBE_TRADING_SEARCH_ID", "legacy-search-9")
    from src.config.accessor import reset_env_config
    reset_env_config()

    run_backtest(str(run_dir))
    assert captured["extra_env"].get("VIBE_TRADING_SEARCH_ID") == "legacy-search-9"


def test_build_registry_injects_session_id_into_backtest_tool(monkeypatch):
    """build_registry wires the host session id into BacktestTool (server path)."""
    from src.tools import build_registry

    registry = build_registry(include_shell_tools=False, session_id="web-session-42")
    tool = registry.get("backtest")
    assert tool is not None
    assert tool._default_session_id == "web-session-42"
