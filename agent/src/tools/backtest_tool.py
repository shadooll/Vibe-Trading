"""Backtest execution tool: validates config.json + signal_engine.py and runs the built-in engine."""

from __future__ import annotations

import json
from pathlib import Path

from backtest.loaders.registry import VALID_SOURCES
from src.agent.progress import emit_progress
from src.agent.tools import BaseTool
from src.core.runner import Runner
from src.tools.path_utils import safe_run_dir


def run_backtest(run_dir: str) -> str:
    """Run backtest: validate config.json + signal_engine.py, invoke built-in engine.

    Args:
        run_dir: Path to the run directory.

    Returns:
        JSON-formatted execution result.
    """
    emit_progress("validate", message="validating run_dir and config")
    try:
        run_path = safe_run_dir(run_dir)
    except ValueError as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

    config_path = run_path / "config.json"
    if not config_path.exists():
        return json.dumps({"status": "error", "error": "config.json not found"}, ensure_ascii=False)

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"config.json parse error: {e}"}, ensure_ascii=False)

    if "source" not in config:
        return json.dumps({"status": "error", "error": "config.json missing 'source' field (tushare/okx/yfinance)"}, ensure_ascii=False)

    if config["source"] not in VALID_SOURCES:
        return json.dumps({"status": "error", "error": f"source must be one of {VALID_SOURCES}, got: {config['source']}"}, ensure_ascii=False)

    signal_path = run_path / "code" / "signal_engine.py"
    if not signal_path.exists():
        return json.dumps({"status": "error", "error": "code/signal_engine.py not found"}, ensure_ascii=False)

    agent_root = Path(__file__).resolve().parents[2]
    entry_script = agent_root / "backtest" / "runner.py"

    source = config.get("source", "?")
    emit_progress(
        "simulate",
        message=f"running backtest engine (source={source})",
    )
    runner = Runner(timeout=300)

    # Trial-ledger search accounting: give the subprocess a server-side search
    # id (never config.json — run_card's config_hash is a file hash of it) and
    # the REAL runtime root so the ledger lands in ~/.vibe-trading, not the
    # ephemeral sandbox HOME the Runner builds. The search id is the server
    # session id (VIBE_GOAL_SESSION_ID) when present; the subprocess revalidates
    # the charset and fails closed if the ledger write fails.
    from src.config.accessor import get_env_config
    from src.config.paths import get_runtime_root
    _cfg = get_env_config()
    extra_env: dict[str, str] = {"VIBE_TRADING_HOME": str(get_runtime_root())}
    _session = getattr(getattr(_cfg, "paths", None), "vibe_goal_session_id", "") or ""
    from backtest.trials import sanitize_search_id
    _search = sanitize_search_id(_session) or sanitize_search_id(
        getattr(getattr(_cfg, "paths", None), "vibe_trading_search_id", "")
    )
    if _search:
        extra_env["VIBE_TRADING_SEARCH_ID"] = _search

    result = runner.execute(
        entry_script,
        run_path,
        cwd=agent_root,
        cli_args=[str(run_path)],
        extra_env=extra_env,
    )

    emit_progress("finalize", message="collecting artifacts")
    artifacts_found = {name: str(path) for name, path in result.artifacts.items()}
    return json.dumps({
        "status": "ok" if result.success else "error",
        "exit_code": result.exit_code,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
        "artifacts": artifacts_found,
        "run_dir": run_dir,
    }, ensure_ascii=False)


class BacktestTool(BaseTool):
    """Backtest execution tool."""

    name = "backtest"
    description = "Run backtest: validate config.json + signal_engine.py, invoke built-in engine."
    parameters = {
        "type": "object",
        "properties": {
            "run_dir": {"type": "string", "description": "Path to the run directory"},
        },
        "required": ["run_dir"],
    }
    repeatable = True
    is_readonly = False

    def execute(self, **kwargs) -> str:
        """Execute backtest."""
        return run_backtest(kwargs["run_dir"])
