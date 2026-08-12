"""Backtest execution tool: validates config.json + signal_engine.py and runs the built-in engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backtest.loaders.registry import VALID_SOURCES
from src.agent.progress import emit_progress
from src.agent.tools import BaseTool
from src.core.runner import Runner
from src.tools.path_utils import safe_run_dir

logger = logging.getLogger(__name__)


def _reap_protected_writes(run_path: Path, config: dict, search_id: str | None) -> None:
    """Perform the protected writes the agent-running subprocess deferred (C′).

    Spec §9.12: on the agent search path the subprocess (which executes
    ``signal_engine.py``) records trial + isolation metadata but must NOT write
    the protected ledger or OOS holdout. This trusted server process — which
    runs no agent code — appends the trial ledger (first-trial boundary lock,
    §9.9), persists the holdout, then computes the DSR on the now-complete
    ledger and merges it back into the run card.

    Reads the subprocess's run card for the handoff; never evaluates or
    unpickles anything the subprocess produced (ledger record is plain scalars;
    the holdout is re-derived server-side from a fresh registry fetch). A ledger
    append failure is logged but does not fail the already-completed run — the
    DSR is then computed with ``ledger_ok=False`` (forced non-authoritative).
    """
    card_path = run_path / "run_card.json"
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return  # no card → nothing to reap (the run itself reported the failure)

    trial = card.get("_trial_record")
    isolation = card.get("data_isolation") or {}
    ledger_status = (card.get("metrics") or {}).get("ledger_status")
    holdout_pending = isolation.get("holdout_status") == "server_pending"
    if ledger_status != "server_pending" and not holdout_pending:
        return  # non-search path: nothing was deferred

    from backtest.trials import append_trial_with_sanction, run_dsr, sanitize_search_id
    from backtest.runner import persist_test_holdout

    # 1. Trial ledger (fail-closed semantics preserved: DSR forced non-
    #    authoritative when the write fails).
    ledger_ok = True
    if isinstance(trial, dict) and search_id:
        record = dict(trial)
        record["search_id"] = search_id  # server-authoritative id, not the subprocess's
        try:
            append_trial_with_sanction(record)
        except Exception as exc:  # noqa: BLE001 - never crash the reaped run
            ledger_ok = False
            logger.error("server-side trial ledger append failed: %s", exc)

    # 2. OOS holdout persistence (server is vibe-owned → no container perm crash).
    holdout_meta: dict = {}
    if holdout_pending:
        try:
            holdout_meta = persist_test_holdout(config, run_path.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("server-side holdout persistence failed: %s", exc)
            holdout_meta = {"holdout_status": "failed"}

    # 3. DSR on the now-complete ledger; merge back into the card.
    card.pop("_trial_record", None)
    if "ledger_status" in (card.get("metrics") or {}):
        card["metrics"]["ledger_status"] = "server_written" if ledger_ok else "failed"
    if holdout_meta and isinstance(card.get("data_isolation"), dict):
        card["data_isolation"].update(holdout_meta)

    valid_end = (card.get("backtest") or {}).get("valid_end")
    clean_search = sanitize_search_id(search_id)
    if valid_end and clean_search and ledger_status == "server_pending":
        try:
            import pandas as pd

            eq_path = run_path / "artifacts" / "equity.csv"
            equity_series = None
            if eq_path.exists():
                eq_df = pd.read_csv(eq_path)
                if "equity" in eq_df.columns:
                    equity_series = pd.Series(eq_df["equity"].to_numpy(dtype="float64"))
            if equity_series is not None and len(equity_series) > 1:
                from backtest.metrics import calc_bars_per_year

                interval = (card.get("backtest") or {}).get("interval", "1D")
                source = (card.get("data_sources") or [config.get("source", "tushare")])[0]
                bars_per_year = calc_bars_per_year(interval, source)
                card["dsr"] = run_dsr(
                    clean_search, equity_series, bars_per_year, ledger_ok=ledger_ok,
                )
        except Exception as exc:  # noqa: BLE001 - DSR must never crash the run
            logger.warning("server-side DSR computation failed: %s", exc)

    card_path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run_backtest(run_dir: str, search_session_id: str | None = None) -> str:
    """Run backtest: validate config.json + signal_engine.py, invoke built-in engine.

    Args:
        run_dir: Path to the run directory.
        search_session_id: Optional host-injected session id used as the
            search-accounting group id (server path). When None (CLI/legacy),
            the search id falls back to the existing env-config chain below.

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
    # ephemeral sandbox HOME the Runner builds. The search id is the host's real
    # session id (injected via the constructor on the server/API path) first,
    # then the env-config chain (VIBE_GOAL_SESSION_ID / VIBE_TRADING_SEARCH_ID);
    # the subprocess revalidates the charset and fails closed if the write fails.
    from src.config.accessor import get_env_config
    from src.config.paths import get_runtime_root
    _cfg = get_env_config()
    extra_env: dict[str, str] = {"VIBE_TRADING_HOME": str(get_runtime_root())}
    from backtest.trials import sanitize_search_id
    _search = (
        sanitize_search_id(search_session_id)
        or sanitize_search_id(getattr(getattr(_cfg, "paths", None), "vibe_goal_session_id", "") or "")
        or sanitize_search_id(getattr(getattr(_cfg, "paths", None), "vibe_trading_search_id", ""))
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

    # C′ (spec §9.12): the agent-running subprocess only RECORDS the trial /
    # isolation metadata — this trusted server process performs the protected
    # writes (ledger append, OOS holdout) and the DSR after the subprocess exits.
    if result.success:
        _reap_protected_writes(run_path, config, _search)

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

    def __init__(self, *, default_session_id: str | None = None) -> None:
        """Inject the host session id (server/API path) for search accounting.

        The LLM never knows the session id; the host runtime injects it at
        construction (same pattern as the goal tools). None on CLI/legacy —
        run_backtest then falls back to the env-config chain unchanged.
        """
        self._default_session_id = default_session_id

    def execute(self, **kwargs) -> str:
        """Execute backtest."""
        return run_backtest(kwargs["run_dir"], search_session_id=self._default_session_id)
