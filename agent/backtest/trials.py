"""Trial ledger for search-accounting (Deflated-Sharpe input).

An LLM agent is the optimizer in this system: it edits parameters, reruns the
backtest, edits again — and only the best variant is ever shown. That selection
bias is invisible to any single-config post-hoc test. This ledger records every
trial of a search so Deflated Sharpe (``deflated_sharpe_ratio``) can count "how
many variants were tried" and deflate the observed Sharpe accordingly.

Ledger: ``get_runtime_root() / "backtest_trials.jsonl"`` (the only legal state
root — raw env reads in ``agent/backtest/`` are a blocking CI-gate violation).

Write semantics: **fail-closed on the agent path**. The ledger is the single
source of truth for the anti-overfitting control; a silent append failure would
let DSR compute significance on a partial trial set and quietly defeat the
whole point. Appends use ``os.open(O_APPEND|O_CREAT|O_WRONLY)`` + flush/fsync
so concurrent processes can't interleave half a line (Windows).
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from backtest.run_card import _file_hash, _json_hash

try:  # POSIX advisory lock (Linux/macOS).
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows advisory byte-range lock.
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

LEDGER_FILENAME = "backtest_trials.jsonl"
_LOCK_SUFFIX = ".lock"

# search_id charset: no path separators, no control chars, bounded length.
_SEARCH_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")


@contextmanager
def _ledger_lock(path: Path) -> Iterator[None]:
    """Hold a blocking cross-process advisory lock for the ledger.

    ``O_APPEND`` is NOT atomic across threads/processes on Windows (a blocked
    thread can have its file position stolen mid-write), so concurrent trials
    interleave/drop rows without a lock. Blocking (not NB) — a writer waits
    rather than dropping its row, which the fail-closed ledger semantics rely on.
    """
    lock_path = path.with_name(path.name + _LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def sanitize_search_id(value: Any) -> Optional[str]:
    """Return a safe search_id, or None if the value is unusable.

    The agent can forge a ``_search_id`` in its config to merge its trials into
    another group (inflate N to dilute DSR) or split groups (dodge the N
    penalty). Only server-supplied ids matching the charset survive; anything
    else is dropped so the ledger never groups on an attacker-chosen key.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if _SEARCH_ID_RE.match(value) else None


def current_search_id() -> Optional[str]:
    """Return the active server-supplied search id, or None if not search-marked.

    Single source of truth for "which env var, sanitised how" — the four call
    sites that need it (trial ledger append, DSR, search-marker check, OOS
    isolation) all read through here so the EnvConfig field and the charset
    check stay in one place. Returns None (never raises) when the config layer
    is unavailable or no valid id is set.
    """
    try:
        from src.config.accessor import get_env_config
        raw = get_env_config().paths.vibe_trading_search_id
    except Exception:
        return None
    return sanitize_search_id(raw)


def ledger_path(root: Optional[Path] = None) -> Path:
    """Return the ledger path under the runtime root."""
    if root is None:
        from src.config.paths import get_runtime_root
        root = get_runtime_root()
    return Path(root) / LEDGER_FILENAME


def config_hash(config: Dict[str, Any], config_file: Optional[Path] = None) -> str:
    """Same function, same source as run_card (never a second implementation).

    run_card hashes the on-disk config.json when present, else the in-memory
    config. Mirror that exactly so a trial row's config_hash matches the card.
    """
    if config_file is not None and Path(config_file).exists():
        return _file_hash(Path(config_file))
    return _json_hash(config)


def append_trial(record: Dict[str, Any], root: Optional[Path] = None) -> None:
    """Atomically append one trial row. Raises on failure (fail-closed).

    The line is serialized with ``json.dumps`` (single line, embedded newlines
    escaped) to preserve framing; flush + fsync before close makes the append
    durable and avoids interleaved half-lines across processes.
    """
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _ledger_lock(path):
        fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)


def _sanctioned_valid_end_locked(path: Path, search_id: str) -> Optional[str]:
    """Return the first recorded ``valid_end`` for a search (the authority).

    Caller MUST hold ``_ledger_lock(path)`` — this reads the ledger without
    taking the lock so it can run inside the compare-and-set critical section
    of :func:`append_trial_with_sanction`. Returns None when no prior trial of
    this search recorded a ``valid_end`` yet.
    """
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or rec.get("search_id") != search_id:
                continue
            value = rec.get("valid_end")
            if value is not None:
                return value
    return None


def append_trial_with_sanction(record: Dict[str, Any], root: Optional[Path] = None) -> None:
    """Append one trial row, locking the search's OOS boundary to its first trial.

    ``valid_sharpe`` is comparable across a search's trials only when every
    trial uses the SAME out-of-sample boundary — ``run_dsr`` pools each trial's
    ``valid_sharpe`` into one DSR. An agent that edits ``valid_end`` mid-search
    would otherwise pool differently-meaning Sharpes into one verdict (review
    RT-3: "change the exam scope to the part you memorised").

    The authority is the first ``valid_end`` this ``search_id`` ever recorded.
    If this row's ``valid_end`` differs, the row is STILL written (honest
    record — the trial happened) but flagged ``valid_end_mismatch: True`` and
    stamped with ``sanctioned_valid_end`` so ``run_dsr`` can exclude it from the
    pooled score.

    The read-authority → compare → append sequence runs inside a single
    ``_ledger_lock`` so two concurrent trials can't both conclude they are the
    first (compare-and-set is atomic). Raises on write failure (fail-closed).
    """
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    search_id = record.get("search_id")
    with _ledger_lock(path):
        if search_id is not None:
            sanctioned = _sanctioned_valid_end_locked(path, search_id)
            current = record.get("valid_end")
            # Only compare when both sides have a boundary. A None-authority
            # means this is the first trial with a valid_end; a None-current
            # means no OOS segment (already excluded from DSR upstream).
            if sanctioned is not None and current is not None and current != sanctioned:
                record["valid_end_mismatch"] = True
                record["sanctioned_valid_end"] = sanctioned
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)


def read_trials(
    search_id: Optional[str] = None,
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Read ledger rows, tolerant of corrupt lines.

    A locally-corrupted or partially-written file must not abort the read of
    the intact rows — bad lines are skipped (and counted on the result via the
    ``_corrupt_lines`` sentinel appended to nothing; callers needing the count
    can use :func:`read_trials_verbose`).
    """
    rows, _ = read_trials_verbose(search_id=search_id, root=root)
    return rows


def read_trials_verbose(
    search_id: Optional[str] = None,
    root: Optional[Path] = None,
) -> tuple[List[Dict[str, Any]], int]:
    """Read ledger rows, returning (rows, n_corrupt_lines)."""
    path = ledger_path(root)
    rows: List[Dict[str, Any]] = []
    corrupt = 0
    if not path.exists():
        return rows, corrupt
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                corrupt += 1
                continue
            if not isinstance(rec, dict):
                corrupt += 1
                continue
            if search_id is not None and rec.get("search_id") != search_id:
                continue
            rows.append(rec)
    return rows, corrupt


def run_dsr(
    search_id: str,
    equity_series: Any,
    bars_per_year: int,
    root: Optional[Path] = None,
    ledger_ok: bool = True,
) -> Dict[str, Any]:
    """Compute Deflated Sharpe for one search from its ledger group.

    Reads this search's trial rows, takes each trial's ``valid_sharpe`` as the
    score (out-of-sample — the only honest input; falls back to unavailable
    rather than self-deceiving on train Sharpe), and uses the selected run's
    own daily equity returns.

    Args:
        search_id: The search group to score.
        equity_series: The SELECTED run's equity curve (daily returns derived).
        bars_per_year: This run's annualisation factor (never hard 252).
        root: Ledger root override (tests).
        ledger_ok: False when the trial-ledger write failed — DSR on a partial
            trial set is not authoritative (review H6).

    Returns:
        The ``deflated_sharpe_ratio`` dict; ``authoritative`` forced False when
        ``ledger_ok`` is False.
    """
    from backtest.validation import deflated_sharpe_ratio

    trials = read_trials(search_id=search_id, root=root)
    # Exclude trials whose OOS boundary drifted from the search's sanctioned
    # first-trial boundary (append_trial_with_sanction flags them): their
    # valid_sharpe measures a different out-of-sample window and would pollute
    # the pooled score. Count the exclusions so the drift is visible, not silent.
    pooled = [t for t in trials if not t.get("valid_end_mismatch")]
    n_excluded = len(trials) - len(pooled)
    scores = [t.get("valid_sharpe") for t in pooled]
    daily = equity_series.pct_change().dropna().values.tolist()
    out = deflated_sharpe_ratio(
        trial_scores=scores,
        daily_returns=daily,
        bars_per_year=bars_per_year,
        mode="oos",
    )
    out["search_id"] = search_id
    if n_excluded:
        out["n_excluded_boundary_drift"] = n_excluded
    if not ledger_ok:
        out["authoritative"] = False
        out["reason"] = (out.get("reason") + "; " if out.get("reason") else "") + \
            "ledger write failed: partial trial set"
    return out
