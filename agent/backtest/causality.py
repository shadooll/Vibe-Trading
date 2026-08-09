"""Signal-level causality check.

The engine's ``_align`` shifts every signal by one bar (next-bar-open
semantics). That is a *decision lag*, not a causality guarantee: an
agent-generated ``signal_engine.py`` can smuggle future information into the
t-1 signal value itself — full-sample normalisation (``close / close.mean()``),
a rolling statistic that forgot its own ``shift``, a cross-sectional rank that
reads the current bar. A one-bar shift cannot wash that out.

The engine-level causality tests only prove *fill prices*
(``execution_open``/``historical_base_price``) don't leak. Nothing checks the
signal *computation* itself. This module is that check: compare the signal a
strategy emits at time ``t`` against the same strategy run on data truncated to
``t`` (the point-in-time oracle). If the two disagree, the signal used future
data.

Reference: EP004 ``factor_causality_check.py`` (full vs truncated comparison).
The Freqtrade built-in ``lookahead-analysis`` false-positives on
cross-sectional strategies, which is why this is a bespoke check rather than a
reuse of that.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _is_finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def check_signal_causality(
    signal_engine: Any,
    data_map: Dict[str, pd.DataFrame],
    full_signals: Optional[Dict[str, pd.Series]] = None,
    *,
    points_per_symbol: int = 5,
    tol: float = 1e-9,
    min_bars: int = 200,
    skip_warmup_bars: Optional[int] = None,
    max_points: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare full-data vs truncated-data signals to detect look-ahead.

    Args:
        signal_engine: Instantiated engine with a ``generate(data_map)`` method.
        data_map: The SAME enriched map the engine will consume (post
            ``_maybe_enrich_*``), so the check sees exactly what execution sees.
        full_signals: Pre-computed ``signal_engine.generate(data_map)``. When
            provided the check reuses it (single generate per run); when None
            it is computed here.
        points_per_symbol: Probe points per symbol.
        tol: Absolute tolerance for the full/trunc comparison.
        min_bars: Symbols with fewer bars are skipped.
        skip_warmup_bars: Probe points inside the first N bars are skipped.
            None = automatic ``min(30, L // 10)``.
        max_points: Global cap on probe points per run (cost/timeout guardrail;
            the AST scrubber blocks dangerous ops, not complexity, and the CLI
            path has no ``Runner(timeout=300)`` backstop).
        seed: Seed to pin ``np.random`` so a ``generate`` that draws randomness
            is at least reproducible (generate must be deterministic — a
            genuinely random one will false-positive).

    Returns:
        Dict with verdict ("PASS" | "FAIL" | "SKIP"), n_compared, max_diff,
        leaks, skipped, reason.
    """
    result: Dict[str, Any] = {
        "verdict": "PASS",
        "n_compared": 0,
        "max_diff": 0.0,
        "leaks": [],
        "skipped": 0,
        "reason": "",
    }

    if not data_map:
        result["verdict"] = "SKIP"
        result["reason"] = "empty data_map"
        return result

    # Pin the global RNG so a generate that draws from np.random is at least
    # reproducible across the full and truncated calls.
    np.random.default_rng(seed)
    np.random.seed(seed)

    try:
        full = full_signals if full_signals is not None else signal_engine.generate(data_map)
    except Exception as exc:  # pragma: no cover - defensive
        result["verdict"] = "SKIP"
        result["reason"] = f"full generate failed: {exc}"
        return result
    if not isinstance(full, dict):
        result["verdict"] = "SKIP"
        result["reason"] = "generate() did not return a dict"
        return result

    total_probes = 0

    for symbol, frame in data_map.items():
        if symbol not in full:
            continue  # generate dropped this symbol — not our business
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            continue

        L = len(frame)
        if L < min_bars:
            continue

        warmup = skip_warmup_bars if skip_warmup_bars is not None else min(30, L // 10)
        lo = max(int(L * 0.35), warmup)
        hi = L - 2
        if hi <= lo:
            continue
        idxs = np.unique(np.linspace(lo, hi, points_per_symbol).astype(int))

        full_series = full[symbol].reindex(frame.index)

        for t in idxs:
            if total_probes >= max_points:
                result["reason"] = result["reason"] or f"probe cap reached ({max_points})"
                _finalize(result)
                return result

            ts = frame.index[t]
            v_full = full_series.iloc[t] if t < len(full_series) else np.nan
            if not _is_finite(v_full):
                continue  # warmup — full signal itself not defined yet

            total_probes += 1

            # Point-in-time oracle: truncate every frame to <= ts.
            truncated_map = {s: f[f.index <= ts] for s, f in data_map.items()}
            try:
                trunc = signal_engine.generate(truncated_map)
            except Exception:
                # Truncated frame is shorter; a legal strategy may go out of
                # bounds. Degrade to SKIP, never abort the run.
                result["skipped"] += 1
                continue

            if not isinstance(trunc, dict) or symbol not in trunc:
                result["skipped"] += 1
                continue

            trunc_series = trunc[symbol].reindex(frame.index)
            v_trunc = trunc_series.iloc[t] if t < len(trunc_series) else np.nan

            if not _is_finite(v_trunc):
                # Full finite but truncated not — could be a point leak OR a
                # length guard ("if len(df) < N: return 0"). Distinguish by the
                # truncated tail: if the LAST >=3 truncated bars are all
                # non-finite while full is finite, it reads as a length guard,
                # not a point leak → degrade to SKIP.
                if _is_length_guard(trunc_series):
                    result["skipped"] += 1
                    continue
                result["leaks"].append({
                    "symbol": symbol,
                    "timestamp": str(ts),
                    "reason": f"full={float(v_full):.6g} finite but trunc non-finite",
                })
                result["verdict"] = "FAIL"
                result["reason"] = "signal differs between full and truncated data"
                result["n_compared"] += 1
                _finalize(result)
                return result

            diff = abs(float(v_full) - float(v_trunc))
            if math.isfinite(diff):
                result["max_diff"] = max(result["max_diff"], diff)
            result["n_compared"] += 1

            if diff > tol:
                result["leaks"].append({
                    "symbol": symbol,
                    "timestamp": str(ts),
                    "reason": f"full={float(v_full):.6g} != trunc={float(v_trunc):.6g}",
                })
                result["verdict"] = "FAIL"
                result["reason"] = "signal differs between full and truncated data"
                _finalize(result)
                return result

    _finalize(result)
    return result


def _is_length_guard(trunc_series: pd.Series, tail: int = 3) -> bool:
    """Whether the truncated signal's tail is wholly non-finite (length guard)."""
    vals = trunc_series.values
    tail_vals = vals[-tail:] if len(vals) >= tail else vals
    return len(tail_vals) > 0 and all(not _is_finite(v) for v in tail_vals)


def _finalize(result: Dict[str, Any]) -> None:
    """Apply the evidence floor: too few comparisons is SKIP, not PASS."""
    if result["verdict"] == "PASS" and result["n_compared"] < 3:
        result["verdict"] = "SKIP"
        result["reason"] = (
            result["reason"] or f"insufficient evidence ({result['n_compared']} compared)"
        )
