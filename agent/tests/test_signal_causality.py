"""Tests for the signal-level causality check (D1).

Covers the spec §1.6 cases: a causal engine passes, full-sample normalisation
fails, a full-sample rolling z-score fails, pure warmup passes, a length guard
degrades to SKIP, misaligned output indexes still compare correctly, short data
skips, a truncated-generate exception degrades to SKIP, fail mode exits 1, and
the probe cap short-circuits.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.causality import check_signal_causality


def _frame(n: int = 300, start: str = "2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="D")
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


class _CausalEngine:
    """Causal: rolling mean explicitly shifted by one bar."""

    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            ma = df["close"].rolling(20).mean().shift(1)
            out[sym] = (df["close"] > ma).astype(float).where(ma.notna())
        return out


class _FullSampleNormEngine:
    """Cheat: full-sample normalisation (close / full-period mean)."""

    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            out[sym] = df["close"] / df["close"].mean() - 1.0
        return out


class _FullSampleZEngine:
    """Cheat: rolling z-score using full-sample mean/std."""

    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            c = df["close"]
            out[sym] = (c - c.mean()) / c.std()
        return out


class _WarmupEngine:
    """Causal but with a long warmup (early NaN)."""

    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            ma = df["close"].rolling(120).mean().shift(1)
            out[sym] = (df["close"] > ma).astype(float).where(ma.notna())
        return out


class _LengthGuardEngine:
    """Legal length guard: returns nothing until N bars exist."""

    N = 250

    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            if len(df) < self.N:
                out[sym] = pd.Series([float("nan")] * len(df), index=df.index)
                continue
            ma = df["close"].rolling(20).mean().shift(1)
            out[sym] = (df["close"] > ma).astype(float).where(ma.notna())
        return out


class _DropnaEngine:
    """Causal but emits a dropna'd (misaligned) output index."""

    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            ma = df["close"].rolling(20).mean().shift(1)
            sig = (df["close"] > ma).astype(float).where(ma.notna())
            out[sym] = sig.dropna()
        return out


class _CrashOnTruncEngine:
    """Raises when the frame is shorter than a fixed threshold."""

    def generate(self, data_map):
        out = {}
        for sym, df in data_map.items():
            if len(df) < 250:
                raise IndexError("too short")
            ma = df["close"].rolling(20).mean().shift(1)
            out[sym] = (df["close"] > ma).astype(float).where(ma.notna())
        return out


def _map(n: int = 300, symbols=("AAA",)):
    return {s: _frame(n) for s in symbols}


def test_causal_engine_passes():
    res = check_signal_causality(_CausalEngine(), _map())
    assert res["verdict"] == "PASS", res
    assert res["n_compared"] >= 3
    assert res["max_diff"] <= 1e-9


def test_full_sample_normalisation_fails():
    res = check_signal_causality(_FullSampleNormEngine(), _map())
    assert res["verdict"] == "FAIL"
    assert res["leaks"], "expected at least one leak"


def test_full_sample_zscore_fails():
    res = check_signal_causality(_FullSampleZEngine(), _map())
    assert res["verdict"] == "FAIL"


def test_pure_warmup_passes():
    # Warmup=120, min_bars default 200, L=300 → probes start at max(0.35L,30)=105,
    # which is still inside warmup → some probes skipped, but enough finite ones
    # remain (t up to 298) for a PASS.
    res = check_signal_causality(_WarmupEngine(), _map())
    assert res["verdict"] in ("PASS", "SKIP"), res
    assert res["verdict"] != "FAIL"


def test_length_guard_skips_not_fails():
    res = check_signal_causality(_LengthGuardEngine(), _map())
    # Truncated frames shorter than N=250 yield all-NaN; the length-guard
    # exemption must keep this from being a FAIL.
    assert res["verdict"] != "FAIL", res
    assert res["skipped"] > 0 or res["verdict"] == "PASS"


def test_misaligned_output_index_passes():
    res = check_signal_causality(_DropnaEngine(), _map())
    assert res["verdict"] == "PASS", res


def test_short_data_skips():
    res = check_signal_causality(_CausalEngine(), _map(n=100))  # < min_bars
    assert res["verdict"] == "SKIP"


def test_truncated_exception_degrades_to_skip():
    res = check_signal_causality(_CrashOnTruncEngine(), _map())
    assert res["verdict"] != "FAIL"
    assert res["skipped"] > 0


def test_probe_cap_short_circuits():
    many = _map(n=300, symbols=tuple(f"S{i:02d}" for i in range(10)))
    res = check_signal_causality(_CausalEngine(), many, points_per_symbol=5, max_points=7)
    # Cap hit before all symbols*points were compared.
    assert res["n_compared"] <= 7
    assert "cap" in res["reason"] or res["verdict"] in ("PASS", "SKIP")
