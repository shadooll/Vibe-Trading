"""Tests for alpha/beta attribution (spec §5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import calc_metrics


def _mk(port_ret, bench_ret):
    idx = pd.date_range("2023-01-01", periods=len(port_ret), freq="D")
    port_eq = 1000.0 * np.cumprod(1 + np.asarray(port_ret))
    eq = pd.Series(port_eq, index=idx)
    bench = pd.Series(np.asarray(bench_ret), index=idx)
    return eq, bench


def test_recovers_beta_and_alpha():
    rng = np.random.default_rng(2)
    n = 300
    bench = rng.normal(0.0004, 0.01, n)
    beta_true, alpha_daily = 0.8, 0.0005
    eps = rng.normal(0, 0.002, n)
    port = beta_true * bench + alpha_daily + eps
    eq, bench_s = _mk(port, bench)
    m = calc_metrics(eq, [], 1000.0, 252, bench_ret=bench_s)
    att = m["attribution"]
    assert att is not None
    assert att["beta"] == pytest.approx(beta_true, abs=1e-2)
    assert att["alpha_annual"] == pytest.approx(alpha_daily * 252, abs=0.05)


def test_attribution_null_without_benchmark():
    rng = np.random.default_rng(3)
    port = rng.normal(0.0005, 0.01, 200)
    eq, _ = _mk(port, np.zeros(200))
    m = calc_metrics(eq, [], 1000.0, 252, bench_ret=None)
    assert m["attribution"] is None


def test_attribution_reports_approximation_residual():
    # Interaction term present (port = β·bench + α·bench²) → residual > 0.
    n = 300
    bench = np.linspace(-0.02, 0.02, n)
    port = 0.9 * bench + 3.0 * bench**2
    eq, bench_s = _mk(port, bench)
    m = calc_metrics(eq, [], 1000.0, 252, bench_ret=bench_s)
    att = m["attribution"]
    assert att is not None
    assert "approximation" in att
    assert att["residual_share"] >= 0.0
    assert "alpha_arith" in att and "beta_arith" in att


def test_exposure_means_from_positions():
    n = 50
    port = np.full(n, 0.001)
    bench = np.full(n, 0.0005)
    eq, bench_s = _mk(port, bench)
    idx = eq.index
    positions = pd.DataFrame({"AAA": [0.5] * 25 + [0.0] * 25, "BBB": [0.3] * n}, index=idx)
    m = calc_metrics(eq, [], 1000.0, 252, bench_ret=bench_s, positions=positions)
    att = m["attribution"]
    assert att["gross_exposure"] == pytest.approx((0.5 + 0.3) * 0.5 + 0.3 * 0.5, abs=1e-6)
    # net invested only over bars with a position.
    assert att["net_exposure_invested"] <= att["gross_exposure"] + 1e-9
    assert att["net_exposure"] <= att["net_exposure_invested"] + 1e-9
