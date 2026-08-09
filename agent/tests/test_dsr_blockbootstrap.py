"""Tests for Deflated Sharpe (D2) and block bootstrap (D3)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from backtest.validation import block_bootstrap, deflated_sharpe_ratio


# ─── Deflated Sharpe ───


def _daily(n=252, mean=0.0005, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sd, n).tolist()


def test_dsr_in_unit_interval():
    out = deflated_sharpe_ratio(
        trial_scores=[0.5, 0.8, 1.1, 0.9, 1.3],
        daily_returns=_daily(),
        bars_per_year=252,
    )
    assert out["DSR"] is not None
    assert 0.0 <= out["DSR"] <= 1.0
    assert out["n_trials"] == 5
    assert out["verdict"] in ("significant", "weak", "not_significant")


def test_dsr_monotone_decreasing_in_n():
    # Fixed sr_var and observed series; more trials → higher luck baseline →
    # lower DSR. (N enters only via the Φ⁻¹ args, so monotonicity holds.)
    daily = _daily(n=400, mean=0.001, sd=0.01, seed=3)
    base = [0.2, 0.5, 0.8, 1.1]  # var fixed
    sharpes = []
    for k in range(2, 8):
        scores = (base * k)[:k]  # repeat pattern to grow N while keeping spread shape
        # force identical variance by construction: use a fixed-variance family
        scores = list(np.linspace(0.2, 1.1, k))
        out = deflated_sharpe_ratio(scores, daily, bars_per_year=252)
        sharpes.append(out["DSR"])
    # Broader N (same endpoints → roughly same var) should not increase DSR.
    assert sharpes[-1] <= sharpes[0] + 1e-9


def test_dsr_more_trials_lower_when_variance_fixed():
    rng = np.random.default_rng(1)
    daily = _daily(n=400, mean=0.0012, sd=0.01, seed=7)
    # Construct two score sets with the SAME variance but different N.
    small = list(rng.normal(0.8, 0.3, 4))
    large = list(rng.normal(0.8, 0.3, 60))
    # equalise variance
    small = (np.array(small) - np.mean(small)) / np.std(small, ddof=1) * 0.3 + 0.8
    large = (np.array(large) - np.mean(large)) / np.std(large, ddof=1) * 0.3 + 0.8
    d_small = deflated_sharpe_ratio(small.tolist(), daily, 252)["DSR"]
    d_large = deflated_sharpe_ratio(large.tolist(), daily, 252)["DSR"]
    assert d_large < d_small


def test_dsr_bars_per_year_matters():
    scores = [0.5, 0.9, 1.2, 0.7]
    daily = _daily()
    d252 = deflated_sharpe_ratio(scores, daily, bars_per_year=252)
    d365 = deflated_sharpe_ratio(scores, daily, bars_per_year=365)
    # crypto (365) has a higher luck baseline sr0 → different (lower) DSR.
    assert d365["sr0_annual"] != d252["sr0_annual"]
    assert d365["DSR"] != d252["DSR"]


def test_dsr_unavailable_when_too_few_trials():
    out = deflated_sharpe_ratio([1.0], _daily(), 252)
    assert out["DSR"] is None
    assert out["verdict"] == "unavailable"
    assert "trials" in out["reason"]


def test_dsr_unavailable_when_too_short():
    out = deflated_sharpe_ratio([0.5, 0.8, 1.0], _daily(n=10), 252)
    assert out["DSR"] is None
    assert "30" in out["reason"]


def test_dsr_unavailable_on_zero_variance():
    out = deflated_sharpe_ratio([0.5, 0.8], [0.001] * 100, 252)
    assert out["DSR"] is None


def test_dsr_train_only_flagged_in_sample():
    out = deflated_sharpe_ratio([0.5, 0.8, 1.0], _daily(), 252, mode="train_only")
    assert out.get("in_sample") is True
    assert "in-sample" in out["reason"]


def test_dsr_matches_reference_formula():
    # Hand-compute the reference (EP004 deflated_sharpe.py) for a fixed input.
    scores = [0.4, 0.7, 1.0, 0.9, 1.2, 0.6]
    rng = np.random.default_rng(11)
    daily = rng.normal(0.0008, 0.012, 300)
    bpy = 252
    out = deflated_sharpe_ratio(scores, daily.tolist(), bpy)

    N = len(scores)
    sr_var = float(np.var(scores, ddof=1))
    norm = NormalDist()
    sr0_daily = math.sqrt(sr_var / bpy) * (
        (1 - 0.5772156649) * norm.inv_cdf(1 - 1 / N)
        + 0.5772156649 * norm.inv_cdf(1 - 1 / (N * math.e))
    )
    assert out["sr0_annual"] == pytest.approx(sr0_daily * math.sqrt(bpy), rel=1e-9)


# ─── Block bootstrap ───


def test_block_bootstrap_prob_profit_consistency():
    # i.i.d. normal returns with positive drift → prob_profit should be high.
    rng = np.random.default_rng(5)
    rets = rng.normal(0.001, 0.01, 252).tolist()
    out = block_bootstrap(rets, n_bootstrap=2000, block=10, seed=7)
    assert 0.0 <= out["prob_profit"] <= 1.0
    assert out["prob_profit"] > 0.5  # positive drift
    assert out["final_P5"] <= out["final_P50"] <= out["final_P95"]


def test_block_bootstrap_seed_reproducible():
    rng = np.random.default_rng(6)
    rets = rng.normal(0.0005, 0.01, 200).tolist()
    a = block_bootstrap(rets, n_bootstrap=500, block=10, seed=42)
    b = block_bootstrap(rets, n_bootstrap=500, block=10, seed=42)
    assert a["final_P50"] == b["final_P50"]
    assert a["prob_profit"] == b["prob_profit"]


def test_block_bootstrap_block1_is_iid():
    # block=1 degenerates to i.i.d. bootstrap; results still well-formed.
    rng = np.random.default_rng(8)
    rets = rng.normal(0.0, 0.01, 100).tolist()
    out = block_bootstrap(rets, n_bootstrap=500, block=1, seed=3)
    assert out["block"] == 1
    assert "prob_profit" in out


def test_block_bootstrap_short_series_no_crash():
    assert "error" in block_bootstrap([0.01, 0.02], n_bootstrap=100)
    # N < block shrinks the block, doesn't crash.
    out = block_bootstrap([0.01, -0.005, 0.02, -0.01, 0.015], n_bootstrap=100, block=10)
    assert out["block"] <= 2


def test_block_bootstrap_fan_chart_shape():
    rng = np.random.default_rng(9)
    rets = rng.normal(0.001, 0.01, 300).tolist()
    out = block_bootstrap(rets, n_bootstrap=500, block=10, keep_paths=400)
    ep = out["equity_paths"]
    assert len(ep["steps"]) <= 400
    assert len(ep["samples"]) <= 30
    assert len(ep["actual"]) == len(ep["band_p50"]) == len(ep["steps"])
