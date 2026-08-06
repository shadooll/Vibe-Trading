"""Tests for the Track B statistics layer (stats.py).

Coverage: power analysis (required_n / power_table), paired bootstrap CI,
Wilcoxon signed-rank, effect_summary fields, and the three-tier gate verdict.
All data is synthetic and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from trade_tools.stats import (
    DEFAULT_COST_FLOOR,
    effect_summary,
    gate_verdict,
    paired_bootstrap_ci,
    power_table,
    required_n,
    wilcoxon_p,
)


# ── 功效分析 ───────────────────────────────────────────────────────────────────
class TestRequiredN:
    def test_known_one_sided_value(self) -> None:
        # delta=0.05, sigma=0.2, alpha=0.05, power=0.8 单侧：
        #   n = ceil(((1.6448536 + 0.8416212) * 0.2 / 0.05)^2) = ceil(98.92) = 99
        assert required_n(0.05, 0.2) == 99

    def test_known_two_sided_value(self) -> None:
        # 双侧 z_{0.975} = 1.9599639 → ceil(125.58) = 126
        assert required_n(0.05, 0.2, side="two") == 126

    def test_known_third_value(self) -> None:
        assert required_n(0.06, 0.2) == 69

    def test_returns_ceil(self) -> None:
        # 结果取整是向上取整：delta 稍大一点即降档，但必为整数
        assert isinstance(required_n(0.05, 0.2), int)

    def test_monotone_in_delta(self) -> None:
        assert required_n(0.05, 0.2) > required_n(0.10, 0.2)

    def test_monotone_in_sigma(self) -> None:
        assert required_n(0.05, 0.1) < required_n(0.05, 0.4)

    def test_two_sided_needs_more(self) -> None:
        assert required_n(0.05, 0.2, side="two") > required_n(0.05, 0.2)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            required_n(0.0, 0.2)
        with pytest.raises(ValueError):
            required_n(0.05, -1.0)
        with pytest.raises(ValueError):
            required_n(0.05, 0.2, alpha=1.5)
        with pytest.raises(ValueError):
            required_n(0.05, 0.2, power=0.0)
        with pytest.raises(ValueError):
            required_n(0.05, 0.2, side="two_sided")


class TestPowerTable:
    def test_rows_monotone_decreasing(self) -> None:
        rows = power_table([0.02, 0.03, 0.05], 0.2)
        assert len(rows) == 3
        assert [r["n_required"] for r in rows] == [619, 275, 99]
        assert rows[0]["delta"] == pytest.approx(0.02)

    def test_matches_required_n(self) -> None:
        rows = power_table([0.05, 0.1], 0.2, side="two")
        assert rows[0]["n_required"] == required_n(0.05, 0.2, side="two")
        assert rows[1]["n_required"] == required_n(0.1, 0.2, side="two")


# ── 配对 bootstrap CI ──────────────────────────────────────────────────────────
_POSITIVE_DELTAS = [0.03, 0.02, 0.05, -0.01, 0.04, 0.01, 0.06, 0.0, 0.02, 0.03]
_CROSS_ZERO = [-0.03, -0.01, 0.01, 0.03, -0.02, 0.02, -0.05, 0.05, -0.04, 0.04]


class TestPairedBootstrapCi:
    def test_seed_reproducible(self) -> None:
        a = paired_bootstrap_ci(_POSITIVE_DELTAS, seed=42)
        b = paired_bootstrap_ci(_POSITIVE_DELTAS, seed=42)
        assert a == b

    def test_interval_contains_observed_mean(self) -> None:
        lo, hi = paired_bootstrap_ci(_POSITIVE_DELTAS, seed=42)
        mean = float(np.mean(_POSITIVE_DELTAS))
        assert lo < mean < hi

    def test_positive_shift_interval_above_zero(self) -> None:
        deltas = [0.05] * 30 + [
            0.01,
            0.02,
            0.04,
            0.06,
            0.03,
            0.02,
            0.05,
            0.01,
            0.03,
            0.04,
        ]
        lo, hi = paired_bootstrap_ci(deltas, seed=42)
        assert lo > 0.0

    def test_symmetric_data_crosses_zero(self) -> None:
        lo, hi = paired_bootstrap_ci(_CROSS_ZERO, seed=42)
        assert lo < 0.0 < hi

    def test_default_alpha_is_90pct(self) -> None:
        # 显式 alpha=0.10 应与默认一致
        lo_default, hi_default = paired_bootstrap_ci(_POSITIVE_DELTAS, seed=42)
        lo_10, hi_10 = paired_bootstrap_ci(_POSITIVE_DELTAS, seed=42, alpha=0.10)
        assert (lo_default, hi_default) == (lo_10, hi_10)

    def test_median_statistic(self) -> None:
        deltas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100.0]
        lo, hi = paired_bootstrap_ci(deltas, seed=42, statistic="median")
        assert lo <= hi
        assert lo > 0.0  # 中位数不受 100 污染，下界显著为正

    def test_wide_alpha_interval_wider(self) -> None:
        lo_10, hi_10 = paired_bootstrap_ci(_POSITIVE_DELTAS, seed=42, alpha=0.10)
        lo_50, hi_50 = paired_bootstrap_ci(_POSITIVE_DELTAS, seed=42, alpha=0.50)
        assert (hi_50 - lo_50) < (hi_10 - lo_10)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap_ci([], seed=42)
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0, 2.0], seed=42, statistic="bogus")
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0, 2.0], seed=42, alpha=1.2)


# ── Wilcoxon 符号秩 ────────────────────────────────────────────────────────────
class TestWilcoxonP:
    def test_all_positive_small(self) -> None:
        # m=5 全正：W+=15, mu=7.5, sigma=sqrt(13.75)=3.7081
        # z=(15-7.5-0.5)/3.7081=1.8877 → p=1-Phi(1.8877)=0.02953
        p = wilcoxon_p([1.0, 2.0, 3.0, 4.0, 5.0])
        assert p == pytest.approx(0.02953, abs=1e-4)

    def test_all_negative_greater_is_large(self) -> None:
        p = wilcoxon_p([-1.0, -2.0, -3.0, -4.0, -5.0], alternative="greater")
        assert p == pytest.approx(1 - 0.02953, abs=1e-4)

    def test_all_negative_less_is_small(self) -> None:
        p = wilcoxon_p([-1.0, -2.0, -3.0, -4.0, -5.0], alternative="less")
        assert p == pytest.approx(0.02953, abs=1e-4)

    def test_ties_average_rank(self) -> None:
        # [0.5,0.5,-0.5]：|.| 全并列，平均秩=2；W+=4, m=3, mu=3,
        # sigma=sqrt(3.5)=1.8708；z=(4-3-0.5)/1.8708=0.2673 → p=0.39463
        p = wilcoxon_p([0.5, 0.5, -0.5])
        assert p == pytest.approx(0.39463, abs=1e-4)

    def test_symmetric_zero_median_two_sided_large(self) -> None:
        # 正负对称（含并列）：W+ = W- = 18, mu=18 → p 接近 1
        p = wilcoxon_p(_CROSS_ZERO, alternative="two")
        assert p > 0.5

    def test_all_zero_returns_one(self) -> None:
        assert wilcoxon_p([0.0, 0.0, 0.0]) == 1.0

    def test_single_positive_half(self) -> None:
        assert wilcoxon_p([1.0]) == pytest.approx(0.5)

    def test_invalid_alternative(self) -> None:
        with pytest.raises(ValueError):
            wilcoxon_p([1.0, 2.0], alternative="two_sided")


# ── 描述统计 ───────────────────────────────────────────────────────────────────
class TestEffectSummary:
    def test_fields_and_counts(self) -> None:
        deltas = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
        s = effect_summary(deltas)
        assert set(s) == {
            "mean",
            "median",
            "trimmed_mean",
            "win_rate",
            "std",
            "n",
            "n_positive",
            "n_negative",
            "n_zero",
        }
        assert s["n"] == 6
        assert s["n_positive"] == 3
        assert s["n_negative"] == 2
        assert s["n_zero"] == 1
        assert s["win_rate"] == pytest.approx(3 / 6)
        assert s["mean"] == pytest.approx(0.03 / 6)
        assert s["median"] == pytest.approx((0.0 + 0.01) / 2)

    def test_trimmed_mean_drops_ends(self) -> None:
        # n=11 → k=1，去掉最小(0)和最大(100) → mean([1..9]) = 5.0
        deltas = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0]
        s = effect_summary(deltas)
        assert s["trimmed_mean"] == pytest.approx(5.0)
        assert s["mean"] != pytest.approx(5.0)

    def test_std_is_sample_std(self) -> None:
        deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = effect_summary(deltas)
        assert s["std"] == pytest.approx(np.std(deltas, ddof=1))

    def test_invalid_empty(self) -> None:
        with pytest.raises(ValueError):
            effect_summary([])


# ── 三档门禁 ───────────────────────────────────────────────────────────────────
class TestGateVerdict:
    def test_pass(self) -> None:
        v = gate_verdict((0.03, 0.09), point_est=0.05, cost_floor=DEFAULT_COST_FLOOR)
        assert v["verdict"] == "pass"
        assert "Phase 4" in v["reason"]
        assert v["ci"] == [0.03, 0.09]
        assert v["point_est"] == 0.05

    def test_abandon(self) -> None:
        v = gate_verdict((-0.05, -0.01), point_est=-0.03)
        assert v["verdict"] == "abandon"
        assert "放弃" in v["reason"]

    def test_insufficient_cross_zero(self) -> None:
        v = gate_verdict((-0.01, 0.04), point_est=0.015)
        assert v["verdict"] == "insufficient"
        assert "跨零" in v["reason"]

    def test_insufficient_below_cost_floor(self) -> None:
        # CI 下界 > 0 但点估计不过成本底线 → 仍是证据不足
        v = gate_verdict((0.005, 0.03), point_est=0.01, cost_floor=0.02)
        assert v["verdict"] == "insufficient"
        assert "成本底线" in v["reason"]

    def test_boundary_upper_zero_not_abandon(self) -> None:
        # 上界恰好 0 → 不算放弃（跨零含 0）
        v = gate_verdict((-0.02, 0.0), point_est=-0.005)
        assert v["verdict"] == "insufficient"

    def test_boundary_point_est_equals_cost_floor_not_pass(self) -> None:
        # 点估计恰好等于成本底线 → 不通过（严格 >）
        v = gate_verdict((0.01, 0.06), point_est=0.02, cost_floor=0.02)
        assert v["verdict"] == "insufficient"

    def test_name_appears_in_reason(self) -> None:
        v = gate_verdict((0.03, 0.09), point_est=0.05, name="r1r4")
        assert "r1r4" in v["reason"]

    def test_reversed_ci_raises(self) -> None:
        with pytest.raises(ValueError):
            gate_verdict((0.05, -0.02), point_est=0.0)

    def test_negative_cost_floor_raises(self) -> None:
        with pytest.raises(ValueError):
            gate_verdict((0.01, 0.05), point_est=0.03, cost_floor=-0.01)
