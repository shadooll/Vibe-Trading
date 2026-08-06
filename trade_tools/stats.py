"""Track B 验证框架的统计层：功效分析 + 配对 bootstrap/Wilcoxon + 三档门禁。

对应 ``export_trade_data/track_b_plan.md`` v2 §三·4（主端点 d_i 配对差）与
§三·5（统计设计）：

- **功效分析先行**（§三·5·1）：:func:`required_n` / :func:`power_table` 按
  单点方差反推样本量（σ≈20%、δ=5% 时约需 n≈100，预算按 60-100 算）。
- **检验用配对 bootstrap CI + Wilcoxon 符号秩**（§三·5·4）：60 天收益偏斜
  厚尾，正态假设不成立；bootstrap 报告区间，Wilcoxon 只作单侧 p 值佐证。
- **三档门禁**（§三·5·3 / §四 Phase 3）：:func:`gate_verdict` 按 bootstrap
  CI + 成本底线给出 通过 / 放弃 / 证据不足。

依赖：只允许 numpy（± pandas）与标准库（``statistics.NormalDist`` 算正态
分位/尾概率），不引入 scipy——避免给仓库加新依赖（requirements-lock.txt
哈希锁定，加依赖要重生成锁文件，CI 会拦）。
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

# ── 常数（命名常量，语义一目了然）──────────────────────────────────────────────
DEFAULT_ALPHA = 0.05  # 显著性水平（默认 5%）
DEFAULT_POWER = 0.80  # 目标功效
DEFAULT_COST_FLOOR = 0.02  # 成本底线：覆盖费用+滑点（约 +2%/60 天）
DEFAULT_BOOT_ALPHA = 0.10  # bootstrap 置信区间 alpha（默认 90% CI）
DEFAULT_N_BOOT = 5000  # bootstrap 重采样次数
BOOT_SEED = 42  # 框架建议的固定 bootstrap 种子（门禁判定要可复现）
TRIMMED_MEAN_PCT = 0.10  # 描述统计去两端比例（10% 每端）
WILCOXON_EXACT_N = 25  # n < 25 时 Wilcoxon 正态近似不精确（精确 p 需穷举 2^n）


# ── 功效分析（§三·5·1）─────────────────────────────────────────────────────────
def required_n(
    delta: float,
    sigma: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    side: str = "one",
) -> int:
    """配对设计所需样本量（单样本检验功效分析）。

    公式：n = ((z_{1-alpha} + z_power) * sigma / delta)^2。``side="one"`` 用
    ``z_{1-alpha}``（单侧），``side="two"`` 用 ``z_{1-alpha/2}``（双侧）。
    z 分位用标准库 ``statistics.NormalDist().inv_cdf``（无 scipy）。返回
    向上取整。

    Args:
        delta: 可检测的效应量 = 主端点（配对差）均值（> 0）。
        sigma: 配对差的单点标准差（> 0）。
        alpha: 显著性水平（0, 1）。
        power: 目标功效（0, 1）。
        side: 单侧 ``"one"`` 或双侧 ``"two"``。
    """
    if delta <= 0:
        raise ValueError(f"delta 必须 > 0（可检测效应量），得到 {delta}")
    if sigma <= 0:
        raise ValueError(f"sigma 必须 > 0（配对差单点标准差），得到 {sigma}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0, 1) 内，得到 {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power 必须在 (0, 1) 内，得到 {power}")
    if side not in ("one", "two"):
        raise ValueError(f"side 必须是 'one' 或 'two'，得到 {side!r}")

    tail = 1 - alpha if side == "one" else 1 - alpha / 2
    z_alpha = NormalDist().inv_cdf(tail)
    z_power = NormalDist().inv_cdf(power)
    return int(math.ceil(((z_alpha + z_power) * sigma / delta) ** 2))


def power_table(
    deltas: list[float],
    sigma: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    side: str = "one",
) -> list[dict]:
    """对一组 delta 输出 (delta, n_required) 功效表（Phase 2 前给用户看）。

    例：``power_table([0.02, 0.03, 0.05], 0.2)`` → ``[{"delta": 0.02,
    "n_required": 619}, ...]``。delta 越小所需样本越多，单调递减。
    """
    return [
        {
            "delta": float(d),
            "n_required": required_n(d, sigma, alpha=alpha, power=power, side=side),
        }
        for d in deltas
    ]


# ── 配对 bootstrap CI（§三·5·4）────────────────────────────────────────────────
def paired_bootstrap_ci(
    deltas: list[float] | np.ndarray,
    alpha: float = DEFAULT_BOOT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int | None = None,
    statistic: str = "mean",
) -> tuple[float, float]:
    """配对差的 bootstrap 分位区间。

    对配对差数组 d_i 放回重采样 ``n_boot`` 次，取重采样统计量分布的
    ``[alpha/2, 1 - alpha/2]`` 分位作为 (下界, 上界)。``alpha=0.10`` 即
    90% 区间（门禁用 90%）。``statistic`` 支持 ``"mean"``（默认）与
    ``"median"``。

    可复现：传入固定 ``seed`` 结果完全一致；门禁判定应固定 seed
    （见 :data:`BOOT_SEED`），否则 bootstrap 随机性可能翻动边界判定。
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("deltas 必须是非空一维数组")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0, 1) 内，得到 {alpha}")
    if n_boot < 1:
        raise ValueError(f"n_boot 必须 >= 1，得到 {n_boot}")
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic 必须是 'mean' 或 'median'，得到 {statistic!r}")

    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    boot = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


# ── Wilcoxon 符号秩（§三·5·4）──────────────────────────────────────────────────
def wilcoxon_p(deltas: list[float] | np.ndarray, alternative: str = "greater") -> float:
    """Wilcoxon 符号秩检验的 p 值（numpy 实现，正态近似）。

    零假设：配对差中位数 = 0。流程：剔除零差 → 对 |d_i| 排秩（并列取平均
    秩）→ 正秩和 W+ → 正态近似 z = (W+ - mu ± 0.5)/sigma（连续性校正，
    方向趋向 H0）→ 尾概率。

    近似限制（如实写明）：正态近似在非零样本 m < :data:`WILCOXON_EXACT_N`
    （25）时不精确（精确 p 需穷举 2^m 种符号组合）；并列秩较多时未做方差
    校正，结果偏保守。本模块是验证框架，m 小时应结合
    :func:`paired_bootstrap_ci` 看区间，不单看 p 值。

    Args:
        deltas: 配对差数组。
        alternative: ``"greater"``（默认，H1: 中位数 > 0）/ ``"less"`` /
            ``"two"``（双侧）。
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.ndim != 1:
        raise ValueError("deltas 必须是一维数组")
    if alternative not in ("greater", "less", "two"):
        raise ValueError(
            f"alternative 必须是 'greater'/'less'/'two'，得到 {alternative!r}"
        )

    nonzero = arr[arr != 0]
    m = nonzero.size
    if m == 0:
        return 1.0  # 全零：无从拒绝 H0

    abs_vals = np.abs(nonzero)
    order = np.argsort(abs_vals, kind="stable")
    sorted_abs = abs_vals[order]
    _, inverse, counts = np.unique(sorted_abs, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts  # 每组在排序后数组中的起点（0-indexed）
    avg_ranks = (starts + 1 + starts + counts) / 2.0  # (首秩+末秩)/2
    rank_sorted = avg_ranks[inverse]
    rank_out = np.empty(m, dtype=float)
    rank_out[order] = rank_sorted
    wplus = float(rank_out[nonzero > 0].sum())

    mu = m * (m + 1) / 4.0
    sigma = math.sqrt(m * (m + 1) * (2 * m + 1) / 24.0)
    if wplus > mu:
        z = (wplus - mu - 0.5) / sigma
    else:
        z = (wplus - mu + 0.5) / sigma
    nd = NormalDist()
    if alternative == "two":
        return min(1.0, 2.0 * nd.cdf(-abs(z)))
    if alternative == "greater":
        return float(nd.cdf(-z))
    return float(nd.cdf(z))


# ── 描述统计（§三·4 主端点全分布报告）─────────────────────────────────────────
def effect_summary(deltas: list[float] | np.ndarray) -> dict:
    """主端点全分布描述统计（不用单一均值表态）。

    返回 ``mean / median / trimmed_mean(去 10% 两端) / win_rate(>0 占比) /
    std(样本, ddof=1) / n / n_positive / n_negative / n_zero``。零差 = 不买
    日记 0 的点，单独计数供审计。
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("deltas 必须是非空一维数组")
    n = arr.size
    k = int(n * TRIMMED_MEAN_PCT)
    n_pos = int((arr > 0).sum())
    n_neg = int((arr < 0).sum())
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "trimmed_mean": float(np.sort(arr)[k : n - k].mean()),
        "win_rate": n_pos / n,
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n - n_pos - n_neg,
    }


# ── 三档门禁（§三·5·3 / §四 Phase 3）───────────────────────────────────────────
def gate_verdict(
    ci: tuple[float, float] | list[float],
    point_est: float,
    cost_floor: float = DEFAULT_COST_FLOOR,
    name: str = "agent",
) -> dict:
    """三档门禁判定（Track B 主端点）。

    - **通过**：CI 下界 > 0 且 点估计 > ``cost_floor`` → 进 Phase 4；
    - **放弃**：CI 上界 < 0 → 永久放弃 Track B，只留 Track A；
    - **证据不足**：CI 跨零，或"下界 > 0 但点估计不过成本底线" → 4 周
      watch-only 试运行继续采集。

    Args:
        ci: bootstrap CI ``(下界, 上界)``。
        point_est: 主端点点估计（配对差均值）。
        cost_floor: 成本底线（覆盖费用+滑点，默认 +2%/60 天）。
        name: 被测臂名称，用于审计文案（默认 ``agent``）。

    Returns:
        ``{"verdict": "pass"|"abandon"|"insufficient", "ci": [lo, hi],
        "point_est": ..., "reason": 中文说明}``。
    """
    lo, hi = float(ci[0]), float(ci[1])
    if lo > hi:
        raise ValueError(f"CI 下界 {lo} > 上界 {hi}")
    if cost_floor < 0:
        raise ValueError(f"cost_floor 必须 >= 0，得到 {cost_floor}")

    if hi < 0:
        verdict = "abandon"
        reason = (
            f"{name} 配对差 90% bootstrap CI 上界 {hi:+.2%} < 0 → "
            "永久放弃 Track B，只留 Track A"
        )
    elif lo > 0 and point_est > cost_floor:
        verdict = "pass"
        reason = (
            f"{name} 配对差 CI 下界 {lo:+.2%} > 0 且点估计 {point_est:+.2%} "
            f"> 成本底线 {cost_floor:.2%} → 进 Phase 4"
        )
    elif lo > 0:
        verdict = "insufficient"
        reason = (
            f"{name} 配对差 CI 下界 {lo:+.2%} > 0 但点估计 {point_est:+.2%} "
            f"未过成本底线 {cost_floor:.2%} → 证据不足，watch-only"
        )
    else:
        verdict = "insufficient"
        reason = (
            f"{name} 配对差 CI [{lo:+.2%}, {hi:+.2%}] 跨零 → 证据不足，"
            "进 4 周 watch-only 试运行"
        )
    return {
        "verdict": verdict,
        "ci": [lo, hi],
        "point_est": float(point_est),
        "reason": reason,
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_POWER",
    "DEFAULT_COST_FLOOR",
    "DEFAULT_BOOT_ALPHA",
    "DEFAULT_N_BOOT",
    "BOOT_SEED",
    "TRIMMED_MEAN_PCT",
    "WILCOXON_EXACT_N",
    "required_n",
    "power_table",
    "paired_bootstrap_ci",
    "wilcoxon_p",
    "effect_summary",
    "gate_verdict",
]
