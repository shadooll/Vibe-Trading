"""trade_tools — 泛化的 A 股纸面交易基础设施。

把交易系统的执行现实（T+1、涨跌停、停牌、gap、滑点、费用、G1-G8 护栏）
编码成可复用的模块。Track A（手工 7 环链）和 Track B（agent 决策验证）
共用同一套口径，避免"手册一套、回测一套"。

模块:
    plan.py       结构化交易计划协议（决策 -> 可执行计划的契约）
    execution.py  A 股纸面交易模拟器（一份计划 + 行情 -> 已实现盈亏）
    gate.py       G1-G8 决策闸（从手册编码的确定性过滤器）
    pit.py        PIT 无幸存者抽样 + as-of 数据包生成器（已实现）
    rules.py      R1-R4 确定性入场理由引擎（手册环③编码）
    validation.py Track B 验证流水线（对照臂 + 逐点结算 + 门禁报告）
    stats.py      功效分析 + 配对 bootstrap/Wilcoxon + 三档门禁（已实现）
    cli.py        统一命令行入口（已实现）

数据输出不落在这里——数据归数据，输出到 export_trade_data/ 下。
"""

from __future__ import annotations

from trade_tools.plan import TradePlan, ProtocolViolation
from trade_tools.execution import ExecutionSimulator, SettlementResult
from trade_tools.gate import GateContext, GateVerdict, gate
from trade_tools.pit import (
    Point,
    UniverseFilter,
    UniverseVerdict,
    build_data_pack,
    compute_snapshot,
    regime_at,
    regime_stratum,
    sample_points,
)
from trade_tools.rules import r1r4_signal, r1_signal, r2_signal, r3_signal, r4_signal
from trade_tools.validation import (
    PointResult,
    make_buy_hold_plan,
    make_r1r4_plan,
    paired_deltas,
    report,
    run_points,
)
from trade_tools.stats import (
    BOOT_SEED,
    DEFAULT_ALPHA,
    DEFAULT_BOOT_ALPHA,
    DEFAULT_COST_FLOOR,
    DEFAULT_N_BOOT,
    DEFAULT_POWER,
    TRIMMED_MEAN_PCT,
    WILCOXON_EXACT_N,
    effect_summary,
    gate_verdict,
    paired_bootstrap_ci,
    power_table,
    required_n,
    wilcoxon_p,
)

__all__ = [
    "TradePlan",
    "ProtocolViolation",
    "ExecutionSimulator",
    "SettlementResult",
    "GateContext",
    "GateVerdict",
    "gate",
    "Point",
    "UniverseFilter",
    "UniverseVerdict",
    "build_data_pack",
    "compute_snapshot",
    "regime_at",
    "regime_stratum",
    "sample_points",
    "r1r4_signal",
    "r1_signal",
    "r2_signal",
    "r3_signal",
    "r4_signal",
    "PointResult",
    "make_buy_hold_plan",
    "make_r1r4_plan",
    "paired_deltas",
    "report",
    "run_points",
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
