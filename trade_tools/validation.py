"""Track B 验证流水线：决策点 → 对照决策者 → 逐点结算 → 配对差 → 三档门禁。

把 pit（决策点）、rules（R1-R4）、execution（结算）、stats（门禁）串成一条
可复现的流水线（track_b_plan.md v2 §三·4 / §三·5）。三条对照臂共用同一套结算：

- **买入持有**（:func:`make_buy_hold_plan`，hold_only）：决策日次日开盘无条件
  买入，持 60 交易日后卖——主基线（同票机会成本）。
- **R1-R4 确定性**（:func:`make_r1r4_plan` + :func:`rules.r1r4_signal`）：手册
  环③规则的确定性编码——描述性对照，不参与门禁。
- **agent**：Phase 2 接入（备忘录 → TradePlan → 同一结算）。本模块留好
  ``maker`` 接口，Phase 2a 把 LLM 决策接进来即可。

数据边界：agent 研究的**数据包**（as-of，无前视）与结算用的**全历史行情**
（决策日后 60 天）是两套数据。本模块的 runner 只做后者；前者由
:func:`pit.build_data_pack` 在 Phase 2 生成。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trade_tools.execution import ExecutionSimulator
from trade_tools.pit import Point
from trade_tools.plan import TradePlan
from trade_tools.rules import r1r4_signal
from trade_tools.stats import (
    BOOT_SEED,
    DEFAULT_COST_FLOOR,
    effect_summary,
    gate_verdict,
    paired_bootstrap_ci,
)

BUY_HOLD_DAYS = 60  # 买入持有基准：60 交易日（v2 结算窗口上限）
DEFAULT_HOLD_DAYS = 20  # R1-R4 交易：手册 4 周时间止损
MAX_POSITION_PCT = 1.0  # 仓位上限


@dataclass
class PointResult:
    """一个决策点在一名决策者下的结算结果。"""

    point: Point
    decision: str  # buy | no_buy
    reason: str  # 触发理由 / 放弃原因 / 空
    net_ret: float  # 已实现净收益；不买 / 未成交 = 0
    filled: bool
    exit_reason: str


def _asof_close(bars: pd.DataFrame, date: str | pd.Timestamp) -> float:
    """决策日（<= date 的最后一根 bar）收盘价；无数据则抛 ValueError。"""
    ts = pd.Timestamp(date)
    hist = bars[bars.index <= ts]
    if hist.empty:
        raise ValueError(f"{date} 无行情")
    return float(hist["close"].iloc[-1])


def _asof_bars(bars: pd.DataFrame, date: str | pd.Timestamp) -> pd.DataFrame:
    """截至决策日的 bars 切片（无前视）。"""
    return bars[bars.index <= pd.Timestamp(date)]


def make_buy_hold_plan(
    point: Point,
    bars: pd.DataFrame,
    account_equity: float,
    hold_days: int = BUY_HOLD_DAYS,
) -> TradePlan:
    """买入持有基线计划：无条件次日开盘买，持 ``hold_days`` 交易日。"""
    return TradePlan(
        symbol=point.symbol,
        decision="buy",
        signal_date=point.date,
        signal_close=_asof_close(bars, point.date),
        max_hold_days=hold_days,
        position_pct=1.0,  # 满仓暴露（基准语义）
        account_equity=account_equity,
        hold_only=True,
        metadata={"reason": "buy_hold"},
    )


def make_r1r4_plan(
    point: Point, bars: pd.DataFrame, account_equity: float
) -> TradePlan | None:
    """R1-R4 确定性交易计划；未触发返回 None（= 空仓）。"""
    asof = _asof_bars(bars, point.date)
    if len(asof) < 200:
        return None
    sig = r1r4_signal(asof, point.regime)
    if sig is None:
        return None
    entry = sig["entry_ref"]
    stop_dist = entry - sig["stop"]
    position_pct = (
        min(MAX_POSITION_PCT, 0.01 / (stop_dist / entry)) if stop_dist > 0 else 1.0
    )
    return TradePlan(
        symbol=point.symbol,
        decision="buy",
        signal_date=point.date,
        signal_close=_asof_close(asof, point.date),
        entry_ref=entry,
        stop_price=sig["stop"],
        target_price=sig["target"],
        max_hold_days=DEFAULT_HOLD_DAYS,
        position_pct=position_pct,
        account_equity=account_equity,
        metadata={"reason": sig["reason"]},
    )


def run_points(
    points: list[Point],
    bars_by_symbol: dict[str, pd.DataFrame],
    maker,
    account_equity: float,
) -> list[PointResult]:
    """对一批决策点跑同一个决策者（``maker(point, bars, equity) -> TradePlan|None``）。

    ``bars_by_symbol`` 必须是全历史（含决策日后 60 天），结算用；不买 / 未触发 /
    数据不足 / 未成交统一记 0 收益（v2 主端点口径：不买日记 0）。
    """
    results: list[PointResult] = []
    for pt in points:
        bars = bars_by_symbol.get(pt.symbol)
        if bars is None or len(bars) < 200:
            results.append(PointResult(pt, "no_buy", "数据不足", 0.0, False, ""))
            continue
        plan = maker(pt, bars, account_equity)
        if plan is None:
            results.append(PointResult(pt, "no_buy", "未触发", 0.0, False, ""))
            continue
        settle = ExecutionSimulator(bars).settle(plan)
        net = settle.net_ret if settle.realized else 0.0
        results.append(
            PointResult(
                pt,
                plan.decision,
                plan.metadata.get("reason", ""),
                net,
                settle.filled,
                settle.exit_reason,
            )
        )
    return results


def paired_deltas(
    agent_results: list[PointResult], baseline_results: list[PointResult]
) -> list[float]:
    """逐点配对差 agent − baseline（同序、同点）；不买 / 未成交记 0。"""
    return [a.net_ret - b.net_ret for a, b in zip(agent_results, baseline_results)]


def report(deltas: list[float], cost_floor: float = DEFAULT_COST_FLOOR) -> dict:
    """主端点报告：全分布描述 + bootstrap CI + 三档门禁。"""
    summary = effect_summary(deltas)
    ci = paired_bootstrap_ci(deltas, seed=BOOT_SEED)
    verdict = gate_verdict(ci, summary["mean"], cost_floor=cost_floor)
    return {
        "n": len(deltas),
        "summary": summary,
        "bootstrap_ci": list(ci),
        "gate": verdict,
    }
