"""Phase 2b sandbox —— 历史决策点的 as-of agent 研究。

Track B 的"无前视"保障（track_b_plan.md v2 §三·1）：让 agent 在一个历史决策日
研究时，把数据工具的日期**锚死到决策日**，物理上摸不到未来数据（评审 CRITICAL 1
的解法）。

- 只注册 as-of 安全的工具子集：
  - ``technical_indicators`` / ``get_market_data`` —— 用 :class:`_PinnedDateTool`
    强制 ``end_date=决策日``（agent 传什么都覆盖，保证 end <= 决策日）。
  - ``load_skill`` —— 静态知识，不算前视。
  - 其余全禁（screen_market / web_search / get_fundamentals / get_sector_info /
    get_fund_flow / bash / ...）。
- 备忘录解析、token 计量复用 :mod:`trade_tools.pilot`。
- 结算走 :mod:`trade_tools.validation`（对每点全历史），见 ``run_points``。

依赖 agent 包（Phase 2b 专用，不进入 trade_tools/__init__ 导出）。

用法（跑一个历史决策点）::

    PYTHONPATH=agent .venv/Scripts/python.exe -m trade_tools.agent_run \\
        --symbol 000725.SZ --name 京东方A --date 2024-05-15 --max-iter 16
"""

from __future__ import annotations

import argparse
from typing import Any

from src.agent.tools import BaseTool, ToolRegistry

from trade_tools.execution import ExecutionSimulator
from trade_tools.pilot import _noop, _read_tokens, parse_memo


# ── 无前视保障：强制工具日期 ──────────────────────────────────────────────────
class _PinnedDateTool(BaseTool):
    """包装一个数据工具，强制它的日期参数（无前视保障）。

    ``forced`` 里的 kwargs 无条件覆盖 agent 传的参数——即使 agent 传了未来的
    end_date 或没传（默认今天），都被覆盖成决策日。
    """

    def __init__(self, inner: BaseTool, **forced: Any) -> None:
        self._inner = inner
        self._forced = forced
        self.name = inner.name
        self.description = inner.description
        self.parameters = inner.parameters
        self.repeatable = getattr(inner, "repeatable", False)
        self.is_readonly = getattr(inner, "is_readonly", True)

    def execute(self, **kwargs: Any) -> str:
        kwargs.update(self._forced)  # 覆盖日期参数
        return self._inner.execute(**kwargs)


def build_asof_registry(
    end_date: str, *, persistent_memory: Any = None
) -> ToolRegistry:
    """Phase 2b 沙箱注册表：只含 as-of 安全的工具子集。"""
    from src.config.loader import load_agent_config
    from src.tools import build_registry

    full = build_registry(
        include_shell_tools=False,
        persistent_memory=persistent_memory,
        agent_config=load_agent_config(),
    )
    registry = ToolRegistry()
    for name in ("technical_indicators", "get_market_data"):
        tool = full.get(name)
        if tool is not None:
            registry.register(_PinnedDateTool(tool, end_date=end_date))
    skill = full.get("load_skill")
    if skill is not None:
        registry.register(skill)
    return registry


def _build_agent(end_date: str, max_iter: int) -> Any:
    """构造 as-of 沙箱 agent（grounding 关闭：Track B 产出就是派生价）。"""
    from src.agent.loop import AgentLoop
    from src.memory.persistent import PersistentMemory
    from src.providers.chat import ChatLLM

    pm = PersistentMemory()
    return AgentLoop(
        registry=build_asof_registry(end_date, persistent_memory=pm),
        llm=ChatLLM(),
        max_iterations=max_iter,
        persistent_memory=pm,
        event_callback=_noop,
        grounding_enabled=False,
    )


def _memo_prompt(symbol: str, name: str, decision_date: str) -> str:
    """构造备忘录研究提示词，把决策日锚定为"今天"（无前视的时间上下文）。"""
    return (
        f"你是资深 A 股交易分析员。先加载并读完 ashare-trading-analyst skill。\n"
        f"研究日期：{decision_date}（把这一天当作'今天'。你拿到的行情已自动锚定到"
        f"该日，只能基于截至该日的数据分析，不许编造该日之后的信息）。\n"
        f"对 {symbol}（{name}）做买入/不买入决策研究。\n"
        f"研究流程（严格遵守 skill）：\n"
        f"环① 用 technical_indicators 判断大盘状态（510300.SH 在 {decision_date} 的 MA200/MA60 方向）\n"
        f"环② 检查 {symbol} 观察池资格（讲得清/流动性/非创业板）\n"
        f"环③ 用 technical_indicators 检查 R1-R4 是否触发（趋势→R1/R2，震荡→R3/R4）\n"
        f"环④ 若触发：先定止损（前低或-3%），目标=入场+2×止损距离（R:R>=2），"
        f"仓位=账户×1%÷止损距离\n"
        f"成本纪律：get_market_data 最多 2 次、technical_indicators 最多 3 次，"
        f"够用就停，禁止重复抓取。\n"
        f"输出研究备忘录（这是最终答案，必须完整给出，不要只说要输出）：\n"
        f"### 结论\n买 或 不买\n"
        f"### 核心依据\n（每条带具体数字和来源）\n"
        f"### 操作计划（仅结论=买时）\n止损：__ 目标：__ 仓位：__%\n"
        f"所有数字必须来自工具输出，不许编造。最后一行必须是你的决策结论。"
    )


def run_agent_decision(
    symbol: str,
    name: str,
    decision_date: str,
    max_iter: int = 16,
    signal_close: float = 0.0,
) -> dict:
    """在一个历史决策日跑一次 agent 研究，返回 memo + 解析 + 成本指标。"""
    agent = _build_agent(decision_date, max_iter)
    result = agent.run(user_message=_memo_prompt(symbol, name, decision_date))
    content = result.get("content", "") or ""
    run_dir = result.get("run_dir", "")
    plan, reason = parse_memo(content, symbol, decision_date, signal_close)
    decision = plan.decision if plan is not None else f"解析失败:{reason}"

    # 结算 agent 备忘录（若解析出合法 buy 计划）：同买入持有共用 ExecutionSimulator。
    net_ret, exit_reason = 0.0, ""
    if plan is not None and plan.decision == "buy":
        try:
            from trade_tools.run_trackb import fetch_point_bars

            settle = ExecutionSimulator(fetch_point_bars(symbol, decision_date)).settle(
                plan
            )
            net_ret = settle.net_ret if settle.realized else 0.0
            exit_reason = settle.exit_reason
        except Exception as exc:  # noqa: BLE001
            net_ret, exit_reason = 0.0, f"结算失败:{exc}"

    return {
        "symbol": symbol,
        "decision_date": decision_date,
        "status": result.get("status", "?"),
        "iterations": int(result.get("iterations", 0)),
        "tokens": _read_tokens(run_dir),
        "content_len": len(content),
        "decision": decision,
        "parse_ok": plan is not None,
        "stop": plan.stop_price if plan is not None else None,
        "target": plan.target_price if plan is not None else None,
        "position_pct": plan.position_pct if plan is not None else None,
        "net_ret": net_ret,
        "exit_reason": exit_reason,
        "content": content,
        "run_dir": run_dir,
    }


def main(argv: list[str] | None = None) -> int:
    import sys

    # Windows 控制台默认 GBK，备忘录里的 ✓ 等字符会崩打印——stdout 重配为 UTF-8。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        prog="trade_tools.agent_run",
        description="Phase 2b as-of agent 研究（单个历史决策点）",
    )
    parser.add_argument("--symbol", required=True, help="标的代码，如 000725.SZ")
    parser.add_argument("--name", default="", help="标的名称（提示词里给 agent）")
    parser.add_argument("--date", required=True, help="决策日 YYYY-MM-DD")
    parser.add_argument("--max-iter", type=int, default=16, help="agent 最大迭代数")
    parser.add_argument(
        "--signal-close", type=float, default=0.0, help="决策日收盘价（0=让工具取）"
    )
    args = parser.parse_args(argv)

    r = run_agent_decision(
        args.symbol, args.name, args.date, args.max_iter, args.signal_close
    )
    print(
        f"决策点 {r['symbol']} @ {r['decision_date']}：status={r['status']} "
        f"iter={r['iterations']} tokens={r['tokens']}"
    )
    print(f"决策：{r['decision']} 解析={r['parse_ok']}")
    print("=== 备忘录 ===")
    print(r["content"])
    print(f"run_dir: {r['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
