"""Phase 2a pilot —— 量化 agent 决策研究的行为与成本。

跑 N 个决策点 × K 次，量四件事（track_b_plan.md v2 §四 Phase 2a）：

1. **token 成本**：每次 run 写 ``llm_usage.json``（在 run_dir 里）。
2. **memo 解析率**：自由格式备忘录 → :class:`TradePlan` 的成败（解析失败 =
   协议违规 = 按不买，正是 pilot 要量的"解析失败率"）。
3. **grounding 吞买**：结论带"买"的备忘录是否被 grounding 拒绝 / 降级为
   "不买"（最终答案是拒绝式 = 强信号）。
4. **单点方差**：同点跑 K 次看决策是否稳定。

进程内调 :class:`AgentLoop`（与 CLI 同一构造路径），拿 ``run()`` 返回的完整
result：``content``（最终答案 = 备忘录）、``react_trace``（工具轨迹）、
``run_dir``。

**注意**：pilot 在"今天"实时跑（不模拟历史决策点），量的是机制（成本/解析/
grounding/方差），不是无前视的历史有效性——后者是 Phase 2b（工具沙箱 +
as-of 数据包）的事。

用法::

    PYTHONPATH=agent .venv/Scripts/python.exe -m trade_tools.pilot \\
        --symbols 000725.SZ,510300.SH --reps 2 --max-iter 12 --equity 100000 \\
        --out export_trade_data/track_b/pilot_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trade_tools.plan import TradePlan

# 决策点：symbol -> 一句话名称（提示词里给 agent，帮它环②"讲得清"）。
DEFAULT_POINTS: dict[str, str] = {
    "000725.SZ": "京东方A（面板/屏幕，LCD/OLED）",
    "510300.SH": "沪深300ETF（大盘指数）",
    "600519.SH": "贵州茅台（白酒龙头）",
}
# 备忘录里买/不买结论的常见措辞。
_REFUSAL_RE = re.compile(
    r"(我不能|我不会|无法给出|无法提供|不生成买入价|无法生成)", re.MULTILINE
)
# 元话语结尾：agent 说"收到/不再调工具/输出备忘录"却没真给结论（pilot 实测出现）。
_META_RE = re.compile(
    r"(不再调用工具|以现有已验证证据|收到，|收到\.|开始输出|即将输出备忘录)",
    re.MULTILINE,
)
# 结构化决策字段：优先"决策结论：X"行，其次"### 结论\nX"小节（pilot 实测：全文扫
# 会把标题里的"买入/不买入"误判成买意图——决策必须从小节读，不扫全文）。
_FINAL_DECISION_RE = re.compile(r"决策结论[：:]?\s*(买|不买|观望)", re.MULTILINE)
_SECTION_DECISION_RE = re.compile(r"#{1,4}\s*结论[^\n]*\n\s*([^\n]+)", re.MULTILINE)
_STOP_RE = re.compile(r"止损[:：]?\s*(\d+(?:\.\d+)?)")
_TARGET_RE = re.compile(r"目标[:：]?\s*(\d+(?:\.\d+)?)")
_POSITION_RE = re.compile(r"仓位[:：]?\s*(\d+(?:\.\d+)?)\s*%")


def _extract_decision(text: str) -> str | None:
    """从结构化决策字段读 buy / no_buy；未给出返回 None。

    顺序：① ``决策结论：买/不买`` 行；② ``### 结论`` 小节的第一行。两处都无 =
    未给出结论（元话语 / 解析失败）。
    """
    m = _FINAL_DECISION_RE.search(text)
    if m:
        return "buy" if m.group(1) == "买" else "no_buy"
    m = _SECTION_DECISION_RE.search(text)
    if m:
        line = m.group(1).strip()
        if "不买" in line or "观望" in line or "不买" in line:
            return "no_buy"
        if line == "买" or "买入" in line or "可买" in line:
            return "buy"
    return None


def _noop(*_a: Any, **_k: Any) -> None:
    """占位事件回调（渲染交给用户，pilot 只要结果）。"""


@dataclass
class PilotRecord:
    """一次 agent 决策研究的全部可测指标。"""

    symbol: str
    rep: int
    status: str  # success / failed
    iterations: int
    tokens: int  # 本次 run 总 token（llm_usage.json totals）
    content_len: int
    refusal: bool  # 最终答案是拒绝式（grounding 降级的强信号）
    buy_intent: bool  # 备忘录表达了"买"意图
    decision: str  # buy / no_buy / refusal / parse_failed
    parse_ok: bool
    stop: float | None = None
    target: float | None = None
    position_pct: float | None = None
    plan_reason: str = ""
    grounding_evidence_n: int = 0
    content: str = ""  # 原文，供人工复核

    def to_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "rep": self.rep,
            "status": self.status,
            "iterations": self.iterations,
            "tokens": self.tokens,
            "content_len": self.content_len,
            "refusal": self.refusal,
            "buy_intent": self.buy_intent,
            "decision": self.decision,
            "parse_ok": self.parse_ok,
            "stop": self.stop,
            "target": self.target,
            "position_pct": self.position_pct,
            "plan_reason": self.plan_reason,
            "grounding_evidence_n": self.grounding_evidence_n,
            "content": self.content,
        }


# ── memo 解析（量"解析失败率"）────────────────────────────────────────────────
def parse_memo(
    text: str, symbol: str, signal_date: str, signal_close: float
) -> tuple[TradePlan | None, str]:
    """尽力把自由格式备忘录解析成合法 TradePlan。

    Returns:
        (plan, reason)。解析失败返回 (None, 原因) —— fail closed，按不买。
        元话语结尾（"收到/不再调工具"却没给结论）单列为 ``no_decision`` 原因。
    """
    if not text:
        return None, "空备忘录"
    if _REFUSAL_RE.search(text):
        return None, "拒绝式回答（疑似 grounding 降级）"
    decision = _extract_decision(text)
    if decision == "no_buy":
        return (
            TradePlan(
                symbol=symbol,
                decision="no_buy",
                signal_date=signal_date,
                signal_close=signal_close,
            ),
            "",
        )
    if decision is None:
        if _META_RE.search(text):
            return None, "元话语结尾（未给买/不买结论）"
        return None, "未给出买/不买结论"

    # 结论=买 → 需要止损/目标/仓位，缺一不可。
    try:
        stop = float(_STOP_RE.search(text).group(1)) if _STOP_RE.search(text) else None
        target = (
            float(_TARGET_RE.search(text).group(1)) if _TARGET_RE.search(text) else None
        )
        pos = (
            float(_POSITION_RE.search(text).group(1)) / 100
            if _POSITION_RE.search(text)
            else None
        )
    except (AttributeError, ValueError):
        return None, "止损/目标/仓位解析失败"
    if stop is None or target is None or pos is None:
        return None, f"缺字段：stop={stop} target={target} position={pos}"

    plan = TradePlan(
        symbol=symbol,
        decision="buy",
        signal_date=signal_date,
        signal_close=signal_close,
        stop_price=stop,
        target_price=target,
        max_hold_days=20,
        position_pct=pos,
        account_equity=0.0,
    )
    violations = plan.validate()
    if violations:
        return None, f"协议校验失败：{'；'.join(violations)}"
    return plan, ""


# ── agent 运行 ────────────────────────────────────────────────────────────────
def _build_agent(max_iter: int, session_id: str = ""):
    """按 CLI 同一构造路径建 AgentLoop（懒 import agent 包）。"""
    from src.agent.loop import AgentLoop
    from src.config.loader import load_agent_config
    from src.memory.persistent import PersistentMemory
    from src.providers.chat import ChatLLM
    from src.tools import build_registry

    pm = PersistentMemory()
    agent_config = load_agent_config()
    return AgentLoop(
        registry=build_registry(
            persistent_memory=pm,
            # Track B 安全红线：绝不让 agent 在执行研究时跑 shell。
            include_shell_tools=False,
            agent_config=agent_config,
            session_id=session_id,
            warn_callback=lambda msg: None,
        ),
        llm=ChatLLM(),
        max_iterations=max_iter,
        persistent_memory=pm,
        event_callback=_noop,
        # Track B：关掉 grounding 价格校验——agent 自算的止损/目标价会被当
        # "未观测冲突/歧义"拒绝（pilot 首点实测 numeric_claim_ambiguous_symbol），
        # 导致决策丢失。Track B 的产出就是这些派生价，live 安全闸在此是反效果。
        grounding_enabled=False,
    )


def _memo_prompt(symbol: str, name: str) -> str:
    """构造备忘录研究提示词（按 ashare-trading-analyst skill 的流程）。"""
    return (
        f"你是资深 A 股交易分析员。先加载并读完 ashare-trading-analyst skill，"
        f"然后对 {symbol}（{name}）做买入/不买入决策研究。\n"
        f"研究流程（严格遵守 skill）：\n"
        f"环① 用 technical_indicators 判断大盘状态（510300.SH 的 MA200/MA60 方向）\n"
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


def _read_tokens(run_dir: str) -> int:
    try:
        data = json.loads(
            (Path(run_dir) / "llm_usage.json").read_text(encoding="utf-8")
        )
        return int(data["totals"]["total_tokens"])
    except Exception:  # noqa: BLE001
        return -1


def _read_grounding_evidence(run_dir: str) -> int:
    try:
        data = json.loads(
            (Path(run_dir) / "artifacts" / "grounding_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        return len(data.get("evidence", []))
    except Exception:  # noqa: BLE001
        return 0


def run_one(
    symbol: str,
    name: str,
    max_iter: int,
    signal_date: str,
    signal_close: float,
    rep: int,
) -> PilotRecord:
    """跑一次决策研究，返回全部指标。"""
    agent = _build_agent(max_iter)
    result = agent.run(user_message=_memo_prompt(symbol, name))
    content = result.get("content", "") or ""
    run_dir = result.get("run_dir", "")

    plan, reason = parse_memo(content, symbol, signal_date, signal_close)
    if plan is not None:
        decision = plan.decision  # buy / no_buy
    elif "元话语" in reason:
        decision = "no_decision"
    elif _REFUSAL_RE.search(content):
        decision = "refusal"
    else:
        decision = "parse_failed"
    return PilotRecord(
        symbol=symbol,
        rep=rep,
        status=result.get("status", "?"),
        iterations=int(result.get("iterations", 0)),
        tokens=_read_tokens(run_dir),
        content_len=len(content),
        refusal=bool(_REFUSAL_RE.search(content)),
        buy_intent=_extract_decision(content) == "buy",
        decision=decision,
        parse_ok=plan is not None,
        stop=plan.stop_price if plan is not None else None,
        target=plan.target_price if plan is not None else None,
        position_pct=plan.position_pct if plan is not None else None,
        plan_reason=reason,
        grounding_evidence_n=_read_grounding_evidence(run_dir),
        content=content,
    )


def _fetch_latest_close(symbol: str) -> float:
    """经 loader 抓取该标的最近一个交易日收盘价（懒 import agent 包）。"""
    from backtest.loaders.registry import resolve_loader

    loader = resolve_loader("a_share")
    data = loader.fetch([symbol], "2026-01-01", "2026-08-31", interval="1d")
    frame = data.get(symbol)
    if frame is None or frame.empty:
        raise ValueError(f"{symbol} 取不到最新收盘")
    return float(frame["close"].iloc[-1])


def run_pilot(
    points: dict[str, str],
    reps: int,
    max_iter: int,
    signal_date: str,
    signal_closes: dict[str, float],
) -> list[PilotRecord]:
    """按 (点 × rep) 顺序跑完，返回全部记录。"""
    records: list[PilotRecord] = []
    for symbol, name in points.items():
        close = signal_closes.get(symbol, 0.0)
        for rep in range(1, reps + 1):
            records.append(run_one(symbol, name, max_iter, signal_date, close, rep))
    return records


def _summary(records: list[PilotRecord]) -> dict:
    n = len(records)
    tokens = [r.tokens for r in records if r.tokens >= 0]
    buy = sum(1 for r in records if r.decision == "buy")
    parse = sum(1 for r in records if r.parse_ok)
    refusal = sum(1 for r in records if r.refusal)
    return {
        "runs": n,
        "tokens_total": sum(tokens),
        "tokens_per_run_avg": round(sum(tokens) / len(tokens)) if tokens else -1,
        "buy_ratio": buy / n if n else 0.0,
        "parse_ok_ratio": parse / n if n else 0.0,
        "refusal_ratio": refusal / n if n else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trade_tools.pilot", description="Phase 2a pilot"
    )
    parser.add_argument(
        "--symbols", default="000725.SZ,510300.SH", help="逗号分隔的决策点标的"
    )
    parser.add_argument("--reps", type=int, default=1, help="每点跑几次（量方差）")
    parser.add_argument(
        "--max-iter", type=int, default=16, help="agent 最大迭代数（限流）"
    )
    parser.add_argument("--signal-date", default="", help="信号日（默认今天）")
    parser.add_argument(
        "--signal-close",
        type=float,
        default=0.0,
        help="信号收盘价（0=让 agent 自己取）",
    )
    parser.add_argument("--out", default="", help="CSV 落盘路径")
    args = parser.parse_args(argv)

    points = {s: DEFAULT_POINTS.get(s, s) for s in args.symbols.split(",") if s}
    # 每点各自的最新收盘：--signal-close 只对单点跑有效，多点各自抓。
    signal_closes = {}
    for symbol in points:
        signal_closes[symbol] = (
            args.signal_close if args.signal_close > 0 else _fetch_latest_close(symbol)
        )
        if args.signal_close <= 0:
            print(f"{symbol} 最新收盘 {signal_closes[symbol]:.2f}")
    records = run_pilot(
        points, args.reps, args.max_iter, args.signal_date, signal_closes
    )

    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(records[0].to_row().keys()))
    writer.writeheader()
    for r in records:
        writer.writerow(r.to_row())

    if args.out:
        Path(args.out).write_text(buf.getvalue(), encoding="utf-8")
        print(f"已落盘 -> {args.out}")

    summary = _summary(records)
    print("=== Pilot 汇总 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    # 控制台明细省略 content（含 ✓ 等非 GBK 字符会崩 Windows 控制台）；完整版在 CSV。
    console_rows = [
        {k: ("" if k == "content" else v) for k, v in r.to_row().items()}
        for r in records
    ]
    print(f"=== 明细（{len(records)} 行，content 见 CSV）===")
    for row in console_rows:
        print(",".join(str(row[k]) for k in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
