"""Phase 2b 全量验证编排（track_b_plan.md v2 §四 Phase 2b）。

流水线：``sample_points``（市场 regime 分层）→ 每点三臂结算（agent / R1-R4 /
买入持有，共用同一 ExecutionSimulator）→ 配对差 → 三档门禁。

- ``generate``（免费）：PIT 分层抽样决策点（regime 用大盘判，v2 §三·2）。
- ``free``（免费）：R1-R4 确定性臂 + 买入持有基线，逐点结算。
- ``agent``（**花 token**）：as-of 沙箱跑 agent 研究，结果缓存到磁盘可续跑。
- ``report``：配对差 + 门禁判定。

数据约束：loader 有 ~500 行截断，抽样用 2023-01~2026-06 需分块抓取
（:func:`fetch_full`）；每个决策点的结算窗口窄（≤500 行）单次抓取即可。

用法::

    # 1. 生成决策点（免费）
    PYTHONPATH=agent .venv/Scripts/python.exe -m trade_tools.run_trackb generate \\
        --universe 000725.SZ,601138.SH --market 510300.SH \\
        --start 2023-01-01 --end 2026-06-01 --per-regime 10 --out points.json
    # 2. 免费臂（R1-R4 + 买入持有）
    ... run_trackb free --points points.json --equity 100000 --out free.json
    # 3. agent 臂（花 token；跑过缓存的点自动跳过）
    ... run_trackb agent --points points.json --max-iter 16 --out agent.json
    # 4. 门禁报告
    ... run_trackb report --points points.json --free free.json --agent agent.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trade_tools.execution import ExecutionSimulator
from trade_tools.pit import Point, sample_points
from trade_tools.validation import (
    make_buy_hold_plan,
    make_r1r4_plan,
    report,
)

# ── 常量 ─────────────────────────────────────────────────────────────────────
MARKET_DEFAULT = "510300.SH"
CHUNK_DAYS = 500  # 分块抓取跨度（loader 500 行截断的安全上限）
SETTLE_LOOKBACK_DAYS = 370  # 结算窗口：决策日前 ~370 天（够 MA200）
SETTLE_FORWARD_DAYS = 110  # 结算窗口：决策日后 ~110 天（60 交易日 + 出场日缓冲）


# ── 数据 ──────────────────────────────────────────────────────────────────────
def _loader():
    from backtest.loaders.registry import resolve_loader

    return resolve_loader("a_share")


def fetch_full(symbol: str, start: str, end: str) -> pd.DataFrame:
    """抓取全历史日线，自动分块拼接绕开 loader 500 行截断。

    返回归一化帧（DatetimeIndex / open high low close volume）。分块拼接的
    复权缝隙（qfq 每块独立调整）是已记录近似：对 regime 分层（比值类判定）
    影响极小。
    """
    loader = _loader()
    frames: list[pd.DataFrame] = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cursor < end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=CHUNK_DAYS), end_ts)
        data = loader.fetch(
            [symbol],
            cursor.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
            interval="1d",
        )
        df = data.get(symbol)
        if df is not None and len(df):
            frames.append(df)
        cursor = chunk_end + pd.Timedelta(days=1)
    if not frames:
        raise ValueError(f"{symbol} 在 {start}~{end} 无行情")
    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    return full[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_point_bars(symbol: str, decision_date: str) -> pd.DataFrame:
    """决策点的结算窗口：决策日前 370 天 ~ 后 90 天（单次抓取，≤500 行）。"""
    ts = pd.Timestamp(decision_date)
    start = (ts - pd.Timedelta(days=SETTLE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (ts + pd.Timedelta(days=SETTLE_FORWARD_DAYS)).strftime("%Y-%m-%d")
    return fetch_full(symbol, start, end)


def _points_from_json(path: str) -> list[Point]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Point(**p) for p in data]


def _dump_points(points: list[Point], path: str) -> None:
    Path(path).write_text(
        json.dumps(
            [{"symbol": p.symbol, "date": p.date, "regime": p.regime} for p in points],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ── 子命令 ────────────────────────────────────────────────────────────────────
def cmd_generate(args: argparse.Namespace) -> int:
    """generate：PIT 分层抽样决策点（regime 用大盘判）。"""
    universe = {}
    for code in args.universe.split(","):
        code = code.strip()
        if not code:
            continue
        print(f"抓取 {code} ...", flush=True)
        universe[code] = fetch_full(code, args.start, args.end)
    print(f"抓取市场 {args.market} ...", flush=True)
    market = fetch_full(args.market, args.start, args.end)["close"]

    points = sample_points(
        universe,
        args.start,
        args.end,
        args.per_regime,
        seed=args.seed,
        market_bars=market,
    )
    _dump_points(points, args.out)
    from collections import Counter

    counts = Counter(p.regime for p in points)
    print(f"决策点 {len(points)} 个 -> {args.out}")
    for regime, n in sorted(counts.items()):
        print(f"  {regime}: {n}")
    return 0


def _settle_arm(points: list[Point], maker, equity: float) -> list[dict]:
    """逐点结算一个对照臂（maker(point, bars, equity) -> TradePlan|None）。"""
    results = []
    for pt in points:
        try:
            bars = fetch_point_bars(pt.symbol, pt.date)
            plan = maker(pt, bars, equity)
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "symbol": pt.symbol,
                    "date": pt.date,
                    "regime": pt.regime,
                    "decision": "no_buy",
                    "reason": f"数据失败:{exc}",
                    "net_ret": 0.0,
                }
            )
            continue
        if plan is None:
            results.append(
                {
                    "symbol": pt.symbol,
                    "date": pt.date,
                    "regime": pt.regime,
                    "decision": "no_buy",
                    "reason": "未触发",
                    "net_ret": 0.0,
                }
            )
            continue
        settle = ExecutionSimulator(bars).settle(plan)
        net = settle.net_ret if settle.realized else 0.0
        results.append(
            {
                "symbol": pt.symbol,
                "date": pt.date,
                "regime": pt.regime,
                "decision": plan.decision,
                "reason": settle.exit_reason,
                "net_ret": round(net, 6),
                "filled": settle.filled,
            }
        )
    return results


def cmd_free(args: argparse.Namespace) -> int:
    """free：免费臂（R1-R4 + 买入持有）逐点结算。"""
    points = _points_from_json(args.points)
    free = {}
    free["r1r4"] = _settle_arm(points, make_r1r4_plan, args.equity)
    free["buy_hold"] = _settle_arm(points, make_buy_hold_plan, args.equity)
    Path(args.out).write_text(
        json.dumps(free, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_filled = sum(1 for r in free["r1r4"] if r.get("filled"))
    bh_filled = sum(1 for r in free["buy_hold"] if r.get("filled"))
    print(f"免费臂 -> {args.out}")
    print(f"  R1-R4: {len(points)} 点, 成交 {n_filled}")
    print(f"  买入持有: {len(points)} 点, 成交 {bh_filled}")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    """agent：as-of 沙箱跑 agent 臂（花 token；已缓存点自动跳过，可续跑）。"""
    from trade_tools.agent_run import run_agent_decision

    points = _points_from_json(args.points)
    out_path = Path(args.out)
    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        done = {(r["symbol"], r["date"]) for r in results}
    else:
        done = set()

    for i, pt in enumerate(points):
        key = (pt.symbol, pt.date)
        if key in done:
            print(f"[{i+1}/{len(points)}] 跳过缓存 {pt.symbol} @ {pt.date}")
            continue
        print(
            f"[{i+1}/{len(points)}] agent 研究 {pt.symbol} @ {pt.date} ...", flush=True
        )
        r = run_agent_decision(pt.symbol, pt.symbol, pt.date, args.max_iter)
        results.append(
            {
                "symbol": r["symbol"],
                "date": r["decision_date"],
                "regime": pt.regime,
                "decision": r["decision"],
                "parse_ok": r["parse_ok"],
                "tokens": r["tokens"],
                "stop": r["stop"],
                "target": r["target"],
                "position_pct": r["position_pct"],
                "net_ret": r["net_ret"],
                "exit_reason": r["exit_reason"],
            }
        )
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total = sum(x.get("tokens", 0) for x in results)
        print(f"  -> {r['decision']} tokens={r['tokens']} 累计={total}", flush=True)

    n = len(results)
    tokens = sum(x.get("tokens", 0) for x in results)
    print(f"agent 臂 {n}/{len(points)} 点, 累计 {tokens} token -> {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """report：agent vs 买入持有配对差 + 三档门禁。"""
    points = _points_from_json(args.points)
    free = json.loads(Path(args.free).read_text(encoding="utf-8"))
    agent_rows = json.loads(Path(args.agent).read_text(encoding="utf-8"))

    bh_by_key = {(r["symbol"], r["date"]): r for r in free["buy_hold"]}
    agent_by_key = {(r["symbol"], r["date"]): r for r in agent_rows}
    deltas = []
    matched = []
    for pt in points:
        key = (pt.symbol, pt.date)
        a, b = agent_by_key.get(key), bh_by_key.get(key)
        if a is None or b is None:
            continue
        d = a.get("net_ret", 0.0) - b["net_ret"]
        deltas.append(d)
        matched.append(
            {
                "symbol": pt.symbol,
                "date": pt.date,
                "regime": pt.regime,
                "agent_decision": a["decision"],
                "agent_net": a.get("net_ret", 0.0),
                "buy_hold_net": b["net_ret"],
                "delta": round(d, 6),
            }
        )
    if not deltas:
        print("无配对点——先跑 agent 臂再 report")
        return 0

    rep = report(deltas)
    print(f"配对 {len(deltas)} 点")
    for row in matched:
        print(
            f"  {row['date']} {row['symbol']} [{row['regime']}] "
            f"agent={row['agent_decision']}({row['agent_net']:+.1%}) "
            f"买入持有({row['buy_hold_net']:+.1%}) 差{row['delta']:+.1%}"
        )
    print("=== 主端点报告 ===")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade_tools.run_trackb", description="Phase 2b 全量验证编排"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="PIT 分层抽样决策点（免费）")
    p.add_argument("--universe", required=True, help="逗号分隔的标的代码")
    p.add_argument("--market", default=MARKET_DEFAULT, help="大盘标的（regime 判定）")
    p.add_argument("--start", required=True, help="决策日范围起点")
    p.add_argument("--end", required=True, help="决策日范围终点")
    p.add_argument("--per-regime", type=int, default=10, help="每层目标点数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="points.json")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("free", help="免费臂（R1-R4 + 买入持有，免费）")
    p.add_argument("--points", required=True)
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--out", default="free.json")
    p.set_defaults(func=cmd_free)

    p = sub.add_parser("agent", help="agent 臂（花 token；缓存续跑）")
    p.add_argument("--points", required=True)
    p.add_argument("--max-iter", type=int, default=16)
    p.add_argument("--out", default="agent.json")
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("report", help="门禁报告")
    p.add_argument("--points", required=True)
    p.add_argument("--free", required=True)
    p.add_argument("--agent", required=True)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # noqa: BLE001
        pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"错误：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
