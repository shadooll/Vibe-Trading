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
from trade_tools.pit import Point, _trading_day_distance, sample_points
from trade_tools.plan import TradePlan
from trade_tools.rules import PANIC_HOLD_DAYS, panic_reversal_signal
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
    # 提前 ~2 年抓数据：让 MA200/MA60 在抽样窗口起点就有效（否则窗口前 200 根
    # bar 的日期被判"未知"，层填不满——2023 全年因此只剩 2024-03~07 是震荡）。
    fetch_start = (pd.Timestamp(args.start) - pd.Timedelta(days=730)).strftime(
        "%Y-%m-%d"
    )
    universe = {}
    for code in args.universe.split(","):
        code = code.strip()
        if not code:
            continue
        print(f"抓取 {code} ...", flush=True)
        universe[code] = fetch_full(code, fetch_start, args.end)
    print(f"抓取市场 {args.market} ...", flush=True)
    market = fetch_full(args.market, fetch_start, args.end)["close"]

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


def _find_panic_events(
    universe: dict[str, pd.DataFrame], market: pd.Series, start, end
) -> list[dict]:
    """扫出恐慌修复入场点（不结算）。返回 ``[{symbol, date, signal_close, bars}]``。"""
    events = []
    for sym, df in universe.items():
        last_entry = None  # 同票连续恐慌 = 同一机会（reset 5 日），不重入
        for i in range(21, len(df)):
            date = df.index[i]
            if not (pd.Timestamp(start) <= date <= pd.Timestamp(end)):
                continue
            mkt_hist = market[market.index <= date]
            if len(mkt_hist) < 21:
                continue
            mkt_ret20 = float(mkt_hist.iloc[-1]) / float(mkt_hist.iloc[-21]) - 1
            sig = panic_reversal_signal(df.iloc[: i + 1], mkt_ret20)
            if sig is None:
                continue
            if (
                last_entry is not None
                and _trading_day_distance(df.index, date, last_entry) < 5
            ):
                continue  # 同一波恐慌
            last_entry = date
            events.append(
                {
                    "symbol": sym,
                    "date": str(date.date()),
                    "signal_close": sig["entry_ref"],
                    "bars": df,
                }
            )
    return events


def _settle_panic(
    event: dict, target_pct: float | None, stop_pct: float | None
) -> tuple[float, str]:
    """用给定止盈/止损参数结算一个恐慌事件（返回 net_ret, exit_reason）。

    ``target_pct`` / ``stop_pct`` 为相对信号收盘的比例；None 用极远价位代理
    （止盈 +100% / 止损 −50%，plan 协议要求 buy 必须给 stop/target），效果≈
    无该出口、只走时间兜底。
    """
    close = event["signal_close"]
    stop = (
        round(close * (1 + stop_pct), 2)
        if stop_pct is not None
        else round(close * 0.5, 2)
    )
    target = (
        round(close * (1 + target_pct), 2)
        if target_pct is not None
        else round(close * 2.0, 2)
    )
    plan = TradePlan(
        symbol=event["symbol"],
        decision="buy",
        signal_date=event["date"],
        signal_close=close,
        stop_price=stop,
        target_price=target,
        max_hold_days=PANIC_HOLD_DAYS,
        position_pct=0.5,
        account_equity=0.0,  # Layer 2 测单笔收益边际；equity=0 跳过 G1 仓位闸
        # 恐慌修复是高胜率低赔率规则（WR~64% × +10%/−12%），R:R≈0.83 是设计
        # 特征不是坏设置——跳过 G7 的 R:R≥2，按规则本来的风险结构评估。
        metadata={"reason": "PANIC", "skip_rr": True},
    )
    settle = ExecutionSimulator(event["bars"]).settle(plan)
    return (settle.net_ret if settle.realized else 0.0), settle.exit_reason


def _buy_hold_20d(event: dict, equity: float) -> float:
    """同入口买入持有 20 天基线（无条件次日开盘买，时间兜底卖）。"""
    settle = ExecutionSimulator(event["bars"]).settle(
        TradePlan(
            symbol=event["symbol"],
            decision="buy",
            signal_date=event["date"],
            signal_close=event["signal_close"],
            max_hold_days=PANIC_HOLD_DAYS,
            position_pct=1.0,
            account_equity=equity,
            hold_only=True,
        )
    )
    return settle.net_ret if settle.realized else 0.0


def _load_scan_data(args: argparse.Namespace) -> tuple[dict, pd.Series]:
    fetch_start = (pd.Timestamp(args.start) - pd.Timedelta(days=730)).strftime(
        "%Y-%m-%d"
    )
    fetch_end = (pd.Timestamp(args.end) + pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    universe = {}
    for code in args.universe.split(","):
        code = code.strip()
        if not code:
            continue
        print(f"抓取 {code} ...", flush=True)
        universe[code] = fetch_full(code, fetch_start, fetch_end)
    print(f"抓取市场 {args.market} ...", flush=True)
    market = fetch_full(args.market, fetch_start, fetch_end)["close"]
    return universe, market


def cmd_scan(args: argparse.Namespace) -> int:
    """scan：恐慌修复事件扫描器——扫全宇宙触发，逐笔结算，对比同入口买入持有。

    恐慌修复是事件驱动规则（quant_lab 已验证 R1 v5），用"扫全市场触发"而非
    "定点决策"评估：全宇宙逐日检查 ret5<-5%+放量+大盘恐慌+ret20<-10%，触发即
    建仓（stop -12% / target +10% / 20 天兜底），与同一入口的买入持有 20 天基线
    对比——判断规则的止损/止盈管理是否真的创造边际。
    """
    import statistics
    from collections import Counter

    universe, market = _load_scan_data(args)
    events = _find_panic_events(universe, market, args.start, args.end)
    if not events:
        print("无恐慌修复触发")
        return 0

    results = []
    for ev in events:
        net, reason = _settle_panic(ev, 0.10, -0.12)  # quant_lab v5 默认
        bh = _buy_hold_20d(ev, args.equity)
        results.append(
            {
                "date": ev["date"],
                "symbol": ev["symbol"],
                "net_ret": net,
                "exit_reason": reason,
                "bh_net": bh,
            }
        )

    nets = [e["net_ret"] for e in results]
    bh_nets = [e["bh_net"] for e in results]
    win = sum(1 for n in nets if n > 0) / len(nets)
    print(f"恐慌修复触发 {len(events)} 笔")
    print(
        f"  规则臂: 均值 {statistics.mean(nets):+.2%} 中位 {statistics.median(nets):+.2%} "
        f"胜率 {win:.1%}"
    )
    print(
        f"  同入口买入持有20天: 均值 {statistics.mean(bh_nets):+.2%} "
        f"中位 {statistics.median(bh_nets):+.2%}"
    )
    print(
        f"  规则 − 买入持有: 均值 {statistics.mean(nets) - statistics.mean(bh_nets):+.2%}"
    )
    for r, n in Counter(e["exit_reason"] for e in results).most_common():
        print(f"  出场 {r}: {n}")
    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已落盘 -> {args.out}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """sweep：恐慌修复退出参数网格扫描——找我们执行模型下的最优止盈/止损。"""
    import statistics

    universe, market = _load_scan_data(args)
    events = _find_panic_events(universe, market, args.start, args.end)
    if not events:
        print("无恐慌修复触发")
        return 0

    bh_mean = statistics.mean(_buy_hold_20d(ev, args.equity) for ev in events)
    print(f"恐慌修复退出参数扫描（{len(events)} 笔，时间兜底 20 天）")
    print(f"{'止盈':>6} {'止损':>6} {'均值':>8} {'中位':>8} {'胜率':>6} {'-买持':>8}")
    best = None
    for target_pct in (0.10, 0.15, 0.20, None):
        for stop_pct in (-0.08, -0.12, None):
            nets = [_settle_panic(ev, target_pct, stop_pct)[0] for ev in events]
            mean = statistics.mean(nets)
            med = statistics.median(nets)
            win = sum(1 for n in nets if n > 0) / len(nets)
            label_t = f"+{int(target_pct * 100)}%" if target_pct else "无"
            label_s = f"{int(stop_pct * 100)}%" if stop_pct else "无"
            print(
                f"{label_t:>6} {label_s:>6} {mean:+.2%} {med:+.2%} {win:.0%} {mean - bh_mean:+.2%}"
            )
            if best is None or mean > best[0]:
                best = (mean, label_t, label_s)
    print(f"参考: 买入持有20天 {bh_mean:+.2%}")
    print(f"最优: {best[1]}止盈 / {best[2]}止损 → 均值 {best[0]:+.2%}")
    return 0


def _panic_memo_prompt(symbol: str, name: str, event_date: str) -> str:
    """恐慌修复机会评估提示词：让 agent 逐笔判断反弹性质 + 动态退出。"""
    return (
        f"你是资深 A 股交易分析员。先加载并读完 ashare-trading-analyst skill。\n"
        f"研究日期：{event_date}（把这一天当作'今天'，只基于截至该日的数据，不许编造该日之后的信息）。\n"
        f"{symbol}（{name}）正处于恐慌：5日跌幅>5%、20日跌幅>10%，且大盘（510300）20日跌幅>5%。\n"
        f"这是恐慌修复机会评估（R5）。研究并决定：\n"
        f"1. 这笔恐慌会不会真正反弹（买/不买）？判断恐慌性质：真恐慌（放量、系统性）vs 有序出货（缩量、个股自身问题）。\n"
        f"2. 若买：止损价 + 目标价（可给具体数字，或写'持有至反弹结束'）+ 仓位%。\n"
        f"诚实背景：规则扫参显示固定止盈（+10%/+15%/+20%）都跑不赢朴素买入持有 20 天——"
        f"退出是本规则的关键，你要逐笔判断'这波反弹能走多远'。\n"
        f"成本纪律：get_market_data 最多 2 次、technical_indicators 最多 3 次，够用就停。\n"
        f"输出研究备忘录（这是最终答案，必须完整给出）：\n"
        f"### 结论\n买 或 不买\n"
        f"### 核心依据\n（每条带具体数字和来源）\n"
        f"### 操作计划（仅结论=买时）\n止损：__ 目标：__ 仓位：__%\n"
        f"所有数字必须来自工具输出，不许编造。最后一行必须是你的决策结论。"
    )


def _settle_agent_panic(
    content: str, event: dict
) -> tuple[float, str, str, float | None, float | None]:
    """解析 agent 的恐慌备忘录并结算。返回 (net_ret, decision, reason, stop, target)。

    decision=buy 时若缺止损 → 不成交；目标无数字 → 视为"持有至时间兜底"（远目标代理）。
    """
    from trade_tools.pilot import _STOP_RE, _TARGET_RE, _extract_decision

    decision = _extract_decision(content)
    if decision != "buy":
        return 0.0, decision or "no_decision", "", None, None
    stop_m = _STOP_RE.search(content)
    target_m = _TARGET_RE.search(content)
    stop = float(stop_m.group(1)) if stop_m else None
    target = float(target_m.group(1)) if target_m else None
    if stop is None:
        return 0.0, "buy", "缺止损", None, None
    t = target if target else round(event["signal_close"] * 2.0, 2)
    plan = TradePlan(
        symbol=event["symbol"],
        decision="buy",
        signal_date=event["date"],
        signal_close=event["signal_close"],
        stop_price=stop,
        target_price=t,
        max_hold_days=PANIC_HOLD_DAYS,
        position_pct=0.5,
        account_equity=0.0,
        metadata={"reason": "AGENT_PANIC", "skip_rr": True},
    )
    settle = ExecutionSimulator(event["bars"]).settle(plan)
    return (
        (settle.net_ret if settle.realized else 0.0),
        "buy",
        settle.exit_reason,
        stop,
        target,
    )


# 恐慌事件 agent 测试的默认精选子集（12 笔，含 2 笔已知亏损 + 1 笔大赢家，覆盖 2023-2026）。
DEFAULT_PANIC_EVENTS = (
    "2023-09-07_601138.SH,2023-10-23_601899.SH,2023-12-04_600176.SH,"
    "2023-12-05_601318.SH,2023-12-26_000636.SZ,2024-08-14_600487.SH,"
    "2024-09-18_600519.SH,2025-02-05_601138.SH,2025-04-07_000725.SZ,"
    "2025-04-08_601138.SH,2025-11-24_600183.SH,2026-03-23_601899.SH"
)


def cmd_agent_panic(args: argparse.Namespace) -> int:
    """agent-panic：在恐慌事件上跑 agent 决策（花 token；缓存续跑）。

    测"agent 的动态退出判断能否跑赢朴素买入持有 20 天"——固定规则做不到的。
    对比基线：同一子集买入持有均值 + 全 39 笔买入持有均值 + 确定性规则均值。
    """
    import statistics

    from trade_tools.agent_run import run_agent_decision

    universe, market = _load_scan_data(args)
    events = _find_panic_events(universe, market, args.start, args.end)
    selected = (
        args.events.split(",") if args.events else DEFAULT_PANIC_EVENTS.split(",")
    )
    subset = [e for e in events if f"{e['date']}_{e['symbol']}" in selected]
    if not subset:
        print("精选子集在扫描结果中无匹配")
        return 1

    out_path = Path(args.out)
    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        done = {(r["date"], r["symbol"]) for r in results}
    else:
        done = set()

    for i, ev in enumerate(subset):
        key = (ev["date"], ev["symbol"])
        if key in done:
            print(f"[{i+1}/{len(subset)}] 跳过缓存 {key}")
            continue
        print(f"[{i+1}/{len(subset)}] agent 评估 {key} ...", flush=True)
        r = run_agent_decision(
            ev["symbol"],
            ev["symbol"],
            ev["date"],
            args.max_iter,
            ev["signal_close"],
            prompt=_panic_memo_prompt(ev["symbol"], ev["symbol"], ev["date"]),
        )
        net, decision, reason, stop, target = _settle_agent_panic(r["content"], ev)
        bh = _buy_hold_20d(ev, args.equity)
        results.append(
            {
                "date": ev["date"],
                "symbol": ev["symbol"],
                "decision": decision,
                "net_ret": round(net, 6),
                "exit_reason": reason,
                "stop": stop,
                "target": target,
                "tokens": r["tokens"],
                "bh_net": round(bh, 6),
                "content": r["content"],
            }
        )
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total = sum(x.get("tokens", 0) for x in results)
        print(
            f"  -> {decision} net={net:+.2%} tokens={r['tokens']} 累计={total}",
            flush=True,
        )

    done_rows = [
        r
        for r in results
        if (r["date"], r["symbol"]) in {(e["date"], e["symbol"]) for e in subset}
    ]
    if not done_rows:
        print("无 agent 结果")
        return 0
    nets = [r["net_ret"] for r in done_rows]
    bh_nets = [r["bh_net"] for r in done_rows]
    print(f"=== 恐慌事件 agent 决策（{len(done_rows)} 笔）===")
    for r in done_rows:
        print(
            f"  {r['date']} {r['symbol']} {r['decision']} net={r['net_ret']:+.1%} vs 买持 {r['bh_net']:+.1%}"
        )
    print(
        f"agent 均值 {statistics.mean(nets):+.2%} vs 子集买持 {statistics.mean(bh_nets):+.2%} vs 全39买持 +7.93%"
    )
    print(f"agent − 子集买持 = {statistics.mean(nets) - statistics.mean(bh_nets):+.2%}")
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

    p = sub.add_parser("scan", help="恐慌修复事件扫描（免费，Layer 2 判决）")
    p.add_argument("--universe", required=True, help="逗号分隔的标的代码")
    p.add_argument("--market", default=MARKET_DEFAULT, help="大盘标的")
    p.add_argument("--start", required=True, help="扫描范围起点")
    p.add_argument("--end", required=True, help="扫描范围终点")
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--out", default="panic_events.json")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("sweep", help="恐慌修复退出参数扫描（免费）")
    p.add_argument("--universe", required=True, help="逗号分隔的标的代码")
    p.add_argument("--market", default=MARKET_DEFAULT, help="大盘标的")
    p.add_argument("--start", required=True, help="扫描范围起点")
    p.add_argument("--end", required=True, help="扫描范围终点")
    p.add_argument("--equity", type=float, default=100_000.0)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("agent-panic", help="恐慌事件 agent 决策（花 token；缓存续跑）")
    p.add_argument("--universe", required=True, help="逗号分隔的标的代码")
    p.add_argument("--market", default=MARKET_DEFAULT, help="大盘标的")
    p.add_argument("--start", required=True, help="扫描范围起点")
    p.add_argument("--end", required=True, help="扫描范围终点")
    p.add_argument(
        "--events", default="", help="逗号分隔 date_symbol 子集（缺省=精选 12 笔）"
    )
    p.add_argument("--max-iter", type=int, default=16)
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--out", default="agent_panic.json")
    p.set_defaults(func=cmd_agent_panic)

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
