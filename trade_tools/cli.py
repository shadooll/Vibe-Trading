"""统一命令行入口（argparse 子命令）。

把 plan / execution / gate / pit / stats 串成一个命令行工具，供个人 A 股
交易者（Windows）直接跑纸面交易、决策点抽样与门禁判定：

- ``check``  : plan.json -> TradePlan -> G1-G8 闸 -> 通过/拦截
- ``settle`` : plan.json + 不复权日线 CSV -> ExecutionSimulator 纸面结算
- ``pack``   : as-of 数据包（``--csv`` 走本地离线路径，否则经 loader 抓取）
- ``sample`` : PIT 分层决策点抽样（纯本地，不碰网络）
- ``power``  : 功效分析表（Phase 2 前看样本量）
- ``verdict``: 三档门禁判定（通过 / 放弃 / 证据不足）

依赖说明：解析与文件读写只用标准库（argparse/json/csv/pathlib）；构造
``pandas.DataFrame`` 仅为喂给 execution/pit 的既有 API（这两个模块本就依赖
pandas），未引入任何新的第三方依赖。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from trade_tools.execution import ExecutionSimulator
from trade_tools.gate import GateContext, gate
from trade_tools.pit import build_data_pack, sample_points
from trade_tools.plan import TradePlan
from trade_tools.stats import (
    DEFAULT_ALPHA,
    DEFAULT_COST_FLOOR,
    DEFAULT_POWER,
    gate_verdict,
    power_table,
)

# ── 命名常量 ────────────────────────────────────────────────────────────────
DEFAULT_LOOKBACK = 400  # pack 默认回溯交易日数
DEFAULT_DELTAS = (0.02, 0.03, 0.05)  # power 默认 delta 列表
DEFAULT_SIGMA = 0.2  # power 默认配对差单点标准差
DEFAULT_SIDE = "one"  # power 默认单侧检验
DEFAULT_SEED = 42  # sample 默认随机种子
DEFAULT_MIN_SPACING = 60  # sample 默认同层最小交易日间距
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
_VERDICT_LABELS = {"pass": "通过", "abandon": "放弃", "insufficient": "证据不足"}


# ── 读取辅助 ──────────────────────────────────────────────────────────────────
def _read_json_object(path: str) -> dict:
    """读取 JSON 文件并确认顶层是对象；出错抛 ValueError（中文信息）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"无法读取 {path} 为 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是 JSON 对象")
    return data


def _make_plan(data: dict) -> TradePlan:
    """从 JSON dict 构造 TradePlan（只取已知字段，数值字段做轻量转类型）。"""
    numeric = (
        "signal_close",
        "entry_ref",
        "stop_price",
        "target_price",
        "position_pct",
        "account_equity",
    )
    integral = ("max_hold_days",)
    known = set(TradePlan.__dataclass_fields__)
    fields = {k: v for k, v in data.items() if k in known}
    try:
        for key in numeric:
            if fields.get(key) is not None:
                fields[key] = float(fields[key])
        for key in integral:
            if fields.get(key) is not None:
                fields[key] = int(fields[key])
        return TradePlan(**fields)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"plan 字段不合法：{exc}") from exc


def _make_gate(data: dict) -> GateContext:
    """从 JSON dict 的可选 ``gate`` 字段构造 GateContext（缺省取默认值）。"""
    gate_data = data.get("gate")
    if gate_data is None:
        return GateContext()
    if not isinstance(gate_data, dict):
        raise ValueError("gate 必须是对象（in_watchlist / daily_trades / ...）")
    known = set(GateContext.__dataclass_fields__)
    return GateContext(**{k: v for k, v in gate_data.items() if k in known})


def _read_ohlcv(path: str) -> pd.DataFrame:
    """读取 OHLCV CSV（date,open,high,low,close,volume）为 DataFrame。

    解析用标准库 csv；返回的 DataFrame 仅用于喂给 execution/pit 的既有 API。
    缺列抛 ValueError（中文信息，列出缺的列名）。
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = [str(c).strip().lower() for c in next(reader, [])]
        rows = [dict(zip(header, row)) for row in reader if any(row)]
    if not rows:
        raise ValueError(f"{path} 为空（无数据行）")
    missing = [c for c in OHLCV_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path} 缺列 {missing}；需要 {list(OHLCV_COLUMNS)}")
    return pd.DataFrame(rows)[list(OHLCV_COLUMNS)]


def _read_universe(directory: str) -> dict[str, pd.DataFrame]:
    """读取 universe 目录里所有 ``*.csv`` 为 ``{symbol: 日线}``。"""
    universe = {
        path.stem: _read_ohlcv(str(path))
        for path in sorted(Path(directory).glob("*.csv"))
    }
    if not universe:
        raise ValueError(f"{directory} 里没有 *.csv 文件")
    return universe


# ── 子命令 ────────────────────────────────────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> int:
    """check：plan.json -> TradePlan -> G1-G8 闸 -> 通过/拦截。"""
    data = _read_json_object(args.plan_json)
    plan, ctx = _make_plan(data), _make_gate(data)
    protocol = plan.validate()
    if protocol:
        print(f"协议违规：{plan.symbol} 未通过计划协议校验")
        for reason in protocol:
            print(f"  - {reason}")
        print("结论：不通过（fail closed，不按协议执行）")
        return 0
    verdict = gate(plan, ctx)
    if verdict.passed:
        print(f"通过：{plan.symbol} 通过 G1-G8 闸")
        return 0
    print(f"拦截：{plan.symbol} 未通过 G1-G8 闸")
    for violation in verdict.violations:
        print(f"  - {violation}")
    return 0


def _fmt_price(price: float | None) -> str:
    """价格格式化为 2 位小数；None 显示 ``-``。"""
    return "-" if price is None else f"{price:.2f}"


def cmd_settle(args: argparse.Namespace) -> int:
    """settle：plan.json + 日线 CSV -> ExecutionSimulator 纸面结算。"""
    data = _read_json_object(args.plan_json)
    plan = _make_plan(data)
    result = ExecutionSimulator(_read_ohlcv(args.ohlcv_csv)).settle(plan)
    print(f"filled={result.filled}")
    print(f"exit_reason={result.exit_reason}")
    print(f"entry={result.entry_date or '-'} @ {_fmt_price(result.entry_price)}")
    print(f"exit={result.exit_date or '-'} @ {_fmt_price(result.exit_price)}")
    print(f"gross_ret={result.gross_ret:+.2%}")
    print(f"net_ret={result.net_ret:+.2%}")
    print(f"cost={result.cost:.2f}")
    print(f"hold_days={result.hold_days}")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    """pack：as-of 数据包；``--csv`` 走本地离线，否则经 loader 抓取。"""
    if args.csv:
        pack = build_data_pack(
            args.symbol, args.as_of, lookback=args.lookback, frame=_read_ohlcv(args.csv)
        )
    else:
        pack = build_data_pack(args.symbol, args.as_of, lookback=args.lookback)
    if not pack["ok"]:
        print(f"失败：{pack['symbol']} 截至 {pack['as_of_date']}：{pack['reason']}")
        return 1
    snap = pack["snapshot"]
    print(
        f"数据包 {pack['symbol']} 截至 {pack['as_of_date']}：{pack['n_bars']} 根 bar，"
        f"regime={pack['regime']}，来源={pack['source']}（{pack['price_basis']}）"
    )
    print(
        f"  收盘 {snap['close']:.2f}  MA20 {snap['ma20']:.2f}  MA60 {snap['ma60']:.2f}  "
        f"MA200 {snap['ma200']:.2f}  RSI {snap['rsi']:.1f}"
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  已落盘 -> {args.out}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """sample：PIT 分层决策点抽样（纯本地，不碰网络）。"""
    universe = _read_universe(args.universe_dir)
    points = sample_points(
        universe,
        args.start,
        args.end,
        args.per_regime,
        seed=args.seed,
        min_spacing=args.min_spacing,
    )
    for point in points:
        print(f"{point.symbol},{point.date},{point.regime}")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    """power：打印功效分析表（delta -> required_n）。"""
    rows = power_table(
        list(args.delta), args.sigma, alpha=args.alpha, power=args.power, side=args.side
    )
    print(
        f"功效分析：sigma={args.sigma:.2f}，alpha={args.alpha:.2f}，"
        f"target power={args.power:.2f}，side={args.side}"
    )
    for row in rows:
        print(f"  delta {row['delta']:.2%} -> required_n = {row['n_required']}")
    return 0


def cmd_verdict(args: argparse.Namespace) -> int:
    """verdict：三档门禁判定，打印 verdict + 中文 reason。"""
    result = gate_verdict(
        (args.ci_lo, args.ci_hi),
        args.point_est,
        cost_floor=args.cost_floor,
        name=args.name,
    )
    label = _VERDICT_LABELS[result["verdict"]]
    print(f"[{label}] {result['reason']}")
    return 0


# ── 入口 ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """构造 argparse 解析器（6 个子命令）。"""
    parser = argparse.ArgumentParser(
        prog="trade_tools", description="A 股纸面交易统一命令行入口"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="子命令")

    p = sub.add_parser("check", help="plan.json -> G1-G8 闸 -> 通过/拦截")
    p.add_argument(
        "plan_json", help="plan.json 路径（TradePlan 全字段 + 可选 gate 字段）"
    )
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("settle", help="plan.json + 日线 CSV -> 纸面结算")
    p.add_argument("plan_json", help="plan.json 路径")
    p.add_argument(
        "ohlcv_csv", help="不复权日线 CSV（date,open,high,low,close,volume）"
    )
    p.set_defaults(func=cmd_settle)

    p = sub.add_parser("pack", help="as-of 数据包（--csv 离线，否则网络抓取）")
    p.add_argument("symbol", help="标的代码，如 600519.SH")
    p.add_argument("as_of", help="决策日 YYYY-MM-DD（含）")
    p.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"回溯交易日数（默认 {DEFAULT_LOOKBACK}）",
    )
    p.add_argument("--csv", help="本地 OHLCV CSV（离线路径，可测试）")
    p.add_argument("--out", help="落盘完整数据包 JSON 的路径")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("sample", help="PIT 分层决策点抽样（纯本地）")
    p.add_argument("--universe-dir", required=True, help="含 <symbol>.csv 日线的目录")
    p.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD", help="决策日范围起点（含）"
    )
    p.add_argument(
        "--end", required=True, metavar="YYYY-MM-DD", help="决策日范围终点（含）"
    )
    p.add_argument("--per-regime", type=int, required=True, help="每层目标点数")
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子（默认 {DEFAULT_SEED}）",
    )
    p.add_argument(
        "--min-spacing",
        type=int,
        default=DEFAULT_MIN_SPACING,
        help=f"同层最小交易日间距（默认 {DEFAULT_MIN_SPACING}）",
    )
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("power", help="功效分析表（Phase 2 前看样本量）")
    p.add_argument(
        "--delta",
        type=float,
        nargs="+",
        default=DEFAULT_DELTAS,
        help=f"可检测效应量列表（默认 {list(DEFAULT_DELTAS)}）",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"单点标准差（默认 {DEFAULT_SIGMA}）",
    )
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="显著性水平")
    p.add_argument("--power", type=float, default=DEFAULT_POWER, help="目标功效")
    p.add_argument(
        "--side", choices=("one", "two"), default=DEFAULT_SIDE, help="单侧/双侧"
    )
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("verdict", help="三档门禁判定（通过/放弃/证据不足）")
    p.add_argument("ci_lo", type=float, help="bootstrap CI 下界")
    p.add_argument("ci_hi", type=float, help="bootstrap CI 上界")
    p.add_argument("point_est", type=float, help="主端点点估计")
    p.add_argument(
        "--cost-floor",
        type=float,
        default=DEFAULT_COST_FLOOR,
        help=f"成本底线（默认 {DEFAULT_COST_FLOOR:.2%}）",
    )
    p.add_argument("--name", default="agent", help="被测臂名称（用于审计文案）")
    p.set_defaults(func=cmd_verdict)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数并分发到子命令；出错打印中文错误并返回非零。"""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"错误：{exc}")
        return 2
    except OSError as exc:
        print(f"错误：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
