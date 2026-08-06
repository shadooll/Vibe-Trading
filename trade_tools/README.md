# trade_tools — 泛化的 A 股纸面交易基础设施

把交易系统的执行现实编码成可复用模块。**Track A（手工 7 环链）和 Track B
（agent 决策验证）共用同一套口径**，避免"手册一套、回测一套"。

## 模块

| 模块 | 职责 | 对应文档 |
|------|------|---------|
| `plan.py` | 结构化交易计划协议：`TradePlan` + 校验。解析失败 = 协议违规 = 按不买（fail closed） | 手册环④、`track_b_plan.md` §三·3 防刷结算 |
| `execution.py` | A股纸面交易模拟器：T+1、涨跌停、停牌、gap、盘中触发、时间止损、tick 滑点、费用 | 手册环⑤⑥、`track_b_plan.md` §三·3 |
| `gate.py` | G1-G8 决策闸：确定性过滤器（含创业板/科创板/北交所禁买） | 手册第二条链 |
| `pit.py` | PIT 无幸存者抽样 + as-of 数据包生成器（regime 判定、宇宙过滤、分层抽样、数据包） | `track_b_plan.md` §三·1/§三·2 |
| `rules.py` | R1-R4 确定性入场理由引擎（手册环③编码；止损/目标按环④口径） | 手册环③④ |
| `validation.py` | Track B 验证流水线：对照臂（买入持有/R1-R4）+ 逐点结算 + 配对差 + 门禁报告 | `track_b_plan.md` §三·4/§三·5 |
| `stats.py` | 功效分析 + 配对 bootstrap/Wilcoxon + 三档门禁 | `track_b_plan.md` §三·4/§三·5 |
| `cli.py` | 统一命令行入口 | 本文件（见下方「命令行」） |

## 用法

```python
from trade_tools.execution import ExecutionSimulator
from trade_tools.plan import TradePlan
from trade_tools.gate import GateContext, gate

plan = TradePlan(
    symbol="000725.SZ", decision="buy",
    signal_date="2026-08-05", signal_close=5.97,
    stop_price=5.79, target_price=6.33, max_hold_days=20,
    position_pct=0.08, account_equity=100_000.0,
)

# 1) 把关：G1-G8
verdict = gate(plan, GateContext(in_watchlist=True))
if verdict.blocked:
    print(verdict.violations)
    raise SystemExit

# 2) 结算：一份计划 + 不复权日线（date/open/high/low/close/volume）→ 已实现盈亏
result = ExecutionSimulator(df).settle(plan)
print(result.exit_reason, result.entry_price, result.exit_price,
      f"{result.net_ret:+.2%}")
```

## 命令行

统一入口 `trade_tools.cli`（`PYTHONPATH=. python -m trade_tools.cli …`），
六个子命令都是纯标准库 argparse 子命令，每行一个 `cmd_xxx(args)` 函数：

| 子命令 | 用途 | 示例 |
|--------|------|------|
| `check` | plan.json -> G1-G8 闸 -> 通过/拦截（含协议违规清单） | `python -m trade_tools.cli check plan.json` |
| `settle` | plan.json + 不复权日线 CSV -> 纸面结算明细 | `python -m trade_tools.cli settle plan.json ohlcv.csv` |
| `pack` | as-of 数据包（`--csv` 走本地离线，去掉即经 loader 抓取网络） | `python -m trade_tools.cli pack 600519.SH 2026-08-05 --csv ohlcv.csv --out pack.json` |
| `sample` | PIT 分层决策点抽样，逐行输出 `symbol,date,regime`（纯本地） | `python -m trade_tools.cli sample --universe-dir universe/ --start 2022-01-01 --end 2023-12-31 --per-regime 10` |
| `power` | 功效分析表（delta -> required_n，Phase 2 前看样本量） | `python -m trade_tools.cli power --delta 0.02 0.03 0.05` |
| `verdict` | 三档门禁判定（通过 / 放弃 / 证据不足 + 中文 reason） | `python -m trade_tools.cli verdict 0.03 0.09 0.05` |

注意：`verdict` 的 CI 下界/上界可能为负，负数位置参数需用 `--` 分隔，如
`python -m trade_tools.cli verdict -- -0.05 -0.01 -0.03`。

`plan.json` 支持 `TradePlan` 全字段（symbol/decision/signal_date/signal_close/
entry_ref/stop_price/target_price/max_hold_days/position_pct/account_equity/
metadata），可选的 `gate` 对象给 `GateContext`（`in_watchlist` 默认 true，
其余默认 0）。

## 关键口径（与手册一致）

- 信号 = 决策日收盘；**入场 = 次日开盘**（T+1）
- 次日高开 > 3% → 追高放弃；次日开盘跌破止损 → 信号失效放弃
- 涨停一字封死 → 无法成交（该点记"无法成交"，不按开盘成交）
- 盘中 high/low 触发止损/目标；**T+1：触发次日开盘成交**
- 止损 gap 穿过 → 按 min(次日开盘, 止损)（最坏情形）
- 跌停封死卖不出 → 顺延；停牌 → 顺延
- 时间止损 = min(计划持仓上限, 手册 4 周=20 交易日, 60 交易日)
- 成交价四舍五入到 0.01 tick 后单向取更差 1 tick（滑点）
- 费用：佣金 max(万2.5, 5元) 双边 + 卖侧印花税万5 + 过户费万0.1
- **价格一律用不复权**（止损/目标/滑点必须在原始 tick 价位上运算）

## 约定

- 数据输出不落在这里——**数据归数据**，输出到 `export_trade_data/` 下。
- 依赖：仅 pandas（纯 Python 实现，不依赖 agent 包，`pit.py` 除外）。
- 测试：`pytest trade_tools/tests`。

## 与 Track A / Track B 的关系

- **Track A 每日纸面单**：手工出计划 → `gate()` 把关 → `ExecutionSimulator` 结算 → 记入 weekly_log。
- **Track B 验证**：agent 备忘录 → `plan.py` 解析（解析失败 = 不买）→ `gate()` 确定性过滤（G1-G8 在回测里"替你把关"）→ `ExecutionSimulator` 结算三臂共用。
- **`pit.py`**：PIT 无幸存者决策点抽样 + as-of 数据包生成。`regime_at`/`regime_stratum`
  与 `daily_check.py` 同一口径（MA200 + MA60）；`UniverseFilter` 做主板/ST(近似)/
  上市≥3年/成交额/停牌过滤，全部只用 <= 决策日的数据；`sample_points` 固定 seed 分层
  抽样（层内间距 ≥ 60 交易日）；`build_data_pack` 出可直接 JSON 落盘的数据包。
- **`stats.py`**：功效分析（`required_n`/`power_table`）、配对 bootstrap CI
  + Wilcoxon 符号秩、主端点描述统计（`effect_summary`）和三档门禁
  （`gate_verdict`：通过 / 放弃 / 证据不足）。**只用 numpy + 标准库
  `statistics`**，不引 scipy。`cli.py` 已落地统一命令行入口（见上方
  「命令行」）。
