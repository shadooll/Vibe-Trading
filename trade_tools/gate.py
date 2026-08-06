"""G1-G8 decision gate.

Encodes ``trading_execution_manual.md`` 第二条链 (铁律) as a deterministic
filter. Track A's human gatekeeper and Track B's backtest use this same gate,
so "把关" is one shared rule set instead of two.

The gate is intentionally state-free: callers feed today's account/regime
state in :class:`GateContext`; the gate only *judges* a single plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trade_tools.plan import MIN_RR, MAX_RISK_PCT, TradePlan

# 用户账户禁买板块（创业板 300 / 科创板 688 / 北交所 8xxxxx）。
BANNED_MARKET_PREFIXES = ("300", "688", "8")


@dataclass
class GateContext:
    """Account/regime state the gate needs to judge a plan."""

    in_watchlist: bool = True  # G8：观察池内才有资格
    daily_trades: int = 0  # G2：当日已成交笔数
    week_trades: int = 0  # G3：本周已成交笔数
    daily_loss_pct: float = 0.0  # G4：当日已实现亏损占账户比例（0-1）
    consecutive_losses: int = 0  # G5：连续亏损笔数
    metadata: dict = field(default_factory=dict)


@dataclass
class GateVerdict:
    """Judgment of one plan under G1-G8."""

    passed: bool
    violations: list[str]

    @property
    def blocked(self) -> bool:
        return not self.passed


def _market_code(symbol: str) -> str:
    """Strip the exchange suffix, e.g. ``000725.SZ`` -> ``000725``."""
    return symbol.split(".")[0]


def gate(plan: TradePlan, ctx: GateContext) -> GateVerdict:
    """Return whether ``plan`` may be executed under G1-G8.

    Non-buy decisions always pass (G6: 今天 0 交易 = 成功).
    """
    if plan.decision != "buy":
        return GateVerdict(passed=True, violations=[])

    violations: list[str] = []

    protocol = plan.validate()
    violations.extend(f"协议: {reason}" for reason in protocol)

    # G8：观察池外 / 禁买板块。
    if not ctx.in_watchlist:
        violations.append("G8: 观察池外 -> 不看、不碰")
    prefix = _market_code(plan.symbol)[:3]
    if plan.symbol and _market_code(plan.symbol).startswith(BANNED_MARKET_PREFIXES):
        violations.append(f"G8: {prefix} 前缀 = 账户禁买板块")

    # G7：R:R >= 2（结构性；plan.validate 已算，这里显式列一条供审计）。
    rr = plan.rr
    if rr is not None and rr < MIN_RR:
        violations.append(f"G7: R:R={rr:.2f} < {MIN_RR}")

    # G1：单笔风险 <= 1% 账户。
    if plan.account_equity > 0 and plan.risk_pct is not None:
        risk_amount = plan.risk_pct * plan.position_value
        if risk_amount > MAX_RISK_PCT * plan.account_equity:
            violations.append("G1: 单笔风险 > 1% 账户")

    # G2/G3：频率。
    if ctx.daily_trades >= 1:
        violations.append("G2: 每日最多 1 笔")
    if ctx.week_trades >= 3:
        violations.append("G3: 每周最多 3 笔")

    # G4/G5：熔断与冷却。
    if ctx.daily_loss_pct >= 0.02:
        violations.append("G4: 单日亏损 >= 2% 熔断")
    if ctx.consecutive_losses >= 3:
        violations.append("G5: 连亏 3 笔 -> 冷却 1 天")

    return GateVerdict(passed=not violations, violations=violations)
