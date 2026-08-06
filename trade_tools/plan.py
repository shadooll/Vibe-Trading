"""Structured trade-plan protocol.

A :class:`TradePlan` is the machine-readable handoff from a decision-maker
(agent or human) to :class:`~trade_tools.execution.ExecutionSimulator`.
Schema/domain validation failures are surfaced as :class:`ProtocolViolation`:
a broken plan is treated exactly like a no-trade (fail closed) — never
re-asked, never force-filled.

Numbers mirror the A-share risk rules in
``export_trade_data/trading_execution_manual.md`` (rings 4-5, G1/G7) and
``export_trade_data/track_b_plan.md`` v2 §三·3 (结算防刷：止损/目标只采纳模板
区间内的值).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DECISIONS = ("buy", "no_buy", "hold")

# G1 / G7 thresholds from the manual.
MIN_RR = 2.0  # R:R 必须 ≥ 2
MAX_RISK_PCT = 0.01  # 单笔风险 ≤ 1% 账户
MIN_POSITION_PCT = 0.01
MAX_POSITION_PCT = 1.0


@dataclass(frozen=True)
class ProtocolViolation:
    """A plan that failed schema/domain validation — fails closed to no_buy."""

    symbol: str
    reasons: tuple[str, ...]


@dataclass
class TradePlan:
    """One machine-readable buy/no_buy/hold decision.

    Attributes:
        symbol: Trading symbol, e.g. ``000725.SZ`` / ``510300.SH``.
        decision: ``buy`` | ``no_buy`` | ``hold``.
        signal_date: Decision day (YYYY-MM-DD); the signal is fixed at close.
        signal_close: Decision-day close price.
        entry_ref: Reference entry price — defaults to ``signal_close`` for a
            next-open order; a conditional plan may set a trigger price.
        stop_price: Stop-loss price (meaningful only when ``decision == buy``).
        target_price: Take-profit price (only when ``decision == buy``).
        max_hold_days: Holding cap in trading days (manual default ~4 weeks).
        position_pct: Fraction of account to deploy (0-1).
        account_equity: Account equity, used for sizing and the G1 check.
        hold_only: Benchmark mode (buy-and-hold): no stop/target, time-stop
            only, entry ignores the discipline filters. Used by the Track B
            buy-and-hold baseline so it settles through the same simulator as
            risk-managed trades.
        metadata: Free-form extras (e.g. ``position_value`` for sizing when
            no equity is given; ``source`` for the decision-maker id).
    """

    symbol: str
    decision: str
    signal_date: str
    signal_close: float
    entry_ref: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    max_hold_days: int | None = None
    position_pct: float | None = None
    account_equity: float = 0.0
    hold_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_price(self) -> float:
        """Reference entry price: the conditional trigger, else signal close."""
        return self.entry_ref if self.entry_ref is not None else self.signal_close

    @property
    def position_value(self) -> float:
        """Deployable capital in CNY. Falls back to 10,000 when unsized."""
        if self.account_equity > 0 and self.position_pct is not None:
            return self.account_equity * self.position_pct
        if self.metadata.get("position_value"):
            return float(self.metadata["position_value"])
        return 10_000.0

    @property
    def risk_pct(self) -> float | None:
        """Stop distance as a fraction of entry price (risk per unit position)."""
        if self.stop_price is None or self.entry_price <= 0:
            return None
        if self.stop_price >= self.entry_price:
            return None
        return (self.entry_price - self.stop_price) / self.entry_price

    @property
    def rr(self) -> float | None:
        """Reward/risk ratio; None when the structure is broken."""
        if self.stop_price is None or self.target_price is None:
            return None
        denom = self.entry_price - self.stop_price
        if denom <= 0:
            return None
        return (self.target_price - self.entry_price) / denom

    def validate(self) -> list[str]:
        """Return protocol violations; empty list means the plan is well-formed."""
        reasons: list[str] = []
        if not self.symbol:
            reasons.append("symbol 为空")
        if self.decision not in DECISIONS:
            reasons.append(f"decision 必须是 {DECISIONS} 之一，得到 {self.decision!r}")
        if self.signal_close <= 0:
            reasons.append("signal_close 必须 > 0")

        if self.decision != "buy":
            return reasons

        if not self.hold_only and (
            self.stop_price is None or self.target_price is None
        ):
            reasons.append("buy（非 hold_only）必须给 stop_price 和 target_price")
        if self.max_hold_days is None:
            reasons.append("buy 必须给 max_hold_days")
        if self.position_pct is None:
            reasons.append("buy 必须给 position_pct")
        elif not (MIN_POSITION_PCT <= self.position_pct <= MAX_POSITION_PCT):
            reasons.append(
                f"position_pct 必须在 [{MIN_POSITION_PCT}, {MAX_POSITION_PCT}] 内，"
                f"得到 {self.position_pct}"
            )

        if not self.hold_only:
            if self.stop_price is not None and self.entry_price is not None:
                if self.stop_price >= self.entry_price:
                    reasons.append("stop_price 必须 < 入场价")
            if self.target_price is not None and self.entry_price is not None:
                if self.target_price <= self.entry_price:
                    reasons.append("target_price 必须 > 入场价")

            rr = self.rr
            if rr is not None and rr < MIN_RR:
                reasons.append(f"R:R={rr:.2f} < {MIN_RR} (G7)")

            risk = self.risk_pct
            if risk is not None and self.account_equity > 0:
                if risk * self.position_value > MAX_RISK_PCT * self.account_equity:
                    reasons.append(
                        f"单笔风险 {risk * self.position_pct:.2%} 账户 > {MAX_RISK_PCT:.0%} (G1)"
                    )
        return reasons
