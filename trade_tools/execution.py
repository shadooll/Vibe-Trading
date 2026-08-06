"""A-share paper-trade execution simulator.

Replicates the execution reality pinned down in
``export_trade_data/track_b_plan.md`` v2 §三·3 and rings 5-6 of
``export_trade_data/trading_execution_manual.md``:

- T+1: entry at the next open; exits at the open of the day after a trigger.
- Entry filters: one-price limit-up bar -> no fill; next open above
  ``signal_close * 1.03`` (追高) or below the stop (破位) -> abandon.
- Intraday triggers on high/low; stop/target exits at the next open
  (gap-through stop fills at the worse of stop / open).
- Suspension carry-forward; a locked limit-down sell defers to the next
  tradeable day.
- Price rounded to the 0.01 tick, then one tick worse (slippage).
- CN costs: commission ``max(万2.5, 5元)`` both sides, sell-side stamp tax
  ``万5``, transfer fee ``万0.1``.

Track A paper trades and Track B agent settlements share this one simulator so
both tracks settle under the same口径.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trade_tools.plan import TradePlan

# ── Market model constants ───────────────────────────────────────────────────
TICK = 0.01  # A 股最小变动单位（元）
GAP_UP_LIMIT = 0.03  # 次日高开 >3% -> 追高违规，放弃
DEFAULT_TIME_STOP_DAYS = 20  # 手册环⑥：目标持仓上限 ~4 周（交易日）
HARD_CAP_DAYS = 60  # v2：结算窗口上限 60 交易日
COMMISSION_RATE = 2.5e-4  # 佣金 万2.5
COMMISSION_MIN = 5.0  # 最低佣金 5 元
STAMP_TAX = 5e-4  # 卖侧印花税 万5
TRANSFER_FEE = 1e-5  # 过户费 万0.1
LIMIT_PCT = 0.10  # 主板涨跌幅 ±10%
_ONE_PRICE_TOL = 1e-9
_LOT = 100  # 一手 100 股


@dataclass(frozen=True)
class Bar:
    """One normalized daily bar (unadjusted price)."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    pre_close: float


@dataclass
class SettlementResult:
    """Outcome of one :class:`TradePlan` through the simulator."""

    filled: bool
    exit_reason: str
    entry_date: str | None = None
    entry_price: float | None = None
    exit_date: str | None = None
    exit_price: float | None = None
    gross_ret: float = 0.0
    net_ret: float = 0.0
    cost: float = 0.0
    hold_days: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def realized(self) -> bool:
        """True when a round trip completed (filled and exited)."""
        return self.filled and self.exit_price is not None


def _round_tick(price: float) -> float:
    """Round to the nearest 0.01 tick."""
    return round(price / TICK) * TICK


def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV columns (case-insensitive) and sort by date ascending."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna().sort_index()
    cols = {str(c).lower(): c for c in df.columns}
    need = ("open", "high", "low", "close", "volume")
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"缺列: {missing}")
    out = df.rename(columns={cols[c]: c for c in need})[list(need)].copy()
    return out.apply(pd.to_numeric, errors="coerce")


class ExecutionSimulator:
    """Settle a :class:`TradePlan` against daily unadjusted OHLCV bars.

    Args:
        df: Daily OHLCV (open/high/low/close/volume), either a ``date`` column
            or a date index, sorted ascending. Suspended days are simply absent
            from the frame — the simulator carries across gaps.
    """

    def __init__(self, df: pd.DataFrame):
        norm = _norm_df(df)
        closes = norm["close"].tolist()
        self._bars: list[Bar] = []
        for i, (date, row) in enumerate(norm.iterrows()):
            pre = closes[i - 1] if i > 0 else float(row["open"])
            self._bars.append(
                Bar(
                    date=str(pd.Timestamp(date).date()),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    pre_close=float(pre),
                )
            )

    # ── Lookup helpers ──────────────────────────────────────────────────────
    def _first_after(self, date: str) -> int | None:
        """Index of the first bar strictly after ``date`` (signal fixed at close)."""
        for i, bar in enumerate(self._bars):
            if bar.date > date:
                return i
        return None

    @staticmethod
    def _is_one_price(bar: Bar) -> bool:
        return (
            abs(bar.open - bar.high) <= _ONE_PRICE_TOL
            and abs(bar.open - bar.low) <= _ONE_PRICE_TOL
            and abs(bar.open - bar.close) <= _ONE_PRICE_TOL
        )

    def _at_limit(self, idx: int, direction: float) -> bool:
        """True when the bar is a locked one-price bar at the ±limit price."""
        bar = self._bars[idx]
        if not self._is_one_price(bar):
            return False
        limit = _round_tick(bar.pre_close * (1 + direction * LIMIT_PCT))
        return abs(bar.close - limit) <= TICK / 2

    def _next_sellable(self, start: int) -> int | None:
        """First bar at/after ``start`` that is not a locked limit-down bar."""
        for i in range(start, len(self._bars)):
            if not self._at_limit(i, -1.0):
                return i
        return None

    # ── Entry ───────────────────────────────────────────────────────────────
    def _try_entry(self, plan: TradePlan, idx: int) -> tuple[float, str | None]:
        """Return (entry_price, abandon_reason); abandon_reason None = filled."""
        bar = self._bars[idx]
        raw_open = bar.open
        if self._at_limit(idx, 1.0):
            return 0.0, "limit_up_no_fill"
        if raw_open > plan.signal_close * (1 + GAP_UP_LIMIT):
            return 0.0, "gap_up_abandon"
        if plan.stop_price is not None and raw_open < plan.stop_price:
            return 0.0, "gap_down_stop_abandon"
        # 买入滑点：四舍五入到 tick 后取更差 1 tick（买 = 更高）。
        return _round_tick(raw_open) + TICK, None

    # ── Exit ────────────────────────────────────────────────────────────────
    def _walk(
        self, plan: TradePlan, entry_idx: int, max_hold: int
    ) -> tuple[str | None, int, int]:
        """Walk bars from entry; return (reason, trigger_idx, last_held_idx)."""
        for i in range(entry_idx, len(self._bars)):
            bar = self._bars[i]
            held = i - entry_idx + 1
            if plan.stop_price is not None and bar.low <= plan.stop_price:
                return "stop", i, i
            if plan.target_price is not None and bar.high >= plan.target_price:
                return "target", i, i
            if held >= max_hold:
                return "time_stop", i, i
        return None, -1, len(self._bars) - 1

    @staticmethod
    def _exit_base(reason: str, sell_bar: Bar, stop_price: float) -> float:
        if reason == "stop":
            # gap 穿过止损：取止损价与卖出日开盘的更低者（真实最坏情形）。
            return min(sell_bar.open, stop_price)
        return sell_bar.open

    def _costs(
        self, entry_price: float, exit_price: float, shares: int
    ) -> tuple[float, float]:
        """Return (net_return, total_cost) after CN fees on a round trip."""
        buy_amount = entry_price * shares
        sell_amount = exit_price * shares
        comm_buy = max(buy_amount * COMMISSION_RATE, COMMISSION_MIN)
        comm_sell = max(sell_amount * COMMISSION_RATE, COMMISSION_MIN)
        stamp = sell_amount * STAMP_TAX
        transfer = (buy_amount + sell_amount) * TRANSFER_FEE
        cost = comm_buy + comm_sell + stamp + transfer
        net_ret = (sell_amount - cost - buy_amount) / buy_amount
        return net_ret, cost

    # ── Public API ──────────────────────────────────────────────────────────
    def settle(self, plan: TradePlan) -> SettlementResult:
        """Settle one plan to a realized P&L (or an explicit no-fill reason)."""
        violations = plan.validate()
        if violations:
            return SettlementResult(
                filled=False, exit_reason="protocol_violation", details=violations
            )
        if plan.decision != "buy":
            return SettlementResult(
                filled=False, exit_reason=f"decision={plan.decision}"
            )

        entry_idx = self._first_after(plan.signal_date)
        if entry_idx is None:
            return SettlementResult(filled=False, exit_reason="no_bar_after_signal")
        entry_bar = self._bars[entry_idx]

        entry_price, abandon = self._try_entry(plan, entry_idx)
        if abandon is not None:
            return SettlementResult(
                filled=False,
                exit_reason=abandon,
                entry_date=entry_bar.date,
                details=[
                    f"次日开盘 {entry_bar.open:.2f}，信号收盘 {plan.signal_close:.2f}"
                ],
            )

        max_hold = min(
            plan.max_hold_days or DEFAULT_TIME_STOP_DAYS,
            DEFAULT_TIME_STOP_DAYS,
            HARD_CAP_DAYS,
        )
        reason, trigger_idx, last_held = self._walk(plan, entry_idx, max_hold)
        if reason is None:  # 数据耗尽，未到任何触发点
            return SettlementResult(
                filled=True,
                exit_reason="no_exit_data",
                entry_date=entry_bar.date,
                entry_price=entry_price,
            )

        # T+1：触发日次日开盘卖出；跌停封死则顺延。
        sell_idx = self._next_sellable(trigger_idx + 1)
        if sell_idx is None:
            return SettlementResult(
                filled=True,
                exit_reason=reason,
                entry_date=entry_bar.date,
                entry_price=entry_price,
                details=["卖出日无可用 bar，无法结算"],
            )
        sell_bar = self._bars[sell_idx]
        base = self._exit_base(reason, sell_bar, plan.stop_price or 0.0)
        exit_price = _round_tick(base) - TICK  # 卖出滑点：更差 1 tick（卖 = 更低）

        shares = int(plan.position_value / entry_price / _LOT) * _LOT
        if shares <= 0:
            shares = _LOT
        net_ret, cost = self._costs(entry_price, exit_price, shares)
        gross_ret = exit_price / entry_price - 1

        return SettlementResult(
            filled=True,
            exit_reason=reason,
            entry_date=entry_bar.date,
            entry_price=entry_price,
            exit_date=sell_bar.date,
            exit_price=exit_price,
            gross_ret=gross_ret,
            net_ret=net_ret,
            cost=cost,
            hold_days=trigger_idx - entry_idx + 1,
            details=[f"触发日 {self._bars[trigger_idx].date}，卖出日 {sell_bar.date}"],
        )
