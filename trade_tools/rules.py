"""R1-R4 确定性入场理由引擎。

把 ``trading_execution_manual.md`` 环③的 4 条入场理由编码成纯函数：给定某标的
截至决策日的日线 + 大盘 regime，返回是否触发 + 结构化交易参数（止损/目标按
环④口径推导：止损 = 前低 / −3% 中更靠下者，目标 = 入场 + 2 × 止损距离）。

供 Track B 的 R1-R4 对照臂使用——保证"回测里的 R1-R4"与"手册里的 R1-R4"
是同一套规则（评审 FINDING 4：agent 臂与 R1-R4 臂共享信号骨架，这里就是那
个骨架的确定性实现）。

输入 ``bars`` 必须是截至决策日的切片（as-of，无前视）；入场参考价 = 信号收盘价。
"""

from __future__ import annotations

import pandas as pd

from trade_tools.pit import REGIME_BEAR, REGIME_RANGE, REGIME_TREND

# ── 环③阈值（手册原文）────────────────────────────────────────────────────────
VOL_SURGE = 1.5  # R2：今日量 > 前5日均量 × 1.5
VOL_SHRINK = 0.7  # R4：缩量 < 前5日均量 × 0.7
RSI_OVERSOLD = 30.0  # R3：超卖线
PULLBACK_DAYS = 5  # R4：回调 ≥ 5 日
RECENT_LOW_BARS = 10  # 止损：前 10 日最低（不含今日）
STOP_FLOOR = 0.97  # 止损最深 −3%（手册"前低 / −3%"取更靠下的）
STOP_MIN_RET = 0.99  # 止损至少距入场 1%（防 R:R 失真）
R1_PULLBACK_WINDOW = 3  # R1：最近 N 日不破 20 日线
R1_PULLBACK_SHRINK = 3  # R1：回踩期量能对比窗口


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI 序列（与 daily_check 同口径：ewm alpha=1/period）。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period).mean()
    return (100 - 100 / (1 + ag / al)).where(al != 0, 100.0)


def _stop_target(entry: float, low: pd.Series) -> tuple[float, float]:
    """环④口径：止损 = 前 10 日最低与 −3% 中更靠下者；目标 = 入场 + 2 × 止损距离。"""
    if len(low) > RECENT_LOW_BARS:
        recent_low = float(low.iloc[-(RECENT_LOW_BARS + 1) : -1].min())
    else:
        recent_low = float(low.min())
    stop = min(recent_low, entry * STOP_FLOOR)
    if stop >= entry:
        stop = entry * STOP_MIN_RET
    target = entry + 2 * (entry - stop)
    return stop, target


def r1_signal(bars: pd.DataFrame) -> dict | None:
    """R1 回踩企稳（趋势）：①趋势中 ②回踩20日线不破 ③回踩缩量 ④收阳站回20日线。"""
    c = bars["close"].astype(float)
    o = bars["open"].astype(float)
    lo = bars["low"].astype(float)
    v = bars["volume"].astype(float)
    ma20 = c.rolling(20).mean()
    ma200 = c.rolling(200).mean()
    if len(bars) < 200 or pd.isna(ma20.iloc[-1]) or pd.isna(ma200.iloc[-1]):
        return None
    close, open_ = c.iloc[-1], o.iloc[-1]
    # ① 趋势中
    if close <= ma200.iloc[-1]:
        return None
    # 回踩前提：从高点回撤（不是创新高突破）
    if close >= float(c.iloc[-21:-1].max()):
        return None
    # ② 最近 R1_PULLBACK_WINDOW 日收盘未破 20 日线
    if any(c.iloc[-k] < ma20.iloc[-k] for k in range(1, R1_PULLBACK_WINDOW + 1)):
        return None
    # ③ 回踩期缩量（近 3 日均量 < 前一段均量）
    recent_vol = float(v.iloc[-R1_PULLBACK_WINDOW:].mean())
    prior_vol = float(
        v.iloc[-R1_PULLBACK_WINDOW - R1_PULLBACK_SHRINK : -R1_PULLBACK_WINDOW].mean()
    )
    if prior_vol > 0 and recent_vol >= prior_vol:
        return None
    # ④ 收阳站回 20 日线
    if close <= open_ or close < ma20.iloc[-1]:
        return None
    stop, target = _stop_target(close, lo)
    return {"reason": "R1", "entry_ref": close, "stop": stop, "target": target}


def r2_signal(bars: pd.DataFrame) -> dict | None:
    """R2 放量突破（趋势）：①收盘创20日新高 ②今日量 > 前5日均量×1.5。"""
    c = bars["close"].astype(float)
    v = bars["volume"].astype(float)
    lo = bars["low"].astype(float)
    if len(bars) < 21:
        return None
    close, vol = c.iloc[-1], v.iloc[-1]
    if close <= float(c.iloc[-21:-1].max()):
        return None  # 未创 20 日新高
    vol_ma5 = float(v.iloc[-6:-1].mean())
    if vol_ma5 <= 0 or vol < vol_ma5 * VOL_SURGE:
        return None  # 未放量 1.5 倍
    stop, target = _stop_target(close, lo)
    return {"reason": "R2", "entry_ref": close, "stop": stop, "target": target}


def r3_signal(bars: pd.DataFrame) -> dict | None:
    """R3 超跌反弹（震荡）：RSI14 < 30 后重新站上 30。"""
    close = bars["close"].astype(float)
    rsi = _rsi_series(close)
    if len(rsi) < 6 or pd.isna(rsi.iloc[-1]):
        return None
    if float(rsi.iloc[-5:].min()) >= RSI_OVERSOLD:
        return None  # 近 5 日从未超卖
    if rsi.iloc[-1] <= RSI_OVERSOLD:
        return None  # 尚未站回 30
    stop, target = _stop_target(float(close.iloc[-1]), bars["low"].astype(float))
    return {
        "reason": "R3",
        "entry_ref": float(close.iloc[-1]),
        "stop": stop,
        "target": target,
    }


def r4_signal(bars: pd.DataFrame) -> dict | None:
    """R4 缩量企稳（震荡）：回调 ≥5 日 + 缩量（<前5日均量×0.7）+ 收阳。"""
    c = bars["close"].astype(float)
    o = bars["open"].astype(float)
    v = bars["volume"].astype(float)
    lo = bars["low"].astype(float)
    if len(bars) < 6:
        return None
    close, open_, vol = c.iloc[-1], o.iloc[-1], v.iloc[-1]
    if close >= c.iloc[-PULLBACK_DAYS - 1]:
        return None  # 未净回调 ≥ 5 日
    vol_ma5 = float(v.iloc[-6:-1].mean())
    if vol_ma5 <= 0 or vol >= vol_ma5 * VOL_SHRINK:
        return None  # 未缩量
    if close <= open_:
        return None  # 未收阳
    stop, target = _stop_target(close, lo)
    return {"reason": "R4", "entry_ref": close, "stop": stop, "target": target}


def r1r4_signal(bars: pd.DataFrame, regime: str) -> dict | None:
    """按大盘 regime 选理由：趋势 → R1/R2；震荡/熊市 → R3/R4；未知 → None。"""
    if regime == REGIME_TREND:
        return r1_signal(bars) or r2_signal(bars)
    if regime in (REGIME_RANGE, REGIME_BEAR):
        return r3_signal(bars) or r4_signal(bars)
    return None
