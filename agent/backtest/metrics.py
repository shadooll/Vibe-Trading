"""Shared backtest metrics, extracted from daily_portfolio.py for reuse.

Provides annualisation helpers, trade statistics, and full metric calculation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.models import TradeRecord

# ─── Annualisation factor mapping ───

# mootdx (A-share) and futu (HK + A-share) are equity sources, so they mirror
# the tushare/akshare column: 252 trading days and a 240-minute session. HK
# sessions are marginally longer (~330 min) — an approximation in line with the
# rest of this annualisation table; the key fix is that intraday mootdx/futu no
# longer fall back to the bars_per_day=1 default, which mis-annualised vol/Sharpe.
_TRADING_DAYS = {
    # existing
    "tushare": 252, "yfinance": 252, "okx": 365, "akshare": 252, "ccxt": 365,
    "mootdx": 252, "futu": 252, "mt5": 260,
    # crypto
    "binance": 365,
    # A-share equity
    "baostock": 252, "tencent": 252, "eastmoney": 252, "sina": 252,
    # US / international equity
    "yahoo": 252, "finnhub": 252, "alphavantage": 252, "tiingo": 252,
    "fmp": 252, "stooq": 252, "longbridge": 252,
    # resampling sources (local files / paid data — interval depends on source
    # data granularity; model as US equity session as a conservative default)
    "local": 252, "qveris": 252,
    # Indian equity
    "india_broker": 252,
    # Korean equity (KRX)
    "pykrx": 252,
}
# mt5 is a forex/CFD feed: 24x5 sessions → 260 trading days, 24h intraday bars.
# US equity (yfinance-style): 6.5h sessions → 390 1m bars/day.
# A-share equity (tushare-style): 4.0h sessions → 240 1m bars/day.
# Crypto (okx/ccxt-style): 24h sessions → 1440 1m bars/day.
# Indian equity: 6.25h sessions → 375 1m bars/day.
# Korean equity (pykrx): 6.5h sessions → 390 1m bars/day. The loader itself
# serves daily bars only, so the intraday rows exist to keep the table complete
# (and correct if KRX intraday ever arrives under this key), not because pykrx
# can return them.
_BARS_PER_DAY = {
    #  --- US/international equity (6.5h session) ---
    "1m":  {"yfinance": 390, "yahoo": 390, "finnhub": 390, "alphavantage": 390,
            "tiingo": 390, "fmp": 390, "stooq": 390, "longbridge": 390,
            "local": 390, "qveris": 390,
            # A-share equity (4.0h session)
            "tushare": 240, "akshare": 240, "baostock": 240, "tencent": 240,
            "eastmoney": 240, "sina": 240, "mootdx": 240, "futu": 240,
            # crypto (24h)
            "okx": 1440, "ccxt": 1440, "binance": 1440,
            # forex/CFD (24h intraday)
            "mt5": 1440,
            # Indian equity (6.25h session)
            "india_broker": 375,
            # Korean equity (6.5h session, 09:00-15:30 KST)
            "pykrx": 390,
            },
    "5m":  {"yfinance": 78,  "yahoo": 78,  "finnhub": 78,  "alphavantage": 78,
            "tiingo": 78,  "fmp": 78,  "stooq": 78,  "longbridge": 78,
            "local": 78, "qveris": 78,
            "tushare": 48,  "akshare": 48,  "baostock": 48,  "tencent": 48,
            "eastmoney": 48,  "sina": 48,  "mootdx": 48,  "futu": 48,
            "okx": 288,  "ccxt": 288,  "binance": 288,
            "mt5": 288,
            "india_broker": 75,
            "pykrx": 78,
            },
    "15m": {"yfinance": 26,  "yahoo": 26,  "finnhub": 26,  "alphavantage": 26,
            "tiingo": 26,  "fmp": 26,  "stooq": 26,  "longbridge": 26,
            "local": 26, "qveris": 26,
            "tushare": 16,  "akshare": 16,  "baostock": 16,  "tencent": 16,
            "eastmoney": 16,  "sina": 16,  "mootdx": 16,  "futu": 16,
            "okx": 96,   "ccxt": 96,   "binance": 96,
            "mt5": 96,
            "india_broker": 25,
            "pykrx": 26,
            },
    "30m": {"yfinance": 13,  "yahoo": 13,  "finnhub": 13,  "alphavantage": 13,
            "tiingo": 13,  "fmp": 13,  "stooq": 13,  "longbridge": 13,
            "local": 13, "qveris": 13,
            "tushare": 8,   "akshare": 8,   "baostock": 8,   "tencent": 8,
            "eastmoney": 8,   "sina": 8,   "mootdx": 8,   "futu": 8,
            "okx": 48,   "ccxt": 48,   "binance": 48,
            "mt5": 48,
            "india_broker": 13,
            "pykrx": 13,
            },
    "1H":  {"yfinance": 7,   "yahoo": 7,   "finnhub": 7,   "alphavantage": 7,
            "tiingo": 7,   "fmp": 7,   "stooq": 7,   "longbridge": 7,
            "local": 7, "qveris": 7,
            "tushare": 4,   "akshare": 4,   "baostock": 4,   "tencent": 4,
            "eastmoney": 4,   "sina": 4,   "mootdx": 4,   "futu": 4,
            "okx": 24,   "ccxt": 24,   "binance": 24,
            "mt5": 24,
            "india_broker": 7,
            "pykrx": 7,
            },
    "4H":  {"yfinance": 2,   "yahoo": 2,   "finnhub": 2,   "alphavantage": 2,
            "tiingo": 2,   "fmp": 2,   "stooq": 2,   "longbridge": 2,
            "local": 2, "qveris": 2,
            "tushare": 1,   "akshare": 1,   "baostock": 1,   "tencent": 1,
            "eastmoney": 1,   "sina": 1,   "mootdx": 1,   "futu": 1,
            "okx": 6,    "ccxt": 6,    "binance": 6,
            "mt5": 6,
            "india_broker": 2,
            "pykrx": 2,
            },
    "1D":  {"yfinance": 1,   "yahoo": 1,   "finnhub": 1,   "alphavantage": 1,
            "tiingo": 1,   "fmp": 1,   "stooq": 1,   "longbridge": 1,
            "local": 1, "qveris": 1,
            "tushare": 1,   "akshare": 1,   "baostock": 1,   "tencent": 1,
            "eastmoney": 1,   "sina": 1,   "mootdx": 1,   "futu": 1,
            "okx": 1,    "ccxt": 1,    "binance": 1,
            "mt5": 1,
            "india_broker": 1,
            "pykrx": 1,
            },
}

# Runner/loaders also emit these aliases; map them onto the table keys above.
_SOURCE_ALIASES = {"yahoo": "yfinance", "binance": "ccxt"}


def _normalize_interval(interval: str) -> str:
    """Map project interval tokens onto ``_BARS_PER_DAY`` keys.

    Minute bars stay lowercase (``1m``); hour/day use the uppercase keys the
    table already stores (``1H`` / ``4H`` / ``1D``). Loaders accept both cases
    after the interval-map fixes; annualisation must too.
    """
    token = str(interval or "1D").strip()
    lower = token.lower()
    if lower in ("1m", "5m", "15m", "30m"):
        return lower
    if lower in ("1h", "4h", "1d"):
        return lower.upper()
    return token


def calc_bars_per_year(interval: str = "1D", source: str = "tushare") -> int:
    """Number of bars per year for annualisation.

    Args:
        interval: Bar size (1m / 5m / 15m / 30m / 1H / 4H / 1D), case-insensitive
            like loaders accept (``1h`` → ``1H``, ``4h`` → ``4H``, ``1d`` → ``1D``).
        source: Data source (any VALID_SOURCES entry). Defaults to 252 days, 1 bar/day
            when source is missing from the table.

    Returns:
        Bars per year.
    """
    interval_key = _normalize_interval(interval)
    source_key = str(source or "").strip().lower()
    source_key = _SOURCE_ALIASES.get(source_key, source_key)
    trading_days = _TRADING_DAYS.get(source_key, 252)
    bars_per_day = _BARS_PER_DAY.get(interval_key, {}).get(source_key, 1)
    return trading_days * bars_per_day


# ─── Sign-safe returns ───

_log = logging.getLogger(__name__)


def bar_returns(close: Any, *, label: str = "") -> Any:
    """Per-bar simple returns, defined only where the prior price is positive.

    ``close.pct_change()`` silently assumes ``price[t-1] > 0``. That holds for
    equities but not for instruments that can print zero or negative prices,
    such as European day-ahead power. As the divisor approaches zero an
    ordinary absolute move reads as an enormous *percentage* move, so a
    compounded aggregate explodes; at exactly zero the next bar is ``inf``,
    which ``fillna`` does not neutralise (it fills ``NaN``) and which collapses
    ``(1 + r).prod()`` to ``nan`` (#872).

    A return is therefore defined only when the previous price is finite and
    strictly positive. Otherwise it is undefined and reported as ``0.0``,
    never ``inf`` or ``nan``. For an all-positive series this is identical to
    ``pct_change().fillna(0.0)``, so ordinary equity/crypto runs are unchanged.

    The prior price is carried forward across gaps. ``pct_change`` did this
    implicitly via ``fill_method='pad'`` — its default on pandas < 3, removed
    in 3.0 — so *not* forwarding it would silently change every gapped series
    (a halt longer than ``_align``'s ``ffill_limit``, a thinly-traded symbol)
    and would make the bit-identity claim above false on the pinned pandas.
    Doing it here explicitly keeps that promise and, unlike the old default,
    gives the same answer on every supported pandas version. Note the effect
    is to attribute the whole across-gap move to the resumed bar; that is what
    a position held through the halt actually earned. Whether that is the right
    convention for *statistics* (it is not, for correlation — see
    ``correlation.py``, which drops the observation instead) is a separate
    question from whether this function may change it as a side effect.

    Args:
        close: Raw price ``Series`` or per-symbol ``DataFrame``.
        label: Optional name used when warning about undefined bars.

    Returns:
        Returns aligned to ``close``, with the first bar ``0.0``.
    """
    # ffill *then* shift: the divisor is the last known price, not NaN.
    prev = close.ffill().shift(1)
    # ``inf > 0`` is True, so finiteness has to be asserted, not assumed.
    usable_prev = np.isfinite(prev) & (prev > 0)
    positive_prev = prev.where(usable_prev)

    undefined = int((prev.notna() & ~usable_prev).to_numpy().sum())
    if undefined:
        # Silence is the actual hazard here: a wrong return is a wrong reported
        # number, not a crash, so the run has to say it defaulted.
        _log.warning(
            "%s: %d bar(s) follow a non-positive or non-finite prior price; "
            "their return is undefined and reported as 0.0 (issue #872)",
            label or "returns",
            undefined,
        )

    # ``close / prev - 1`` is exactly the expression ``pct_change`` uses, kept
    # verbatim so results for positive series are bit-identical, not merely
    # equal to within floating-point error.
    ret = close / positive_prev - 1
    return ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def buy_and_hold_return(close: Any) -> Optional[float]:
    """Total buy-and-hold return as a price relative, not a compounded product.

    ``(1 + close.pct_change()).prod() - 1`` telescopes to ``P_end / P_start - 1``
    only while every price is positive; once a near-zero price enters the
    series the product diverges from what a held position actually earned. On
    the reproduction in #872 it reported ``+39,560%`` where the price relative
    gives ``-42.7%``. This computes the price relative directly, so the two
    agree exactly for ordinary series and it stays honest for the rest.

    Args:
        close: Raw price series, already ``dropna()``-ed.

    Returns:
        Total return, or ``None`` when the entry price is not strictly
        positive and no honest percentage exists.
    """
    if len(close) < 2:
        return None
    first = float(close.iloc[0])
    last = float(close.iloc[-1])
    # ``inf > 0`` is True, so an infinite entry price would otherwise yield a
    # clean-looking -100% instead of "no honest percentage exists".
    if not (np.isfinite(first) and first > 0) or not np.isfinite(last):
        return None
    return last / first - 1.0


def win_rate_and_stats(trades: List[TradeRecord]) -> Dict[str, float]:
    """Win rate and P&L statistics from completed trades.

    Args:
        trades: Completed round-trip trades.

    Returns:
        Dict with win_rate, profit_loss_ratio, max_consecutive_loss,
        avg_holding_bars, profit_factor.
    """
    if not trades:
        return {
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "max_consecutive_loss": 0,
            "avg_holding_bars": 0.0,
            "profit_factor": 0.0,
        }

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]

    win_rate = len(wins) / len(trades)

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else 1e-10
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    max_consec = 0
    cur_consec = 0
    for t in trades:
        if t.pnl < 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    hold_bars = [t.holding_bars for t in trades if t.holding_bars > 0]
    avg_holding = float(np.mean(hold_bars)) if hold_bars else 0.0

    return {
        "win_rate": win_rate,
        "profit_loss_ratio": round(profit_loss_ratio, 4),
        "max_consecutive_loss": max_consec,
        "avg_holding_bars": round(avg_holding, 1),
        "profit_factor": round(profit_factor, 4),
    }


def by_symbol_stats(trades: List[TradeRecord]) -> Dict[str, Dict[str, Any]]:
    """Per-symbol trade statistics.

    Args:
        trades: Completed round-trip trades.

    Returns:
        {symbol: {count, win_rate, total_pnl, avg_pnl}}.
    """
    groups: Dict[str, list] = {}
    for t in trades:
        groups.setdefault(t.symbol, []).append(t)

    result = {}
    for sym, sym_trades in groups.items():
        pnls = [t.pnl for t in sym_trades]
        wins = [p for p in pnls if p > 0]
        result[sym] = {
            "count": len(sym_trades),
            "win_rate": round(len(wins) / len(sym_trades), 4) if sym_trades else 0.0,
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(float(np.mean(pnls)), 2) if pnls else 0.0,
        }
    return result


def by_exit_reason_stats(trades: List[TradeRecord]) -> Dict[str, Dict[str, Any]]:
    """Per-exit-reason trade statistics.

    Args:
        trades: Completed round-trip trades.

    Returns:
        {reason: {count, total_pnl}}.
    """
    groups: Dict[str, list] = {}
    for t in trades:
        groups.setdefault(t.exit_reason, []).append(t)

    result = {}
    for reason, reason_trades in groups.items():
        pnls = [t.pnl for t in reason_trades]
        result[reason] = {
            "count": len(reason_trades),
            "total_pnl": round(sum(pnls), 2),
        }
    return result


def calc_turnover_series(positions: pd.DataFrame) -> pd.Series:
    """Per-bar weight-implied portfolio turnover from a position frame.

    Turnover for a bar is ``0.5 * sum_i |w_{t,i} - w_{t-1,i}|``, so a full
    rotation from one asset to another counts as 1.0 (matching the
    ``turnover_aware`` optimizer's convention). The first bar's turnover is
    ``0.5 * sum_i |w_{0,i}|``, treating the initial allocation as entry from
    cash. Turnover is measured on the weight frame the caller supplies. It
    does not know whether the execution engine filled, rounded, or rejected
    those target positions.

    Args:
        positions: Position-weight matrix (index=timestamp, columns=codes).

    Returns:
        Per-bar turnover series indexed like ``positions``; empty when the
        input is empty.
    """
    if positions is None or positions.empty:
        return pd.Series(dtype=float)
    filled = positions.fillna(0.0)
    prev = filled.shift(1).fillna(0.0)
    return 0.5 * (filled - prev).abs().sum(axis=1)


def calc_trade_turnover_series(
    trades: List[TradeRecord],
    equity_curve: pd.Series,
) -> pd.Series:
    """Per-bar turnover from actual entry and exit allocations.

    Each filled leg contributes its margin-equivalent traded value. Dividing
    gross traded value by twice the portfolio equity preserves the existing
    convention: entering a 100% allocation counts as 0.5 and rotating a 100%
    allocation between two assets counts as 1.0.

    Args:
        trades: Completed trades carrying actual entry/exit margin values.
        equity_curve: Portfolio equity used to normalize traded values.

    Returns:
        Per-bar realized turnover aligned to ``equity_curve``. Bars without
        fills are zero and remain part of the average-turnover denominator.
    """
    if equity_curve is None or equity_curve.empty:
        return pd.Series(dtype=float)

    traded_margin = pd.Series(0.0, index=equity_curve.index, dtype=float)
    for trade in trades:
        for timestamp, margin in (
            (trade.entry_time, trade.entry_margin),
            (trade.exit_time, trade.exit_margin),
        ):
            try:
                margin_value = float(margin)
            except (TypeError, ValueError):
                continue
            if (
                timestamp in traded_margin.index
                and np.isfinite(margin_value)
                and margin_value > 0
            ):
                traded_margin.loc[timestamp] += margin_value

    denominator = 2.0 * equity_curve.abs().replace(0.0, np.nan)
    return (traded_margin / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ─── Three-way split segment metrics (spec §4) ───

# Trade-count floors (EP004 EP004ValidLoss semantics): too few trades is
# statistical noise, scored as bad rather than passed.
MIN_TOTAL_TRADES = 30
MIN_VALID_TRADES = 50


def _segment_sharpe(seg_returns: pd.Series, bpy: int) -> float:
    """Sharpe for one segment's per-bar returns, same annualisation as the run."""
    if len(seg_returns) < 2:
        return 0.0
    arr = seg_returns.to_numpy(dtype=float, copy=False)
    if not np.isfinite(arr).all():
        return 0.0
    vol = float(seg_returns.std())
    sharpe = float(seg_returns.mean() / (vol + 1e-10) * np.sqrt(bpy))
    return sharpe if np.isfinite(sharpe) else 0.0


def _segment_max_dd(seg_equity: pd.Series) -> float:
    if len(seg_equity) == 0:
        return 0.0
    peak = seg_equity.cummax()
    dd = (seg_equity - peak) / peak.replace(0, 1)
    return float(dd.min())


def _segment_bpy(
    seg_index: pd.DatetimeIndex,
    bars_per_year: Optional[int],
) -> int:
    """Annualisation factor for one segment (review M12).

    Mirrors calc_metrics: when bars_per_year is None (cross-market), derive the
    segment's own calendar-day factor from its first/last dates rather than
    assuming 252.
    """
    if bars_per_year is not None:
        return bars_per_year
    n = len(seg_index)
    if n < 2:
        return 252
    diff = seg_index[-1] - seg_index[0]
    calendar_days = diff.days if hasattr(diff, "days") else 0
    years = calendar_days / 365.25 if calendar_days > 0 else 1.0
    return int(n / years) if years > 0 else 252


def calc_segment_metrics(
    equity_curve: pd.Series,
    trades: List[TradeRecord],
    train_end: Any,
    valid_end: Any,
    bars_per_year: Optional[int] = 252,
) -> Dict[str, Any]:
    """Split equity into train/valid/test segments and score each.

    Segment boundaries: train = [start, train_end), valid = [train_end,
    valid_end), test = [valid_end, end]. A trade belongs to the segment its
    EXIT falls in (review M7): a cross-segment position's realised PnL lands in
    the closing segment's equity, so its n_trades is counted there too, keeping
    n_trades aligned with segment equity.

    NOTE (calibration difference vs EP004): per-segment Sharpe here uses the
    engine's per-bar returns, no natural-day zero-filling, and the run's own
    annualisation (possibly calendar-day for cross-market). EP004 slices on
    close_date, zero-fills natural days, and uses √365. unified_score is
    internally self-consistent but NOT directly comparable to EP004 fixtures.

    Returns:
        {"train": {...}, "valid": {...}, "test": {...}} with sharpe,
        total_return, max_drawdown, n_trades per segment; empty segments have
        n_trades=0 and null metrics.
    """
    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(valid_end)
    idx = equity_curve.index
    port_ret = equity_curve.pct_change().fillna(0.0)

    def _slice(mask) -> Dict[str, Any]:
        seg_eq = equity_curve[mask]
        seg_ret = port_ret[mask]
        n = len(seg_eq)
        if n == 0:
            return {"sharpe": None, "total_return": None, "max_drawdown": None, "n_trades": 0}
        bpy = _segment_bpy(seg_eq.index, bars_per_year)
        # Segment total return is relative to the segment's OWN start (path-
        # consistent: it answers "what did this window return"), not the run's
        # initial cash.
        total_ret = float(seg_eq.iloc[-1] / seg_eq.iloc[0] - 1) if seg_eq.iloc[0] > 0 else 0.0
        # n_trades by EXIT segment (cross-segment realised PnL lands here too).
        seg_start, seg_stop = seg_eq.index[0], seg_eq.index[-1]
        n_trades = sum(1 for t in trades if seg_start <= t.exit_time <= seg_stop)
        return {
            "sharpe": _segment_sharpe(seg_ret, bpy),
            "total_return": total_ret,
            "max_drawdown": _segment_max_dd(seg_eq),
            "n_trades": n_trades,
        }

    return {
        "train": _slice(idx < te),
        "valid": _slice((idx >= te) & (idx < ve)),
        "test": _slice(idx >= ve),
    }


def calc_unified_score(segments: Dict[str, Any]) -> Optional[float]:
    """EP004ValidLoss negated (higher is better): valid − 0.5·max(0, train − valid).

    The gap penalty punishes a strategy that fits train far better than valid —
    the overfitting signature. Returns None when either segment Sharpe is
    unavailable (no split, empty segment, or an insufficient-trade floor hit).
    """
    train = segments.get("train") or {}
    valid = segments.get("valid") or {}
    train_sharpe = train.get("sharpe")
    valid_sharpe = valid.get("sharpe")
    if train_sharpe is None or valid_sharpe is None:
        return None
    return float(valid_sharpe - 0.5 * max(0.0, train_sharpe - valid_sharpe))


def validation_floor(
    segments: Optional[Dict[str, Any]],
    total_n_trades: int,
) -> Optional[str]:
    """Trade-count floor (spec §4.2). Returns the failing segment or None.

    - valid segment present but n_trades < MIN_VALID_TRADES → "valid"
    - overall n_trades < MIN_TOTAL_TRADES → "overall"
    Checked valid-first so the more specific floor wins.
    """
    if segments:
        valid = segments.get("valid") or {}
        valid_n = valid.get("n_trades", 0)
        if valid.get("sharpe") is not None and valid_n < MIN_VALID_TRADES:
            return "valid"
        # Empty test segment (valid_end == end_date) → mark "test".
        test = segments.get("test") or {}
        if test.get("sharpe") is None and test.get("n_trades", 0) == 0:
            return "test"
    if total_n_trades < MIN_TOTAL_TRADES:
        return "overall"
    return None


def _calc_attribution(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    beta: float,
    bpy: int,
    positions: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """alpha/beta return attribution (spec §5.2).

    The decomposition is an APPROXIMATION, not an identity (review M8):
    ``total_return`` is a compounded product while ``beta_contribution =
    β·bench_total`` is linear, so ``total_return ≠ alpha + beta`` in general —
    the residual absorbs the α·β·bench interaction term and any non-linearity.
    The honest arithmetic form is also reported (``alpha_arith = Σ(port − β·bench)``,
    ``beta_arith = β·Σbench``); a negative-beta book in a strong bull benchmark
    amplifies the residual, so R² and the residual share are reported too.

    Args:
        port_ret: Portfolio per-bar returns (fillna(0), aligned).
        bench_ret: Benchmark per-bar returns (reindexed, fillna(0)).
        beta: Regression beta (same basis as ``benchmark_beta``).
        bpy: Annualisation factor.
        positions: Target-weight frame for exposure means (optional).
    """
    bench_var = float(bench_ret.var()) if len(bench_ret) > 1 else 0.0
    port_var = float(port_ret.var()) if len(port_ret) > 1 else 0.0

    alpha_d = port_ret - beta * bench_ret
    alpha_annual = float(alpha_d.mean()) * bpy
    # Single-variable regression: R² = squared correlation.
    r_squared = 0.0
    if port_var > 0 and bench_var > 0:
        r_squared = float(beta * beta * bench_var / port_var)

    bench_total = float((1 + bench_ret).prod() - 1)
    port_total = float((1 + port_ret).prod() - 1)
    beta_contribution = beta * bench_total
    alpha_contribution = port_total - beta_contribution
    residual = port_total - (alpha_contribution + beta_contribution)

    out: Dict[str, Any] = {
        "beta": float(beta),
        "alpha_annual": float(alpha_annual),
        "r_squared": float(r_squared),
        "beta_contribution": float(beta_contribution),
        "alpha_contribution": float(alpha_contribution),
        # Honest arithmetic decomposition + residual (approximation disclosed).
        "alpha_arith": float(alpha_d.sum()),
        "beta_arith": float(beta * bench_ret.sum()),
        "residual": float(residual),
        "residual_share": float(abs(residual) / abs(port_total)) if abs(port_total) > 1e-12 else 0.0,
        "approximation": (
            "alpha/beta split is linear; residual absorbs interaction + non-linearity"
        ),
    }

    if positions is not None and len(positions) > 0:
        w = positions.fillna(0.0)
        net = w.sum(axis=1).abs()
        gross = w.abs().sum(axis=1)
        out["net_exposure"] = float(net.mean())
        invested = gross > 0
        out["net_exposure_invested"] = float(net[invested].mean()) if invested.any() else 0.0
        out["gross_exposure"] = float(gross.mean())
    return out


def calc_metrics(
    equity_curve: pd.Series,
    trades: List[TradeRecord],
    initial_cash: float,
    bars_per_year: Optional[int] = 252,
    bench_ret: Optional[pd.Series] = None,
    positions: Optional[pd.DataFrame] = None,
    turnover_series: Optional[pd.Series] = None,
    train_end: Any = None,
    valid_end: Any = None,
) -> Dict[str, Any]:
    """Full set of performance metrics.

    Args:
        equity_curve: Equity time series (index=timestamp, values=equity).
        trades: Completed round-trip trades.
        initial_cash: Starting capital.
        bars_per_year: Bars per year for annualisation. None = auto-detect
            from equity curve dates (calendar-day method, for cross-market).
        bench_ret: Benchmark per-bar return series (optional).
        positions: Position-weight frame used as a backward-compatible
            turnover fallback when ``turnover_series`` is not supplied.
        turnover_series: Actual per-bar execution turnover (optional). When
            supplied, it takes precedence over position-implied turnover.
        train_end: Optional three-way split boundary (spec §4). With
            ``valid_end``, adds ``segments`` + ``unified_score``.
        valid_end: Optional three-way split boundary.

    Returns:
        Metrics dictionary (compatible with daily_portfolio format).
    """
    if len(equity_curve) == 0:
        return _empty_metrics(initial_cash)

    n = len(equity_curve)

    # Calendar-day annualization for cross-market (bars_per_year=None)
    if bars_per_year is None:
        first, last = equity_curve.index[0], equity_curve.index[-1]
        diff = last - first
        calendar_days = diff.days if hasattr(diff, "days") else 0
        years = calendar_days / 365.25 if calendar_days > 0 else 1.0
        bpy = int(n / years) if years > 0 else 252
    else:
        bpy = bars_per_year

    port_ret = equity_curve.pct_change().fillna(0.0)
    # Equity that touches zero then recovers (100 → 0 → 50) yields non-finite
    # pct_change values; options metrics already skip risk ratios in that case.
    returns_finite = bool(np.isfinite(port_ret.to_numpy(dtype=float, copy=False)).all())

    total_ret = float(equity_curve.iloc[-1] / initial_cash - 1)
    # A leveraged/short book can end at or below zero equity (``total_ret <= -1``).
    # ``(1 + total_ret) ** fractional`` would then raise a negative base to a
    # fractional power, which Python evaluates to a ``complex`` and crashes the
    # subsequent ``float(...)``. A total wipeout annualises to -100%.
    growth = 1 + total_ret
    if growth <= 0:
        ann_ret = -1.0
    else:
        # Explosive equity paths (e.g. 1 → 1e6 in a few bars) overflow
        # ``float(growth ** …)`` on CPython; treat as non-finite annualisation.
        try:
            ann_ret = float(growth ** (bpy / max(n, 1)) - 1)
        except OverflowError:
            ann_ret = float("inf")
        if not np.isfinite(ann_ret):
            ann_ret = float("inf")
    # ``Series.std()`` uses ddof=1, so a single-observation return series
    # (e.g. a one-bar backtest) yields NaN and poisons the Sharpe ratio.
    # Guard the small sample the same way ``downside_std`` is guarded below.
    vol = float(port_ret.std()) if len(port_ret) > 1 and returns_finite else 0.0
    sharpe = (
        float(port_ret.mean() / (vol + 1e-10) * np.sqrt(bpy))
        if returns_finite
        else 0.0
    )
    if not np.isfinite(sharpe):
        sharpe = 0.0

    # Drawdown
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak.replace(0, 1)
    max_dd = float(dd.min())

    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Sortino
    if returns_finite:
        downside = port_ret[port_ret < 0]
        downside_std = float(downside.std()) if len(downside) > 1 else 1e-10
        sortino = float(port_ret.mean() / (downside_std + 1e-10) * np.sqrt(bpy))
    else:
        sortino = 0.0
    if not np.isfinite(sortino):
        sortino = 0.0

    trade_stats = win_rate_and_stats(trades)

    # Prefer execution-derived turnover; retain the position-frame fallback
    # for external callers of calc_metrics that do not have fill records.
    turnover_values = (
        turnover_series.reindex(equity_curve.index).fillna(0.0).clip(lower=0.0)
        if turnover_series is not None
        else calc_turnover_series(positions)
        if positions is not None
        else pd.Series(dtype=float)
    )
    avg_turnover = float(turnover_values.mean()) if len(turnover_values) > 0 else 0.0
    total_turnover = float(turnover_values.sum()) if len(turnover_values) > 0 else 0.0

    # Benchmark comparison
    bench_return = 0.0
    excess = 0.0
    ir = 0.0
    tracking_error = 0.0
    bench_beta = 0.0
    if bench_ret is not None and len(bench_ret) > 0:
        bench_return = float((1 + bench_ret).prod() - 1)
        excess = total_ret - bench_return
        aligned_bench = bench_ret.reindex(port_ret.index).fillna(0.0)
        active_ret = port_ret - aligned_bench
        # Same ddof=1 small-sample guard as ``vol`` / ``downside_std`` so the
        # information ratio stays finite for a single-observation series.
        active_std = float(active_ret.std()) if len(active_ret) > 1 and returns_finite else 0.0
        ir = (
            float(active_ret.mean() / (active_std + 1e-10) * np.sqrt(bpy))
            if returns_finite
            else 0.0
        )
        if not np.isfinite(ir):
            ir = 0.0
        # The information ratio's own denominator, annualised. A
        # benchmark-relative mandate is written around this number, and it was
        # being computed and thrown away.
        tracking_error = active_std * np.sqrt(bpy) if returns_finite else 0.0
        if not np.isfinite(tracking_error):
            tracking_error = 0.0
        bench_var = float(aligned_bench.var()) if len(aligned_bench) > 1 else 0.0
        if returns_finite and bench_var > 0:
            covariance = float(port_ret.cov(aligned_bench))
            bench_beta = covariance / bench_var
            if not np.isfinite(bench_beta):
                bench_beta = 0.0

    # alpha/beta attribution (spec §5). None when no benchmark — never a crash.
    attribution: Optional[Dict[str, Any]] = None
    if bench_ret is not None and len(bench_ret) > 0 and returns_finite:
        attribution = _calc_attribution(port_ret, aligned_bench, bench_beta, bpy, positions)

    metrics: Dict[str, Any] = {
        "final_value": float(equity_curve.iloc[-1]),
        "total_return": total_ret,
        "annual_return": ann_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "calmar": round(calmar, 4),
        "sortino": round(sortino, 4),
        "win_rate": trade_stats["win_rate"],
        "profit_loss_ratio": trade_stats["profit_loss_ratio"],
        "profit_factor": trade_stats["profit_factor"],
        "max_consecutive_loss": trade_stats["max_consecutive_loss"],
        "avg_holding_days": trade_stats["avg_holding_bars"],
        "trade_count": len(trades),
        "benchmark_return": round(bench_return, 6),
        "excess_return": round(excess, 6),
        "information_ratio": round(ir, 4),
        "tracking_error": round(float(tracking_error), 6),
        "benchmark_beta": round(float(bench_beta), 4),
        "avg_turnover": round(avg_turnover, 6),
        "total_turnover": round(total_turnover, 6),
        "attribution": attribution,
    }

    # Three-way split segments + unified score + trade-count floor (spec §4).
    # No split → byte-identical legacy behaviour (regression): none of these
    # keys are added, and the trade floor is NOT applied (a legacy run with
    # few trades must not suddenly flip to insufficient).
    if train_end is not None and valid_end is not None:
        segments = calc_segment_metrics(
            equity_curve, trades, train_end, valid_end, bars_per_year,
        )
        metrics["segments"] = segments
        metrics["unified_score"] = calc_unified_score(segments)
        floor = validation_floor(segments, len(trades))
        if floor is not None:
            metrics["validation_insufficient"] = floor
            # Too-few-trades valid → statistical noise; unified_score is void.
            if floor == "valid":
                metrics["unified_score"] = None

    return metrics


def _empty_metrics(initial_cash: float) -> Dict[str, Any]:
    """Return zero-valued metrics when no data is available."""
    return {
        "final_value": initial_cash,
        "total_return": 0, "annual_return": 0, "max_drawdown": 0,
        "sharpe": 0, "calmar": 0, "sortino": 0,
        "win_rate": 0, "profit_loss_ratio": 0, "profit_factor": 0,
        "max_consecutive_loss": 0, "avg_holding_days": 0, "trade_count": 0,
        "benchmark_return": 0, "excess_return": 0, "information_ratio": 0,
        "tracking_error": 0.0, "benchmark_beta": 0.0,
        "avg_turnover": 0.0, "total_turnover": 0.0,
    }
