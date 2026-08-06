"""Read-only technical indicator tool.

Computes RSI, MACD, Bollinger Bands, SMA, and EMA for a given symbol using the
existing market-data pipeline. All computation is pure Python (numpy/pandas);
no new dependencies.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.agent.tools import BaseTool
from src.market_data import fetch_market_data

logger = logging.getLogger(__name__)

# ── Indicator defaults ────────────────────────────────────────────────────────
_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_BB_PERIOD = 20
_BB_STD = 2.0
_SMA_PERIODS = (20, 50, 200)
_EMA_PERIOD = 20
_DEFAULT_LOOKBACK = 200
_MAX_LOOKBACK = 500


def _compute_sma(close: pd.Series, period: int) -> float | None:
    """Simple moving average over the last *period* bars."""
    if len(close) < period:
        return None
    return float(close.iloc[-period:].mean())


def _compute_ema(close: pd.Series, period: int) -> float | None:
    """Exponential moving average over the full series."""
    if len(close) < period:
        return None
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


def _compute_rsi(close: pd.Series, period: int = _RSI_PERIOD) -> float | None:
    """Relative Strength Index (Wilder smoothing) over *period* bars."""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _compute_macd(
    close: pd.Series,
    fast: int = _MACD_FAST,
    slow: int = _MACD_SLOW,
    signal: int = _MACD_SIGNAL,
) -> dict[str, float | None] | None:
    """MACD line, signal line, and histogram."""
    if len(close) < slow + signal:
        return None
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd_line": round(float(macd_line.iloc[-1]), 4),
        "signal_line": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }


def _compute_bollinger(
    close: pd.Series,
    period: int = _BB_PERIOD,
    num_std: float = _BB_STD,
) -> dict[str, float | None] | None:
    """Bollinger Bands: upper, middle (SMA), lower."""
    if len(close) < period:
        return None
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return {
        "upper": round(float(sma.iloc[-1] + num_std * std.iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower": round(float(sma.iloc[-1] - num_std * std.iloc[-1]), 2),
    }


def _to_dataframe(obj: Any) -> pd.DataFrame | None:
    """Normalize ``fetch_market_data`` output into a DataFrame.

    ``fetch_market_data`` returns one of three shapes per symbol: a list of
    row dicts (no truncation), a ``cap_rows`` wrapper dict ``{rows, returned,
    truncated, policy, hint, data}`` (when truncated), or — for other callers —
    a raw DataFrame. Accept all three, and re-key dates onto the index so
    ``close.index`` carries real timestamps.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        df = pd.DataFrame(obj["data"])
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    else:
        return None
    if df.empty:
        return df
    date_col = next(
        (c for c in df.columns if str(c).lower() in ("date", "datetime", "trade_date", "day", "时间")),
        None,
    )
    if date_col is not None:
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        except Exception:  # noqa: BLE001 — non-date index is a cosmetic loss only
            pass
    return df


def _resolve_as_of(raw: Any) -> str:
    """Normalize the optional ``end_date`` argument to a ``YYYY-MM-DD`` string.

    空值/缺省 → 今天；非法格式 → 今天并记日志；未来日期 → 钳到今天。
    返回的字符串可直接用作 ``fetch_market_data`` 的 ``end_date``。
    """
    if raw is None:
        return datetime.now().strftime("%Y-%m-%d")
    value = str(raw).strip()
    if not value:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        logger.warning("invalid end_date %r, falling back to today", value)
        return datetime.now().strftime("%Y-%m-%d")
    today = datetime.now().date()
    if parsed.date() > today:
        logger.warning("end_date %s is in the future, clamping to today", value)
        return today.strftime("%Y-%m-%d")
    return parsed.strftime("%Y-%m-%d")


class TechnicalIndicatorTool(BaseTool):
    """Compute common technical indicators for a symbol.

    Fetches OHLCV data through the existing loader pipeline, then computes
    RSI, MACD, Bollinger Bands, SMA, and EMA. All math is pure Python; no
    new dependencies, no network calls beyond what the loaders already do.
    """

    name = "technical_indicators"
    description = (
        "Compute common technical indicators (RSI, MACD, Bollinger Bands, "
        "SMA, EMA) for a trading symbol. Uses the project's data loaders "
        "to fetch price history, then computes indicators locally."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": (
                    "Trading symbol, e.g. AAPL for US stocks, "
                    "600519.SH for A-shares, BTC-USDT for crypto."
                ),
            },
            "interval": {
                "type": "string",
                "description": "Bar interval: 1d (default), 1wk, or 1mo.",
                "default": "1d",
            },
            "lookback": {
                "type": "integer",
                "description": (
                    "Number of bars to fetch. Default 200, max 500. "
                    "More bars = more accurate long-term indicators (SMA 200)."
                ),
                "default": _DEFAULT_LOOKBACK,
            },
            "end_date": {
                "type": "string",
                "description": (
                    "As-of date YYYY-MM-DD (optional). When given, prices and "
                    "all indicators are computed only up to this date; defaults "
                    "to today. Future dates are clamped to today."
                ),
                "default": "",
            },
        },
        "required": ["symbol"],
    }
    repeatable = True
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        symbol = str(kwargs.get("symbol", "")).strip()
        interval = str(kwargs.get("interval", "1d")).strip()
        lookback_raw = kwargs.get("lookback", _DEFAULT_LOOKBACK)

        if not symbol:
            return json.dumps({"ok": False, "error": "symbol is required"})

        try:
            lookback = int(lookback_raw)
        except (TypeError, ValueError):
            lookback = _DEFAULT_LOOKBACK
        lookback = max(10, min(lookback, _MAX_LOOKBACK))

        # Fetch enough bars to cover the longest indicator window + buffer,
        # anchored at the resolved as-of date (defaults to today).
        end_date = _resolve_as_of(kwargs.get("end_date"))
        as_of = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (as_of - timedelta(days=lookback * 2)).strftime("%Y-%m-%d")

        try:
            data = fetch_market_data(
                codes=[symbol],
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                # max_rows=0: fetch the full window without cap_rows sampling.
                # The window (lookback*2 calendar days) bounds the bar count;
                # sampling would silently distort SMA/RSI computed on the bars.
                max_rows=0,
            )
        except Exception as exc:
            logger.debug("fetch_market_data failed for %s: %s", symbol, exc)
            return json.dumps({"ok": False, "error": f"Failed to fetch data: {exc}"})

        df = _to_dataframe(data.get(symbol))
        if df is None or df.empty:
            return json.dumps({"ok": False, "error": f"No data returned for {symbol}"})

        close_col = next(
            (c for c in ("close", "Close", "CLOSE", "adj_close") if c in df.columns), None
        )
        if close_col is None:
            return json.dumps({"ok": False, "error": "No close price column in data"})
        close = pd.to_numeric(df[close_col], errors="coerce").dropna()
        if close.empty:
            return json.dumps({"ok": False, "error": "No usable close prices in data"})

        # ── Compute indicators ────────────────────────────────────────────
        indicators: dict[str, Any] = {
            "rsi_14": _compute_rsi(close),
            "macd": _compute_macd(close),
            "bollinger": _compute_bollinger(close),
        }
        for period in _SMA_PERIODS:
            indicators[f"sma_{period}"] = _compute_sma(close, period)
        indicators[f"ema_{_EMA_PERIOD}"] = _compute_ema(close, _EMA_PERIOD)

        latest_close = float(close.iloc[-1]) if len(close) > 0 else None
        latest_date = str(close.index[-1])[:10] if hasattr(close, "index") and len(close) > 0 else None

        return json.dumps(
            {
                "ok": True,
                "symbol": symbol,
                "interval": interval,
                "latest_close": latest_close,
                "latest_date": latest_date,
                "indicators": indicators,
            },
            ensure_ascii=False,
            default=str,
        )
