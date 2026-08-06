"""Shared synthetic-bar builders for trade_tools tests."""

from __future__ import annotations

import datetime as _dt

import pandas as pd


def make_bars(prices: list[list[float] | None], start="2024-01-02") -> pd.DataFrame:
    """Build a daily OHLCV frame from ``[open, high, low, close, volume]`` rows.

    Dates are consecutive business days from ``start``. A ``None`` row is
    skipped (simulates a suspended day).
    """
    rows: list[tuple] = []
    date = _dt.date.fromisoformat(start)
    for price in prices:
        if price is not None:
            rows.append((str(date), *price))
        date += _dt.timedelta(days=1)
        while date.weekday() >= 5:
            date += _dt.timedelta(days=1)
    return pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume"]
    )


def make_scenario(
    signal_close: float,
    after_signal: list[list[float] | None],
    signal_vol: float = 1_000_000.0,
    start="2024-01-02",
) -> pd.DataFrame:
    """Signal-day bar (at ``start``, flat at ``signal_close``) plus following bars.

    The signal day sits at ``start``; entry happens on the next bar. This gives
    every entry bar a real ``pre_close`` for the limit-up/down checks.
    """
    signal = [signal_close, signal_close, signal_close, signal_close, signal_vol]
    return make_bars([signal, *after_signal], start=start)
