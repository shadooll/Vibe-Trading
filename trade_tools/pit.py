"""PIT 无幸存者决策点抽样 + as-of 数据包生成器。

Track B 验证框架（``export_trade_data/track_b_plan.md`` §三·1 / §三·2）的数据基础：

- **无前视**：所有过滤、指标、regime 判定只用 <= 决策日的数据。
- **分层抽样**：趋势 / 震荡 / 熊市 各抽 N 点，同层点间距 >= 60 交易日，固定 seed
  可复现（决策点预注册，写死进 Phase 1 脚本）。
- **数据包**：每点一个 JSON dict（截至决策日的行情 + 指标快照 + regime 标签），
  agent 研究时只读数据包，摸不到实盘工具。

与 ``export_trade_data/daily_check.py`` 同一口径（MA200 + MA60 判 regime）；
与 ``execution.py`` 共用 OHLCV 规范化和不复权概念——但本模块经 loader 注册表取数，
链路多数成员返回前复权价，见 :func:`build_data_pack`。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from trade_tools.execution import _norm_df

# ── 常数（命名常量，语义一目了然）──────────────────────────────────────────────
DEFAULT_LOOKBACK = 400  # 数据包默认回溯交易日数
MIN_SNAPSHOT_BARS = 200  # 计算 MA200 所需的最小 bar 数
MA60_UP_SHIFT = 5  # daily_check: ma60_up = 今 MA60 > 5 个交易日前 MA60（iloc[-6]）
MIN_LISTING_YEARS = 3  # 上市 >= 3 年
MIN_AVG_AMOUNT_60D = 1e8  # 近 60 日均成交额 > 1 亿
MIN_BARS_60D = 60  # 近 60 个交易日窗口
SUSPENSION_GAP_DAYS = 5  # 决策日距最近 bar 的自然日上限（> 此值视为停牌中）
MIN_BARS_30D = 15  # 近 30 自然日窗口内最少 bar 数（低于此值视为长期停牌）
MIN_POINT_SPACING = 60  # 同 regime 层内决策点间距（交易日）
DEFAULT_SEED = 42
ST_LIMIT_MAX_RET = 0.055  # ST 近似：主板 ST 日涨跌幅限制 ±5%（普通 ±10%）

REGIME_TREND = "趋势"
REGIME_RANGE = "震荡"
REGIME_BEAR = "熊市"
REGIME_UNKNOWN = "未知"
# 抽样分层（不含"未知"——未知 = 不交易，不抽）。
STRATUM_ORDER: tuple[str, ...] = (REGIME_TREND, REGIME_RANGE, REGIME_BEAR)

# a_share 链路中返回前复权价的来源（用于数据包 price_basis 标注）。
_QFQ_SOURCES = frozenset({"tencent", "baostock", "akshare", "tushare"})


# ── regime 判定（与 daily_check.py 完全同一算法）────────────────────────────────
def _regime_flags(
    close: pd.Series, as_of: str | pd.Timestamp
) -> tuple[bool, bool, bool] | None:
    """返回 (价>MA200, 价<MA200, MA60 向上)；空序列返回 None。

    与 ``daily_check.market_state`` 逐项一致：MA60 向上 = 今 MA60 > 5 个交易日前
    MA60（``iloc[-6]``）。数据不足时比较得 False（与 daily_check 行为一致）。
    ``close`` 需按日期索引（DatetimeIndex）。
    """
    if close.empty:
        return None
    hist = close[close.index <= as_of]
    if hist.empty:
        return None
    c = hist.astype(float)
    last = float(c.iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    ma60 = float(c.rolling(60).mean().iloc[-1])
    ma60_prev = (
        float(c.rolling(60).mean().iloc[-(MA60_UP_SHIFT + 1)])
        if len(c) >= MA60_UP_SHIFT + 1
        else float("nan")
    )
    return last > ma200, last < ma200, ma60 > ma60_prev


def regime_at(close: pd.Series, as_of: str | pd.Timestamp) -> str:
    """三态 regime：趋势 / 震荡 / 未知（与 daily_check.market_state 完全一致）。

    - 趋势：价 > MA200 且 MA60 向上
    - 震荡：价 < MA200（含 MA60 向下 = Track A 状态机里的"熊市"子类）
    - 未知：其余（价 >= MA200 但 MA60 未向上，或数据不足）
    """
    flags = _regime_flags(close, as_of)
    if flags is None:
        return REGIME_UNKNOWN
    above, below, up = flags
    if above and up:
        return REGIME_TREND
    if below:
        return REGIME_RANGE
    return REGIME_UNKNOWN


def regime_stratum(close: pd.Series, as_of: str | pd.Timestamp) -> str:
    """四态分层：趋势 / 震荡 / 熊市 / 未知，供抽样分层用。

    与 :func:`regime_at` 唯一区别：把"价 < MA200"按 MA60 方向拆成
    ``震荡``（MA60 向上）和 ``熊市``（MA60 向下）两层，对应 track_b_plan §三·2
    的"趋势 10 + 震荡 10 + 熊市 10"分层。
    """
    flags = _regime_flags(close, as_of)
    if flags is None:
        return REGIME_UNKNOWN
    above, below, up = flags
    if above and up:
        return REGIME_TREND
    if below:
        return REGIME_RANGE if up else REGIME_BEAR
    return REGIME_UNKNOWN


# ── PIT 宇宙过滤 ──────────────────────────────────────────────────────────────
def _board_ok(symbol: str) -> bool:
    """仅主板：``60xxxx.SH`` / ``00xxxx.SZ``；排除 300(创业板)/688(科创板)/8xxxxx(北交所)。"""
    if "." not in symbol:
        return False
    code, suffix = symbol.split(".", maxsplit=1)
    if suffix not in ("SH", "SZ") or not (len(code) == 6 and code.isdigit()):
        return False
    return code.startswith("60") if suffix == "SH" else code.startswith("00")


def _default_st_estimate(df: pd.DataFrame) -> bool:
    """PIT ST 近似判定：主板 ST 股日涨跌幅限制为 ±5%（普通股 ±10%）。

    若决策日前近 60 个交易日的日涨跌幅绝对值从未超过 ``ST_LIMIT_MAX_RET``，
    判为"疑似 ST"。局限（如实写明）：

    - 数据通道（loader 注册表）不暴露 namechange 简称历史，无法精确 PIT 判 ST；
    - 会误伤长期横盘、无大波动的普通股；
    - 无法识别刚戴帽/摘帽、仍按 ±10% 涨跌幅交易的过渡期。

    有简称历史时用 ``UniverseFilter(st_check=...)`` 覆盖；不想判 ST 传
    ``lambda df: False``。
    """
    if len(df) < MIN_BARS_60D:
        return False  # 数据不足不判（宁放勿杀）
    ret = df["close"].astype(float).pct_change().tail(MIN_BARS_60D).dropna()
    if ret.empty:
        return False
    return bool(ret.abs().max() < ST_LIMIT_MAX_RET)


@dataclass(frozen=True)
class UniverseVerdict:
    """一次 PIT 宇宙过滤的判定结果。"""

    eligible: bool
    failures: tuple[str, ...] = ()
    checks: dict[str, object] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.eligible


@dataclass(frozen=True)
class UniverseFilter:
    """PIT 宇宙过滤规则：给定标的截至决策日的价格历史，判断是否有资格入池。

    全部检查只用 <= 决策日的数据（无前视）。``df`` 应为已规范化（DatetimeIndex、
    open/high/low/close/volume）的日线；为精确判"上市 >= 3 年"，最好回溯到上市日
    （至少回溯到决策日前 3 年），否则"上市年限"退化为"最早可见 bar 距今 >= 3 年"。

    Attributes:
        min_listing_years: 上市年限门槛（最早可见 bar 距今）。
        min_avg_amount_60d: 近 60 日均成交额门槛（close*volume 近似，严格大于）。
        volume_multiplier: volume 换算到"股"的系数。A 股数据源 volume 单位不一：
            tencent/tushare 为手（一手 100 股，传 100.0），baostock 为股（默认 1.0）。
        suspension_gap_days: 决策日距最近 bar 的自然日上限（超过视为停牌中）。
        st_check: ST 判定函数，接收 <= 决策日的行情、返回"疑似 ST"；None = 不判。
    """

    min_listing_years: int = MIN_LISTING_YEARS
    min_avg_amount_60d: float = MIN_AVG_AMOUNT_60D
    volume_multiplier: float = 1.0
    suspension_gap_days: int = SUSPENSION_GAP_DAYS
    st_check: Callable[[pd.DataFrame], bool] | None = _default_st_estimate

    def judge(
        self, symbol: str, df: pd.DataFrame, as_of: str | pd.Timestamp
    ) -> UniverseVerdict:
        """判定 ``symbol`` 截至 ``as_of`` 是否有资格进入 PIT 标的池。"""
        failures: list[str] = []
        checks: dict[str, object] = {}

        if not _board_ok(symbol):
            failures.append(f"{symbol} 非主板（需 60xxxx.SH / 00xxxx.SZ）")
            return UniverseVerdict(
                eligible=False, failures=tuple(failures), checks=checks
            )

        ts = pd.Timestamp(as_of)
        if not isinstance(df.index, pd.DatetimeIndex):
            df = _norm_df(df)  # 允许传入 date 列帧；热路径（抽样）已传规范化切片，跳过
        if df.empty:
            return UniverseVerdict(
                eligible=False, failures=("决策日无行情",), checks=checks
            )
        hist = (
            df if df.index[-1] <= ts else df[df.index <= ts]
        )  # 防御性截断，确保无前视
        if hist.empty:
            return UniverseVerdict(
                eligible=False, failures=("决策日无行情",), checks=checks
            )

        if self.st_check is not None and self.st_check(hist):
            failures.append("疑似 ST/*ST（近似：近 60 日日涨跌幅从未超过 ±5.5%）")

        first = hist.index[0]
        age_limit = ts - pd.DateOffset(years=self.min_listing_years)
        checks["earliest_bar_date"] = str(first.date())
        listing_ok = bool(first <= age_limit)
        checks["listing_age_ok"] = listing_ok
        if not listing_ok:
            failures.append(
                f"上市不满 {self.min_listing_years} 年（最早可见 bar {first.date()}，"
                f"需 <= {age_limit.date()}）"
            )

        amount = (
            hist["close"].astype(float)
            * hist["volume"].astype(float)
            * self.volume_multiplier
        )
        avg60 = (
            float(amount.tail(MIN_BARS_60D).mean())
            if len(hist) >= MIN_BARS_60D
            else float("nan")
        )
        checks["avg_amount_60d"] = avg60
        amount_ok = len(hist) >= MIN_BARS_60D and avg60 > self.min_avg_amount_60d
        if not amount_ok:
            failures.append(
                f"近 60 日均成交额 {avg60:.3e} 未超过 {self.min_avg_amount_60d:.0e}"
            )

        gap = int((ts - hist.index[-1]).days)
        checks["gap_to_decision_days"] = gap
        if gap > self.suspension_gap_days:
            failures.append(f"决策日前 {gap} 天无行情（疑似停牌）")
        window_start = ts - pd.Timedelta(days=30)
        bars_30d = int(((hist.index >= window_start) & (hist.index <= ts)).sum())
        checks["bars_in_30d"] = bars_30d
        if bars_30d < MIN_BARS_30D:
            failures.append(f"近 30 日仅 {bars_30d} 个 bar（疑似长期停牌）")

        return UniverseVerdict(
            eligible=not failures, failures=tuple(failures), checks=checks
        )


# ── 决策点抽样 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Point:
    """一个决策点：标的 + 决策日 + regime 标签（预注册、固定 seed 抽取）。"""

    symbol: str
    date: str
    regime: str


def _trading_day_distance(
    dates: pd.DatetimeIndex, a: pd.Timestamp, b: pd.Timestamp
) -> int:
    """两日期之间的交易日距离（含端点；同一天 = 1）。

    用给定标的自身的日线日历计数。全部 A 股共享同一交易日历，跨标的计数与
    自身标的日历一致；标的停牌只会让计数偏小，等价于间距更保守（更宽）。
    """
    lo, hi = (a, b) if a <= b else (b, a)
    return int(((dates >= lo) & (dates <= hi)).sum())


def sample_points(
    universe: Mapping[str, pd.DataFrame],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    per_regime: int,
    *,
    seed: int = DEFAULT_SEED,
    min_spacing: int = MIN_POINT_SPACING,
    regime_fn: Callable[[pd.Series, str | pd.Timestamp], str] = regime_stratum,
    universe_filter: UniverseFilter | None = None,
    market_bars: pd.Series | None = None,
) -> list[Point]:
    """确定性（固定 seed）分层抽样决策点。

    每个候选 ``(symbol, date)`` 只用该标的 <= date 的行情判定（无前视）：
    过 ``universe_filter``（若给出）且归入当前层才进入候选池。同层内按固定
    seed 洗牌后贪心挑选，任意两个已选点在同层内的交易日间距 >= ``min_spacing``。

    **regime 判定对象**（Pilot 实测修正）：决策点的 regime 标签必须反映**大盘**
    状态（手册环①：用 510300 判 趋势/震荡/未知 决定能用哪套理由），不是个股状态。
    传 ``market_bars``（如 510300.SH 的收盘序列）则每个点按大盘判 regime；
    不传则回退按个股判（纯分层/测试用）。

    Args:
        universe: ``{symbol: 日线}``（date 列或 DatetimeIndex 均可），完整历史。
        start_date / end_date: 决策日候选范围（含端点）。
        per_regime: 每层目标点数；池子不够时该层返回实际抽到的点。
        seed: 固定随机种子，保证可复现。
        min_spacing: 同层决策点最小交易日间距。
        regime_fn: 分层函数（默认 :func:`regime_stratum`，趋势/震荡/熊市/未知）。
        universe_filter: PIT 宇宙过滤器；None = 不做宇宙过滤（只分层）。
        market_bars: 大盘收盘序列（DatetimeIndex）。给定时每个点的 regime =
            大盘在决策日的判定；缺省按个股判。
    """
    if per_regime <= 0:
        raise ValueError(f"per_regime 必须 > 0，得到 {per_regime}")
    if min_spacing <= 0:
        raise ValueError(f"min_spacing 必须 > 0，得到 {min_spacing}")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end:
        raise ValueError(f"start_date {start.date()} > end_date {end.date()}")

    frames = {symbol: _norm_df(frame) for symbol, frame in universe.items()}
    market = None
    if market_bars is not None:
        market = market_bars.astype(float).sort_index()
    rng = random.Random(seed)
    points: list[Point] = []

    for stratum in STRATUM_ORDER:
        candidates: list[tuple[pd.Timestamp, str]] = []
        for symbol in sorted(frames):
            f = frames[symbol]
            dates = f.index
            close = f["close"].astype(float)
            for i, date in enumerate(dates):
                if not (start <= date <= end):
                    continue
                hist = f.iloc[: i + 1]
                if universe_filter is not None:
                    if not universe_filter.judge(symbol, hist, date).eligible:
                        continue
                # regime 用大盘（若给）判，不用个股——环①决定用哪套理由。
                regime_series = market if market is not None else close.iloc[: i + 1]
                if regime_fn(regime_series, date) != stratum:
                    continue
                candidates.append((date, symbol))

        stratum_rng = random.Random(rng.getrandbits(128))
        stratum_rng.shuffle(candidates)
        picked: list[tuple[pd.Timestamp, str]] = []
        for date, symbol in candidates:
            dates = frames[symbol].index
            if all(
                _trading_day_distance(dates, date, pdate) >= min_spacing
                for pdate, _ in picked
            ):
                picked.append((date, symbol))
                if len(picked) >= per_regime:
                    break
        points.extend(
            Point(symbol=symbol, date=str(date.date()), regime=stratum)
            for date, symbol in picked
        )

    return sorted(points, key=lambda p: (p.date, p.symbol))


# ── as-of 数据包生成 ──────────────────────────────────────────────────────────
def compute_snapshot(df: pd.DataFrame) -> dict:
    """截至最后一根 bar 的指标快照，与 ``daily_check.compute_state`` 完全一致。

    仅用 ``df`` 内的数据（调用方负责把 ``df`` 截断到 <= 决策日）。要求 >= 200 根
    bar（否则 MA200 为 NaN）；缺失时返回与 daily_check 相同的 NaN 行为。
    """
    c = df["close"].astype(float)
    last = float(c.iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma60 = float(c.rolling(60).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    ma60_prev = float(c.rolling(60).mean().iloc[-(MA60_UP_SHIFT + 1)])
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1]
    al = loss.ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1]
    rsi = float(100 - 100 / (1 + ag / al)) if al else 100.0
    v = df["volume"].astype(float)
    vol_ratio = float(v.iloc[-1] / v.iloc[-6:-1].mean())
    high20 = float(c.iloc[-21:-1].max())
    ret20 = (last / float(c.iloc[-21]) - 1) * 100
    return {
        "close": last,
        "ma20": ma20,
        "ma60": ma60,
        "ma200": ma200,
        "ma60_up": ma60 > ma60_prev,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "high20": high20,
        "new_high20": last > high20,
        "ret20": ret20,
    }


def _fetch_frame(symbol: str, as_of: str, lookback: int) -> tuple[pd.DataFrame, str]:
    """经 loader 注册表抓取截至 ``as_of``（含）的日线；返回 (frame, 来源名)。

    绝不直接调 yfinance/tushare——统一走 ``backtest.loaders.registry`` 的 a_share
    回退链，与回测引擎同一通道。回溯起点按 ``lookback * 7/5 + 30`` 自然日放宽，
    覆盖节假日与停牌造成的缺口。
    """
    from backtest.loaders.registry import resolve_loader

    loader = resolve_loader("a_share")
    cal_days = int(round(lookback * 7 / 5)) + 30
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=cal_days)).strftime("%Y-%m-%d")
    data = loader.fetch([symbol], start, as_of, interval="1d")
    frame = data.get(symbol)
    if frame is None or frame.empty:
        raise ValueError(f"{symbol} 截至 {as_of} 无行情")
    return _norm_df(frame), loader.name


def _error_pack(symbol: str, ts: pd.Timestamp, reason: str) -> dict:
    return {
        "ok": False,
        "symbol": symbol,
        "as_of_date": str(ts.date()),
        "reason": reason,
    }


def build_data_pack(
    symbol: str,
    as_of_date: str | pd.Timestamp,
    lookback: int = DEFAULT_LOOKBACK,
    *,
    frame: pd.DataFrame | None = None,
) -> dict:
    """生成截至 ``as_of_date``（含）的 as-of 决策数据包。

    返回可直接 ``json.dumps`` 的 dict：``ok / symbol / as_of_date / source /
    price_basis / n_bars / bars(行情) / snapshot(指标快照) / regime(三态标签)``。
    所有值只用 <= 决策日的 bar（无前视），指标口径与 ``daily_check.compute_state``
    一致。

    Args:
        symbol: 标的代码，如 ``600519.SH``。
        as_of_date: 决策日（含）；抓取和所有指标都截止到这一天。
        lookback: 数据包回溯的交易日数（需 >= 200 才能算 MA200）。
        frame: 注入现成 DataFrame（离线/测试用，避免网络）。缺省时经
            ``backtest.loaders.registry`` 的 a_share 链抓取。

    复权说明：链路多数成员返回前复权价（tencent/baostock/akshare/tushare），
    ``price_basis`` 如实标注；指标为比值型（MA/RSI/涨跌幅），对复权基准不敏感。
    若需原始不复权价（例如对齐 ExecutionSimulator 的止损/目标 tick），请用
    ``frame=`` 注入不复权数据。
    """
    ts = pd.Timestamp(as_of_date)
    if frame is None:
        raw, source = _fetch_frame(symbol, ts.strftime("%Y-%m-%d"), lookback)
    else:
        raw, source = _norm_df(frame), "injected"

    hist = raw[raw.index <= ts].tail(lookback)
    if hist.empty:
        return _error_pack(symbol, ts, "决策日无行情")
    if len(hist) < MIN_SNAPSHOT_BARS:
        return _error_pack(
            symbol, ts, f"可用 bar {len(hist)} < {MIN_SNAPSHOT_BARS}，无法计算 MA200"
        )

    snapshot = compute_snapshot(hist)
    regime = regime_at(hist["close"], hist.index[-1])
    price_basis = "qfq" if source in _QFQ_SOURCES else "as_served"
    if source == "injected":
        price_basis = "as_provided"

    return {
        "ok": True,
        "symbol": symbol,
        "as_of_date": str(ts.date()),
        "source": source,
        "price_basis": price_basis,
        "n_bars": len(hist),
        "bars": [
            {
                "date": str(date.date()),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": float(row["volume"]),
            }
            for date, row in hist.iterrows()
        ],
        "snapshot": snapshot,
        "regime": regime,
    }


__all__ = [
    "DEFAULT_LOOKBACK",
    "MIN_SNAPSHOT_BARS",
    "MIN_POINT_SPACING",
    "DEFAULT_SEED",
    "REGIME_TREND",
    "REGIME_RANGE",
    "REGIME_BEAR",
    "REGIME_UNKNOWN",
    "STRATUM_ORDER",
    "Point",
    "UniverseVerdict",
    "UniverseFilter",
    "regime_at",
    "regime_stratum",
    "sample_points",
    "compute_snapshot",
    "build_data_pack",
]
