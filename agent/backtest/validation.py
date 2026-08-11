"""Statistical validation for backtest results.

Three independent tools:
  - Monte Carlo permutation test: is the strategy significantly better than random?
  - Bootstrap Sharpe CI: how stable is the risk-adjusted return?
  - Walk-Forward analysis: is performance consistent across time windows?

Usage: called automatically by BaseEngine.run_backtest when config[\"validation\"]
is present, or invoked directly on backtest outputs.
"""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from backtest.models import TradeRecord

_EULER_MASCHERONI = 0.5772156649


# ─── Deflated Sharpe Ratio ───


def deflated_sharpe_ratio(
    trial_scores: Sequence[float],
    daily_returns: Sequence[float],
    bars_per_year: int,
    mode: str = "oos",
    seed: int = 42,
) -> Dict[str, Any]:
    """Deflated Sharpe Ratio (Bailey & López de Prado).

    Answers the question the plain Sharpe cannot: *is this result skill, or the
    best of N lucky trials?* The agent IS the optimizer here — it edits, reruns,
    and hands over the best variant — so the observed Sharpe must be deflated by
    the search that produced it.

    Algorithm (faithful to EP004 ``deflated_sharpe.py``):
      - ``sr_var = var(trial_scores, ddof=1)`` — spread of the search's scores.
      - Luck baseline daily Sharpe
        ``sr0 = sqrt(sr_var / bars_per_year) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))``.
      - Observed daily Sharpe from the strategy's own daily returns.
      - Non-normality correction via skew ``g3`` and kurtosis ``g4 = kurt + 3``:
        ``denom = 1 - g3·sr + (g4-1)/4 · sr²``; ``z = (sr - sr0)·√(T-1)/√denom``;
        ``DSR = Φ(z)``.

    Args:
        trial_scores: One objective score per trial of the search (recommended:
            ``valid_sharpe``). ``N = len(trial_scores)``.
        daily_returns: The SELECTED strategy's daily equity returns (not annualised).
        bars_per_year: The SAME annualisation factor the scores used — crypto/forex
            365, A-share 252. Never hard-default 252: it inflates crypto's sr0 by
            √(365/252)≈1.20 and biases DSR low (review M2).
        mode: "oos" (default; requires out-of-sample scores) or "train_only"
            (explicit escape; the result is flagged in-sample / inflated).
        seed: Unused for the math (deterministic); kept for a stable signature.

    Returns:
        Dict with n_trials, sr_var, observed_sharpe_annual, sr0_annual, T_days,
        skew, kurtosis, DSR, verdict, reason, authoritative. DSR is None when not
        computable (N<2, T<30, zero variance, non-positive denom) — never raises.
    """
    del seed  # deterministic; signature stability only
    result: Dict[str, Any] = {
        "n_trials": 0,
        "sr_var": None,
        "observed_sharpe_annual": None,
        "sr0_annual": None,
        "T_days": 0,
        "skew": None,
        "kurtosis": None,
        "DSR": None,
        "verdict": "unavailable",
        "reason": "",
        "authoritative": True,
    }

    if mode == "train_only":
        result["mode"] = "train_only"
        result["in_sample"] = True
    elif mode != "oos":
        result["reason"] = f"unknown mode {mode!r}"
        result["authoritative"] = False
        return result

    scores = np.asarray([s for s in trial_scores if s is not None], dtype=float)
    scores = scores[np.isfinite(scores)]
    N = len(scores)
    result["n_trials"] = N
    if N < 2:
        result["reason"] = f"need >= 2 trials, got {N}"
        return result

    rets = pd.Series(list(daily_returns), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    T = len(rets)
    result["T_days"] = T
    if T < 30:
        result["reason"] = f"need >= 30 daily returns, got {T}"
        return result

    std = float(rets.std())  # ddof=1
    # Near-zero std (e.g. a constant-return series whose ddof=1 std is a float
    # residue like 6.5e-19, not exactly 0) is still a zero-variance path: the
    # Sharpe denominator is meaningless and DSR must not divide by it.
    mean_abs = float(rets.abs().mean())
    if std <= 0 or not math.isfinite(std) or std < max(1e-12, mean_abs * 1e-6):
        result["reason"] = "daily returns have zero/non-finite variance"
        return result

    sr_var = float(np.var(scores, ddof=1))
    result["sr_var"] = sr_var

    norm = NormalDist()
    sr0_daily = math.sqrt(sr_var / bars_per_year) * (
        (1 - _EULER_MASCHERONI) * norm.inv_cdf(1 - 1 / N)
        + _EULER_MASCHERONI * norm.inv_cdf(1 - 1 / (N * math.e))
    )
    result["sr0_annual"] = sr0_daily * math.sqrt(bars_per_year)

    sr_daily = float(rets.mean()) / std
    result["observed_sharpe_annual"] = sr_daily * math.sqrt(bars_per_year)

    g3 = float(rets.skew())
    g4 = float(rets.kurt()) + 3.0  # pandas kurt is excess kurtosis
    result["skew"] = g3
    result["kurtosis"] = g4

    denom = 1 - g3 * sr_daily + (g4 - 1) / 4 * sr_daily**2
    if denom <= 0 or not math.isfinite(denom):
        result["reason"] = f"non-positive denominator ({denom:.4g})"
        return result

    z = (sr_daily - sr0_daily) * math.sqrt(T - 1) / math.sqrt(denom)
    dsr = norm.cdf(z)
    result["DSR"] = dsr

    if mode == "train_only":
        result["verdict"] = "in_sample"
        result["reason"] = "train-only mode: in-sample, inflated, not an out-of-sample conclusion"
    elif dsr >= 0.95:
        result["verdict"] = "significant"
    elif dsr >= 0.90:
        result["verdict"] = "weak"
    else:
        result["verdict"] = "not_significant"
    return result


# ─── Block Bootstrap Monte Carlo ───


def _equity_paths_payload(
    paths_matrix: "np.ndarray",
    actual: "np.ndarray",
    initial_capital: float,
    n_cols: int = 400,
    n_rows: int = 30,
) -> Dict[str, Any]:
    """Build the fan-chart payload shared by monte_carlo_test and block_bootstrap.

    Single schema (initial_capital + five percentile bands + row-sampled paths)
    so the frontend renders both with MonteCarloPathsChart unchanged. Keeping the
    construction here eliminates the two-site schema-drift trap. ``paths_matrix``
    is (n_paths, n_steps); ``actual`` is the observed equity series (n_steps).
    """
    n_steps = paths_matrix.shape[1]
    col_idx = np.unique(np.linspace(0, n_steps - 1, min(n_steps, n_cols)).astype(int))
    sample_rows = np.unique(
        np.linspace(0, paths_matrix.shape[0] - 1, min(n_rows, paths_matrix.shape[0])).astype(int)
    )
    payload: Dict[str, Any] = {
        "steps": (col_idx + 1).tolist(),
        "initial_capital": round(float(initial_capital), 2),
        "actual": np.round(actual[col_idx], 2).tolist(),
    }
    for q in (5, 25, 50, 75, 95):
        payload[f"band_p{q}"] = np.round(
            np.percentile(paths_matrix[:, col_idx], q, axis=0), 2
        ).tolist()
    payload["samples"] = np.round(paths_matrix[np.ix_(sample_rows, col_idx)], 2).tolist()
    return payload


def block_bootstrap(
    returns: Sequence[float],
    n_bootstrap: int = 5000,
    block: int = 10,
    start: float = 1000.0,
    seed: int = 7,
    keep_paths: int = 400,
) -> Dict[str, Any]:
    """Block-bootstrap resampling → terminal-value distribution + prob(profit).

    Unlike the permutation test (which shuffles PnL and destroys the serial
    correlation of a trend strategy's win/loss runs) or the i.i.d. bootstrap,
    this resamples CONTIGUOUS blocks, preserving short-range autocorrelation
    (EP004 ``mc_bootstrap.py``, block=10). Output is the terminal equity
    distribution: prob(profit), P5/P50/P95 — the most direct answer to "how
    much of this curve is luck".

    Args:
        returns: Per-period returns. DEFAULT is day-level equity returns
            (``equity_curve.pct_change().dropna()``) — the portfolio-equity
            measure that matches ``calc_metrics``' Sharpe. Trade-level
            (``pnl / entry_margin``) is a per-margin return: overlapping or
            leveraged books OVERSTATE path variance, so scale by
            ``entry_margin / equity_at_entry`` first or treat it as an accepted
            approximation (review M4).
        n_bootstrap: Number of resampled paths.
        block: Block length; ``block=1`` degenerates to i.i.d. bootstrap.
        start: Starting equity for path reconstruction.
        seed: Random seed (reproducible via ``np.random.default_rng``).
        keep_paths: How many full paths to retain for the fan chart.

    Returns:
        Dict with n, iters, block, prob_profit, orig_final, final_P5/P50/P95,
        mean_ret, equity_paths (fan-chart payload aligned to ``monte_carlo_test``).
    """
    rets = np.asarray(list(returns), dtype=float)
    rets = rets[np.isfinite(rets)]
    N = len(rets)
    if N < 5:
        return {"error": f"need at least 5 return observations, got {N}", "n": N}

    if block < 1:
        block = 1
    if N < block:
        block = max(1, N // 2)

    rng = np.random.default_rng(seed)
    finals = np.empty(n_bootstrap)
    # Retain up to keep_paths full paths for the fan chart; downsample path
    # length to <=400 points (aligned to monte_carlo_test's payload).
    n_keep = min(keep_paths, n_bootstrap, 30)
    kept = np.empty((n_keep, N))

    max_start = max(1, N - block + 1)
    for i in range(n_bootstrap):
        idx: list[int] = []
        while len(idx) < N:
            s = int(rng.integers(0, max_start))
            idx.extend(range(s, min(s + block, N)))
        idx = idx[:N]
        eq = start * np.cumprod(1.0 + rets[idx])
        finals[i] = eq[-1]
        if i < n_keep:
            kept[i] = eq

    orig_final = float(start * np.cumprod(1.0 + rets)[-1])

    result: Dict[str, Any] = {
        "n": N,
        "iters": n_bootstrap,
        "block": block,
        "prob_profit": round(float(np.mean(finals > start)), 4),
        "orig_final": round(orig_final, 2),
        "final_P5": round(float(np.percentile(finals, 5)), 2),
        "final_P50": round(float(np.percentile(finals, 50)), 2),
        "final_P95": round(float(np.percentile(finals, 95)), 2),
        "mean_ret": round(float(rets.mean()), 8),
    }

    # Fan-chart payload: downsample along steps (<=400) and along paths (<=30).
    # Schema mirrors monte_carlo_test.equity_paths exactly (shared builder) so the
    # frontend renders both with MonteCarloPathsChart unchanged.
    result["equity_paths"] = _equity_paths_payload(
        kept, start * np.cumprod(1.0 + rets), start,
    )
    return result


# ─── Monte Carlo Permutation Test ───


def monte_carlo_test(
    trades: List[TradeRecord],
    initial_capital: float,
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Shuffle trade PnL order to test path significance.

    Null hypothesis: the observed Sharpe / max-drawdown is no better than
    a random ordering of the same trades.

    Args:
        trades: Completed round-trip trades from backtest.
        initial_capital: Starting capital.
        n_simulations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with actual_sharpe, p_value_sharpe, actual_max_dd,
        p_value_max_dd, simulated_sharpes (percentiles).
    """
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, Integral) or n_simulations < 1:
        return {
            "error": f"n_simulations must be >= 1, got {n_simulations}",
            "p_value_sharpe": 1.0,
        }
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        return {"error": f"seed must be >= 0, got {seed}", "p_value_sharpe": 1.0}
    if len(trades) < 3:
        return {"error": "need at least 3 trades", "p_value_sharpe": 1.0}

    pnls = np.array([t.pnl for t in trades])
    actual = _path_metrics(pnls, initial_capital)

    rng = np.random.default_rng(seed)
    sharpe_count = 0
    dd_count = 0
    sim_sharpes = []
    # The full path matrix feeds the fan-chart payload; skip it for
    # pathological sizes so a huge run cannot balloon memory or the JSON.
    keep_paths = n_simulations * len(pnls) <= 2_000_000
    sim_equities = np.empty((n_simulations, len(pnls))) if keep_paths else None

    for i in range(n_simulations):
        shuffled = rng.permutation(pnls)
        if sim_equities is not None:
            sim_equities[i] = initial_capital + np.cumsum(shuffled)
        sim = _path_metrics(shuffled, initial_capital)
        sim_sharpes.append(sim["sharpe"])
        if sim["sharpe"] >= actual["sharpe"]:
            sharpe_count += 1
        if sim["max_dd"] >= actual["max_dd"]:  # less negative = "better"
            dd_count += 1

    sim_arr = np.array(sim_sharpes)
    result = {
        "actual_sharpe": round(actual["sharpe"], 4),
        "actual_max_dd": round(actual["max_dd"], 4),
        "p_value_sharpe": round(sharpe_count / n_simulations, 4),
        "p_value_max_dd": round(dd_count / n_simulations, 4),
        "simulated_sharpe_mean": round(float(sim_arr.mean()), 4),
        "simulated_sharpe_std": round(float(sim_arr.std()), 4),
        "simulated_sharpe_p5": round(float(np.percentile(sim_arr, 5)), 4),
        "simulated_sharpe_p95": round(float(np.percentile(sim_arr, 95)), 4),
        "n_simulations": n_simulations,
        "n_trades": len(trades),
        "sharpe_samples": [round(float(s), 4) for s in sim_sharpes],
    }
    if sim_equities is not None:
        result["equity_paths"] = _equity_paths_payload(
            sim_equities, initial_capital + np.cumsum(pnls), initial_capital,
        )
    return result


def _path_metrics(pnls: np.ndarray, initial_capital: float) -> Dict[str, float]:
    """Compute Sharpe and max drawdown from a PnL sequence."""
    equity = initial_capital + np.cumsum(pnls)
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0.0])
    std = returns.std()
    sharpe = float(returns.mean() / (std + 1e-10) * np.sqrt(252))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(dd.min())
    return {"sharpe": sharpe, "max_dd": max_dd}


# ─── Bootstrap Sharpe CI ───


def bootstrap_sharpe_ci(
    equity_curve: pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    bars_per_year: int = 252,
    seed: int = 42,
) -> Dict[str, Any]:
    """Resample daily returns to estimate Sharpe confidence interval.

    Args:
        equity_curve: Equity time series.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        bars_per_year: Annualisation factor.
        seed: Random seed.

    Returns:
        Dict with observed_sharpe, ci_lower, ci_upper, median_sharpe,
        prob_positive (fraction of samples with Sharpe > 0).
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or n_bootstrap < 1:
        return {"error": f"n_bootstrap must be >= 1, got {n_bootstrap}"}
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not math.isfinite(float(confidence))
        or not 0.0 < confidence < 1.0
    ):
        return {"error": f"confidence must be in (0, 1), got {confidence}"}
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        return {"error": f"seed must be >= 0, got {seed}"}

    returns = equity_curve.pct_change().dropna().values
    if len(returns) < 5:
        return {"error": "need at least 5 return observations"}

    observed = _sharpe(returns, bars_per_year)

    rng = np.random.default_rng(seed)
    boot_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=len(returns), replace=True)
        boot_sharpes.append(_sharpe(sample, bars_per_year))

    arr = np.array(boot_sharpes)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(arr, alpha * 100))
    upper = float(np.percentile(arr, (1 - alpha) * 100))
    prob_pos = float(np.mean(arr > 0))

    result = {
        "observed_sharpe": round(observed, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "median_sharpe": round(float(np.median(arr)), 4),
        "prob_positive": round(prob_pos, 4),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }
    if n_bootstrap <= 20_000:
        result["sharpe_samples"] = [round(float(s), 4) for s in boot_sharpes]
    return result


def _sharpe(returns: np.ndarray, bars_per_year: int = 252) -> float:
    std = returns.std()
    return float(returns.mean() / (std + 1e-10) * np.sqrt(bars_per_year))


# ─── Walk-Forward Analysis ───


def walk_forward_analysis(
    equity_curve: pd.Series,
    trades: List[TradeRecord],
    n_windows: int = 5,
    bars_per_year: int = 252,
) -> Dict[str, Any]:
    """Split backtest into sequential windows, check consistency.

    Each window is evaluated independently (returns normalised to window start).

    Args:
        equity_curve: Equity time series.
        trades: Completed trades.
        n_windows: Number of non-overlapping windows.
        bars_per_year: Annualisation factor.

    Returns:
        Dict with per_window stats, consistency metrics.
    """
    if isinstance(n_windows, bool) or not isinstance(n_windows, Integral) or n_windows < 1:
        return {"error": f"n_windows must be >= 1, got {n_windows}"}
    if len(equity_curve) < n_windows * 2:
        return {"error": f"need at least {n_windows * 2} bars for {n_windows} windows"}

    indices = equity_curve.index
    window_size = len(indices) // n_windows
    windows = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i < n_windows - 1 else len(indices)
        win_eq = equity_curve.iloc[start_idx:end_idx]
        win_start = indices[start_idx]
        win_end = indices[end_idx - 1]

        # Per-window trades
        win_trades = [t for t in trades if win_start <= t.entry_time <= win_end]

        # Per-window metrics
        ret = float(win_eq.iloc[-1] / win_eq.iloc[0] - 1) if win_eq.iloc[0] > 0 else 0.0
        win_returns = win_eq.pct_change().dropna().values
        sharpe = _sharpe(win_returns, bars_per_year) if len(win_returns) > 1 else 0.0

        peak = win_eq.cummax()
        dd = (win_eq - peak) / peak.replace(0, 1)
        max_dd = float(dd.min())

        win_pnls = [t.pnl for t in win_trades]
        win_rate = len([p for p in win_pnls if p > 0]) / len(win_pnls) if win_pnls else 0.0

        windows.append(
            {
                "window": i + 1,
                "start": str(win_start.date()) if hasattr(win_start, "date") else str(win_start),
                "end": str(win_end.date()) if hasattr(win_end, "date") else str(win_end),
                "return": round(ret, 6),
                "sharpe": round(sharpe, 4),
                "max_dd": round(max_dd, 6),
                "trades": len(win_trades),
                "win_rate": round(win_rate, 4),
            }
        )

    # Consistency metrics
    returns_list = [w["return"] for w in windows]
    sharpes_list = [w["sharpe"] for w in windows]
    profitable_windows = sum(1 for r in returns_list if r > 0)

    return {
        "n_windows": n_windows,
        "windows": windows,
        "profitable_windows": profitable_windows,
        "consistency_rate": round(profitable_windows / n_windows, 4),
        "return_mean": round(float(np.mean(returns_list)), 6),
        "return_std": round(float(np.std(returns_list)), 6),
        "sharpe_mean": round(float(np.mean(sharpes_list)), 4),
        "sharpe_std": round(float(np.std(sharpes_list)), 4),
    }


# ─── Runner integration ───


def run_validation(
    config: Dict[str, Any],
    equity_curve: pd.Series,
    trades: List[TradeRecord],
    initial_capital: float,
    bars_per_year: int = 252,
) -> Dict[str, Any]:
    """Run configured validation checks.

    Reads from config["validation"]:
      - monte_carlo: {n_simulations, seed}
      - bootstrap: {n_bootstrap, confidence, seed}
      - walk_forward: {n_windows}

    Args:
        config: Backtest config (must contain "validation" key).
        equity_curve: Equity time series.
        trades: Completed trades.
        initial_capital: Starting capital.
        bars_per_year: Annualisation factor.

    Returns:
        Dict keyed by validation type with results.
    """
    v_cfg = config.get("validation", {})
    results: Dict[str, Any] = {}

    if "monte_carlo" in v_cfg:
        mc_cfg = v_cfg["monte_carlo"] if isinstance(v_cfg["monte_carlo"], dict) else {}
        results["monte_carlo"] = monte_carlo_test(
            trades,
            initial_capital,
            n_simulations=mc_cfg.get("n_simulations", 1000),
            seed=mc_cfg.get("seed", 42),
        )

    if "bootstrap" in v_cfg:
        bs_cfg = v_cfg["bootstrap"] if isinstance(v_cfg["bootstrap"], dict) else {}
        results["bootstrap"] = bootstrap_sharpe_ci(
            equity_curve,
            bars_per_year=bars_per_year,
            n_bootstrap=bs_cfg.get("n_bootstrap", 1000),
            confidence=bs_cfg.get("confidence", 0.95),
            seed=bs_cfg.get("seed", 42),
        )

    if "walk_forward" in v_cfg:
        wf_cfg = v_cfg["walk_forward"] if isinstance(v_cfg["walk_forward"], dict) else {}
        results["walk_forward"] = walk_forward_analysis(
            equity_curve,
            trades,
            n_windows=wf_cfg.get("n_windows", 5),
            bars_per_year=bars_per_year,
        )

    if "block_bootstrap" in v_cfg:
        bb_cfg = v_cfg["block_bootstrap"] if isinstance(v_cfg["block_bootstrap"], dict) else {}
        unit = bb_cfg.get("unit", "day")
        if unit == "trade":
            # Per-margin returns — OVERSTATES variance under overlap/leverage
            # (review M4); flagged as the accepted approximation.
            rets = [
                t.pnl / t.entry_margin
                for t in trades
                if getattr(t, "entry_margin", 0) and t.entry_margin > 0
            ]
        else:
            rets = equity_curve.pct_change().dropna().values.tolist()
        bb = block_bootstrap(
            rets,
            n_bootstrap=bb_cfg.get("iters", 5000),
            block=bb_cfg.get("block", 10),
            start=bb_cfg.get("start", 1000.0),
            seed=bb_cfg.get("seed", 7),
        )
        if unit == "trade":
            bb["unit"] = "trade"
            bb["approximation"] = "per-margin returns; overlap/leverage not accounted"
        results["block_bootstrap"] = bb

    return results


# ─── Standalone CLI ───


def _load_equity(run_dir: Path) -> pd.Series:
    """Load equity curve from artifacts/equity.csv."""
    path = run_dir / "artifacts" / "equity.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    for col in ("equity", "nav", "value"):
        if col in df.columns:
            return df[col]
    raise ValueError(
        f"equity.csv must contain an equity/nav/value column; got {list(df.columns)}"
    )


def _load_trades(run_dir: Path) -> List[TradeRecord]:
    """Load trades from artifacts/trades.csv and convert to TradeRecord list."""
    path = run_dir / "artifacts" / "trades.csv"
    df = pd.read_csv(path)
    if df.empty:
        return []

    # trades.csv has entry+exit row pairs; extract exit rows (they have pnl != 0)
    trades = []
    exit_rows = df[df["pnl"] != 0].reset_index(drop=True)
    for _, row in exit_rows.iterrows():
        hold = pd.to_numeric(row.get("holding_days", 0), errors="coerce")
        holding_bars = 0 if pd.isna(hold) else int(hold)
        trades.append(
            TradeRecord(
                symbol=str(row.get("code", "")),
                direction=1 if row.get("side") == "sell" else -1,
                entry_price=0.0,
                exit_price=float(row.get("price", 0)),
                entry_time=pd.Timestamp(row.get("timestamp", "2000-01-01")),
                exit_time=pd.Timestamp(row.get("timestamp", "2000-01-01")),
                size=float(row.get("qty", 0)),
                leverage=1.0,
                pnl=float(row.get("pnl", 0)),
                pnl_pct=float(row.get("return_pct", 0)),
                exit_reason=str(row.get("reason", "signal")),
                holding_bars=holding_bars,
                commission=0.0,
            )
        )
    return trades


def _parse_run_dir(argv: List[str]) -> Path:
    """Validate CLI input and return a usable run directory path."""
    if len(argv) < 2:
        raise SystemExit("Usage: python -m backtest.validation <run_dir>")

    raw_run_dir = argv[1]
    if not raw_run_dir.strip():
        raise SystemExit("run_dir must be a non-empty path")
    if "\0" in raw_run_dir:
        raise SystemExit("Invalid run_dir path: embedded NUL byte")

    try:
        run_dir = Path(raw_run_dir).expanduser()
        exists = run_dir.exists()
        is_dir = run_dir.is_dir() if exists else False
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Invalid run_dir path: {exc}") from exc

    if not exists:
        raise SystemExit(f"run_dir does not exist: {run_dir}")
    if not is_dir:
        raise SystemExit(f"run_dir is not a directory: {run_dir}")
    return run_dir


def _json_safe(value: Any) -> Any:
    """Return a JSON-strict copy of validation results."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_validation_json(path: Path, results: Dict[str, Any]) -> Dict[str, Any]:
    """Write validation results to ``path`` as strict, RFC-8259 JSON.

    A validation metric can be non-finite (e.g. a Sharpe computed from a path
    whose equity touches zero), and ``json.dumps`` emits bare ``NaN`` /
    ``Infinity`` tokens for those by default (``allow_nan=True``) — tokens that
    strict parsers reject. Sanitize with :func:`_json_safe` (non-finite → null)
    and serialize with ``allow_nan=False`` so every writer of
    ``artifacts/validation.json`` produces the same valid JSON. Returns the
    sanitized payload that was written.
    """
    safe_results = _json_safe(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return safe_results


def main(run_dir: Path) -> Dict[str, Any]:
    """Run all three validations on existing backtest artifacts.

    Reads equity.csv, trades.csv, and config.json from run_dir.

    Args:
        run_dir: Directory with artifacts/ subdirectory.

    Returns:
        Validation results dict.
    """
    # Load config for initial_cash
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}
    initial_capital = config.get("initial_cash", 1_000_000)

    equity = _load_equity(run_dir)
    trades = _load_trades(run_dir)

    results = {
        "monte_carlo": monte_carlo_test(trades, initial_capital),
        "bootstrap": bootstrap_sharpe_ci(equity),
        "walk_forward": walk_forward_analysis(equity, trades),
    }

    out = run_dir / "artifacts" / "validation.json"
    safe_results = write_validation_json(out, results)

    print(json.dumps(safe_results, indent=2, allow_nan=False))
    return safe_results


if __name__ == "__main__":
    import sys

    main(_parse_run_dir(sys.argv))
