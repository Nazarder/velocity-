"""
Stablecoin Velocity Long/Short Strategy.

Velocity = rolling DEX volume / rolling stablecoin supply
Signal:  rank chains by velocity each rebalance period
Trade:   long top-tercile native tokens, short bottom-tercile
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    MIN_CHAINS,
    MIN_SUPPLY_USD,
    REBALANCE_FREQ,
    TOP_QUANTILE,
    VELOCITY_WINDOW,
    Chain,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Build aligned data ──────────────────────────────────────────────────────

def build_panel(all_data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align supply, volume, and price data into three DataFrames
    with common daily index and chain columns.
    """
    supply_frames, volume_frames, price_frames = {}, {}, {}

    for name, d in all_data.items():
        s, v, p = d["supply"], d["volume"], d["price"]
        if s.empty or v.empty or p.empty:
            continue
        supply_frames[name] = s["supply"] if "supply" in s.columns else s.iloc[:, 0]
        volume_frames[name] = v["volume"] if "volume" in v.columns else v.iloc[:, 0]
        price_frames[name] = p["price"] if "price" in p.columns else p.iloc[:, 0]

    if not supply_frames:
        raise ValueError("No chains with complete data")

    supply = pd.DataFrame(supply_frames)
    volume = pd.DataFrame(volume_frames)
    price = pd.DataFrame(price_frames)

    # Align to common date range where we have all three
    common_idx = supply.index.intersection(volume.index).intersection(price.index)
    common_idx = common_idx.sort_values()

    supply = supply.reindex(common_idx).ffill().bfill()
    volume = volume.reindex(common_idx).fillna(0)
    price = price.reindex(common_idx).ffill().bfill()

    return supply, volume, price


# ── Velocity calculation ────────────────────────────────────────────────────

def compute_velocity(
    supply: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = VELOCITY_WINDOW,
    min_supply: float = MIN_SUPPLY_USD,
) -> pd.DataFrame:
    """
    Velocity = rolling_sum(volume, window) / rolling_mean(supply, window)

    Chains with supply below min_supply are masked as NaN.
    """
    roll_vol = volume.rolling(window, min_periods=max(1, window // 2)).sum()
    roll_sup = supply.rolling(window, min_periods=max(1, window // 2)).mean()

    # Mask low-supply chains
    roll_sup = roll_sup.where(roll_sup >= min_supply)

    velocity = roll_vol / roll_sup
    velocity = velocity.replace([np.inf, -np.inf], np.nan)
    return velocity


# ── Ranking & basket construction ───────────────────────────────────────────

def rank_chains(velocity: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (0 = lowest velocity, 1 = highest)."""
    return velocity.rank(axis=1, pct=True)


def assign_baskets(
    ranks: pd.DataFrame,
    top_q: float = TOP_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two boolean DataFrames:
      long_mask  = True for chains in the top quantile
      short_mask = True for chains in the bottom quantile
    """
    long_mask = ranks >= (1 - top_q)
    short_mask = ranks <= top_q
    return long_mask, short_mask


# ── Backtest engine ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    name: str
    chains_used: list[str]
    strategy_returns: pd.Series      # daily L/S returns
    long_returns: pd.Series          # daily long-leg returns
    short_returns: pd.Series         # daily short-leg returns
    cumulative_strategy: pd.Series
    cumulative_long: pd.Series
    cumulative_short: pd.Series
    velocity_df: pd.DataFrame        # for analysis
    metrics: dict


def backtest(
    supply: pd.DataFrame,
    volume: pd.DataFrame,
    price: pd.DataFrame,
    group_name: str = "all",
    chain_filter: list[str] | None = None,
    velocity_window: int = VELOCITY_WINDOW,
    rebalance_freq: int = REBALANCE_FREQ,
    top_quantile: float = TOP_QUANTILE,
    min_chains: int = MIN_CHAINS,
) -> BacktestResult | None:
    """
    Run the velocity L/S backtest on a subset of chains.

    Returns None if not enough chains have data.
    """
    # Filter to requested chains
    if chain_filter is not None:
        cols = [c for c in chain_filter if c in supply.columns]
    else:
        cols = list(supply.columns)

    if len(cols) < min_chains:
        print(f"  Skipping '{group_name}': only {len(cols)} chains (need {min_chains})")
        return None

    sup = supply[cols].copy()
    vol = volume[cols].copy()
    prc = price[cols].copy()

    # Compute velocity
    vel = compute_velocity(sup, vol, window=velocity_window)

    # Daily returns of native tokens
    returns = prc.pct_change().fillna(0)

    # Build signal & portfolio
    ranks = rank_chains(vel)
    long_mask, short_mask = assign_baskets(ranks, top_q=top_quantile)

    # Rebalance logic: hold positions for rebalance_freq days
    rebalance_dates = vel.index[velocity_window::rebalance_freq]

    active_long = pd.DataFrame(False, index=vel.index, columns=cols)
    active_short = pd.DataFrame(False, index=vel.index, columns=cols)

    for rd in rebalance_dates:
        # Get positions at rebalance date
        l_row = long_mask.loc[rd]
        s_row = short_mask.loc[rd]

        # Valid chains (have velocity data)
        valid = vel.loc[rd].notna()
        l_row = l_row & valid
        s_row = s_row & valid

        # Find next rebalance date
        next_dates = rebalance_dates[rebalance_dates > rd]
        end = next_dates[0] if len(next_dates) > 0 else vel.index[-1]

        # Apply positions for the holding period
        mask = (vel.index >= rd) & (vel.index < end)
        active_long.loc[mask] = l_row.values
        active_short.loc[mask] = s_row.values

    # Equal-weight returns within each leg
    n_long = active_long.sum(axis=1).replace(0, np.nan)
    n_short = active_short.sum(axis=1).replace(0, np.nan)

    long_ret = (returns * active_long).sum(axis=1) / n_long
    short_ret = (returns * active_short).sum(axis=1) / n_short

    long_ret = long_ret.fillna(0)
    short_ret = short_ret.fillna(0)

    # L/S strategy: long high-velocity, short low-velocity
    strategy_ret = long_ret - short_ret

    # Trim to period where we actually have positions
    start_idx = velocity_window
    strategy_ret = strategy_ret.iloc[start_idx:]
    long_ret = long_ret.iloc[start_idx:]
    short_ret = short_ret.iloc[start_idx:]

    # Cumulative
    cum_strategy = (1 + strategy_ret).cumprod()
    cum_long = (1 + long_ret).cumprod()
    cum_short = (1 + short_ret).cumprod()

    # Metrics
    metrics = compute_metrics(strategy_ret, long_ret, short_ret)

    return BacktestResult(
        name=group_name,
        chains_used=cols,
        strategy_returns=strategy_ret,
        long_returns=long_ret,
        short_returns=short_ret,
        cumulative_strategy=cum_strategy,
        cumulative_long=cum_long,
        cumulative_short=cum_short,
        velocity_df=vel,
        metrics=metrics,
    )


# ── Performance metrics ─────────────────────────────────────────────────────

def compute_metrics(
    strategy: pd.Series,
    long_leg: pd.Series,
    short_leg: pd.Series,
) -> dict:
    """Compute key performance metrics."""
    days = len(strategy)
    if days == 0:
        return {}

    ann_factor = 365

    def _ann_return(s: pd.Series) -> float:
        total = (1 + s).prod() - 1
        n_years = len(s) / ann_factor
        if n_years <= 0:
            return 0.0
        return (1 + total) ** (1 / n_years) - 1

    def _sharpe(s: pd.Series) -> float:
        if s.std() == 0:
            return 0.0
        return s.mean() / s.std() * np.sqrt(ann_factor)

    def _max_drawdown(s: pd.Series) -> float:
        cum = (1 + s).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        return dd.min()

    def _win_rate(s: pd.Series) -> float:
        # Weekly win rate
        weekly = s.resample("W").sum()
        if len(weekly) == 0:
            return 0.0
        return (weekly > 0).mean()

    total_ret = (1 + strategy).prod() - 1
    long_total = (1 + long_leg).prod() - 1
    short_total = (1 + short_leg).prod() - 1

    return {
        "period_start": str(strategy.index[0].date()),
        "period_end": str(strategy.index[-1].date()),
        "days": days,
        "total_return": total_ret,
        "ann_return": _ann_return(strategy),
        "sharpe": _sharpe(strategy),
        "max_drawdown": _max_drawdown(strategy),
        "weekly_win_rate": _win_rate(strategy),
        "long_total_return": long_total,
        "short_total_return": short_total,
        "long_ann_return": _ann_return(long_leg),
        "short_ann_return": _ann_return(short_leg),
        "long_sharpe": _sharpe(long_leg),
        "short_sharpe": _sharpe(short_leg),
        "daily_vol": strategy.std() * np.sqrt(ann_factor),
    }


def format_metrics(result: BacktestResult) -> str:
    """Pretty-print backtest metrics."""
    m = result.metrics
    if not m:
        return f"  {result.name}: No data\n"

    lines = [
        f"{'=' * 60}",
        f"  Hypothesis: {result.name}",
        f"  Chains: {', '.join(result.chains_used)}",
        f"  Period: {m['period_start']} -> {m['period_end']} ({m['days']} days)",
        f"{'─' * 60}",
        f"  L/S Strategy:",
        f"    Total Return:     {m['total_return']:>+8.1%}",
        f"    Ann. Return:      {m['ann_return']:>+8.1%}",
        f"    Sharpe Ratio:     {m['sharpe']:>8.2f}",
        f"    Max Drawdown:     {m['max_drawdown']:>8.1%}",
        f"    Weekly Win Rate:  {m['weekly_win_rate']:>8.1%}",
        f"    Ann. Volatility:  {m['daily_vol']:>8.1%}",
        f"{'─' * 60}",
        f"  Long Leg (high velocity):",
        f"    Total Return:     {m['long_total_return']:>+8.1%}",
        f"    Ann. Return:      {m['long_ann_return']:>+8.1%}",
        f"    Sharpe:           {m['long_sharpe']:>8.2f}",
        f"{'─' * 60}",
        f"  Short Leg (low velocity):",
        f"    Total Return:     {m['short_total_return']:>+8.1%}",
        f"    Ann. Return:      {m['short_ann_return']:>+8.1%}",
        f"    Sharpe:           {m['short_sharpe']:>8.2f}",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)
