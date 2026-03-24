"""Generate charts for the velocity L/S strategy backtest."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from strategy import BacktestResult

OUTPUT_DIR = "output"


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_cumulative_pnl(result: BacktestResult, filename: str | None = None):
    """Plot cumulative returns: strategy, long leg, short leg."""
    _ensure_dir()
    if filename is None:
        filename = f"cumulative_pnl_{result.name}.png"

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(result.cumulative_strategy.index, result.cumulative_strategy.values,
            label="L/S Strategy", linewidth=2, color="#2196F3")
    ax.plot(result.cumulative_long.index, result.cumulative_long.values,
            label="Long (high velocity)", linewidth=1.2, color="#4CAF50", alpha=0.8)
    ax.plot(result.cumulative_short.index, result.cumulative_short.values,
            label="Short (low velocity)", linewidth=1.2, color="#F44336", alpha=0.8)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Stablecoin Velocity L/S Strategy — {result.name}", fontsize=14)
    ax.set_ylabel("Cumulative Return (1 = start)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    m = result.metrics
    textstr = (
        f"Total: {m['total_return']:+.1%}  |  "
        f"Sharpe: {m['sharpe']:.2f}  |  "
        f"MaxDD: {m['max_drawdown']:.1%}  |  "
        f"WinRate: {m['weekly_win_rate']:.0%}"
    )
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=10,
            ha="center", style="italic", color="gray")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_velocity_heatmap(result: BacktestResult, filename: str | None = None):
    """Heatmap of velocity ranks over time (monthly sampled)."""
    _ensure_dir()
    if filename is None:
        filename = f"velocity_heatmap_{result.name}.png"

    vel = result.velocity_df[result.chains_used].copy()
    if vel.empty:
        return

    # Resample to monthly for readability
    vel_monthly = vel.resample("MS").mean()
    ranks = vel_monthly.rank(axis=1, pct=True)

    fig, ax = plt.subplots(figsize=(16, max(4, len(result.chains_used) * 0.5)))

    im = ax.imshow(
        ranks.T.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        interpolation="nearest",
    )

    # Labels
    ax.set_yticks(range(len(ranks.columns)))
    ax.set_yticklabels(ranks.columns, fontsize=9)

    # X-axis: show every 3rd month
    n_dates = len(ranks.index)
    step = max(1, n_dates // 15)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in ranks.index[::step]],
        rotation=45, ha="right", fontsize=8,
    )

    ax.set_title(f"Velocity Rank Heatmap — {result.name}", fontsize=13)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Velocity Percentile Rank")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_velocity_timeseries(result: BacktestResult, filename: str | None = None):
    """Plot raw velocity for each chain over time."""
    _ensure_dir()
    if filename is None:
        filename = f"velocity_ts_{result.name}.png"

    vel = result.velocity_df[result.chains_used].copy()
    if vel.empty:
        return

    # Resample to weekly for cleaner lines
    vel_w = vel.resample("W").mean()

    fig, ax = plt.subplots(figsize=(14, 7))
    for col in vel_w.columns:
        ax.plot(vel_w.index, vel_w[col], label=col, linewidth=1.2, alpha=0.8)

    ax.set_title(f"Stablecoin Velocity Over Time — {result.name}", fontsize=14)
    ax.set_ylabel("Velocity (DEX Volume / Stablecoin Supply)")
    ax.set_xlabel("Date")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_all(result: BacktestResult):
    """Generate all plots for a backtest result."""
    plot_cumulative_pnl(result)
    plot_velocity_heatmap(result)
    plot_velocity_timeseries(result)


def plot_summary_comparison(results: dict[str, BacktestResult]):
    """Bar chart comparing key metrics across hypothesis groups."""
    _ensure_dir()

    names = []
    total_rets, sharpes, drawdowns = [], [], []

    for name, r in results.items():
        if r is None or not r.metrics:
            continue
        names.append(name)
        total_rets.append(r.metrics["total_return"] * 100)
        sharpes.append(r.metrics["sharpe"])
        drawdowns.append(r.metrics["max_drawdown"] * 100)

    if not names:
        return

    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Total return
    bars1 = axes[0].bar(x, total_rets, width, color="#2196F3")
    axes[0].set_title("Total Return (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].bar_label(bars1, fmt="%.1f%%", fontsize=8)

    # Sharpe
    bars2 = axes[1].bar(x, sharpes, width, color="#4CAF50")
    axes[1].set_title("Sharpe Ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].bar_label(bars2, fmt="%.2f", fontsize=8)

    # Max Drawdown
    bars3 = axes[2].bar(x, drawdowns, width, color="#F44336")
    axes[2].set_title("Max Drawdown (%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[2].bar_label(bars3, fmt="%.1f%%", fontsize=8)

    fig.suptitle("Velocity L/S Strategy — Hypothesis Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
