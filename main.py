#!/usr/bin/env python3
"""
Stablecoin Velocity Long/Short Strategy
========================================

Hypothesis: blockchains with higher stablecoin velocity
(on-chain transfer volume / stablecoin supply) should see their native tokens
outperform those with lower velocity.

Strategy: long top-tercile velocity chains, short bottom-tercile.
Tested across multiple groupings (L1, L2, EVM, non-EVM, all).

Data sources:
    - Stablecoin transfer volume: Dune Analytics (query 6899878)
    - Stablecoin supply: DefiLlama
    - Token prices: DefiLlama / CoinGecko

Usage:
    python main.py                          # Fetch live data (Dune + DefiLlama)
    python main.py --csv data/allium.csv    # Supply+volume from Allium CSV, prices from DefiLlama
    python main.py --sample                 # Use generated sample data (for testing)
"""

import argparse
import os
import sys

from config import CHAINS, GROUPS, MIN_CHAINS
from data_fetcher import fetch_all_data, fetch_prices_only, load_allium_csv
from strategy import BacktestResult, backtest, build_panel, format_metrics
from visualize import plot_all, plot_summary_comparison, plot_velocity_by_chain


def has_cached_data() -> bool:
    """Check if cache directory has any data files."""
    cache_dir = "cache"
    if not os.path.isdir(cache_dir):
        return False
    return any(f.endswith(".json") for f in os.listdir(cache_dir))


def generate_sample_if_needed():
    """Generate sample data if no cache exists or API is unavailable."""
    from sample_data import generate_sample_data
    print("Generating sample data for testing...\n")
    generate_sample_data()
    print()


def main():
    parser = argparse.ArgumentParser(description="Stablecoin Velocity L/S Strategy")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to Allium CSV file (supply+volume from CSV, prices from DefiLlama)")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample data instead of live API")
    parser.add_argument("--alpha", type=float, default=0.005,
                        help="Velocity signal strength in sample data (default: 0.02)")
    parser.add_argument("--volume-source", choices=["dex", "transfer"], default="transfer",
                        help="Volume source: 'dex' (DefiLlama DEX) or 'transfer' (Dune stablecoin transfers)")
    args = parser.parse_args()

    if args.csv:
        vol_label = f"Allium CSV ({args.csv})"
    elif args.volume_source == "transfer":
        vol_label = "Stablecoin Transfer Volume (Dune)"
    else:
        vol_label = "DEX Volume (DefiLlama)"

    print("=" * 60)
    print("  Stablecoin Velocity L/S Strategy Backtest")
    print(f"  Volume source: {vol_label}")
    print("=" * 60)
    print()

    # ── Step 1: Get data ─────────────────────────────────────────────────
    if args.csv:
        print(f"Step 1: Loading supply + volume from CSV: {args.csv}\n")
        all_data = load_allium_csv(args.csv)
        print(f"\nStep 1b: Fetching token prices from DefiLlama...\n")
        all_data = fetch_prices_only(all_data)
    elif args.sample:
        from sample_data import generate_sample_data
        print(f"Step 1: Generating sample data (alpha={args.alpha})...\n")
        generate_sample_data(velocity_alpha=args.alpha)
        print()
        print("Step 1b: Loading from cache...\n")
        all_data = fetch_all_data(volume_source="dex")  # sample data uses DEX volume cache
    else:
        print(f"Step 1: Fetching data (volume: {args.volume_source})...\n")
        all_data = fetch_all_data(volume_source=args.volume_source)

        # Check if we got any data
        has_data = any(
            not d["supply"].empty and not d["volume"].empty and not d["price"].empty
            for d in all_data.values()
        )
        if not has_data:
            print("\n  !! No live data available (API unreachable).")
            print("  !! Falling back to sample data...\n")
            generate_sample_if_needed()
            all_data = fetch_all_data(volume_source="dex")

    # ── Step 2: Build aligned panel ──────────────────────────────────────
    print("\nStep 2: Building aligned data panel...\n")
    try:
        supply, volume, price = build_panel(all_data)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    available_chains = list(supply.columns)
    print(f"  Chains with complete data: {len(available_chains)}")
    print(f"  {', '.join(available_chains)}")
    print(f"  Date range: {supply.index[0].date()} -> {supply.index[-1].date()}")
    print(f"  Total days: {len(supply)}")
    print()

    # ── Step 2b: Velocity by chain chart (Allium-style) ─────────────────
    print("Step 2b: Generating velocity-by-chain chart...\n")
    plot_velocity_by_chain(supply, volume)
    print()

    # ── Step 3: Run backtests for each hypothesis ────────────────────────
    print("Step 3: Running backtests...\n")
    results: dict[str, BacktestResult | None] = {}

    for group_name, filter_fn in GROUPS.items():
        group_chains = [
            c.name for c in CHAINS
            if filter_fn(c) and c.name in available_chains
        ]

        if len(group_chains) < MIN_CHAINS:
            print(f"  [{group_name}] Skipped: {len(group_chains)} chains < {MIN_CHAINS} min")
            results[group_name] = None
            continue

        print(f"  [{group_name}] Running with {len(group_chains)} chains: {', '.join(group_chains)}")
        result = backtest(
            supply, volume, price,
            group_name=group_name,
            chain_filter=group_chains,
        )
        results[group_name] = result

    # ── Step 4: Print results ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60 + "\n")

    for group_name, result in results.items():
        if result is not None:
            print(format_metrics(result))
            print()

    # ── Step 5: Generate charts ──────────────────────────────────────────
    print("\nStep 5: Generating charts...\n")

    for group_name, result in results.items():
        if result is not None:
            print(f"  Charts for '{group_name}':")
            plot_all(result)
            print()

    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        plot_summary_comparison(valid_results)

    print("\nDone! Check the 'output/' directory for charts.")


if __name__ == "__main__":
    main()
