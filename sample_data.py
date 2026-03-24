"""
Generate realistic sample data for testing when DefiLlama API is unavailable.

Uses plausible ranges for stablecoin supply, DEX volume, and token prices
based on historical data patterns for each chain. Introduces correlations
between velocity and subsequent price performance to test strategy detection.
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import CHAINS, Chain

CACHE_DIR = "cache"
SEED = 42


def _ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)


# Realistic parameters per chain (approximate historical ranges)
CHAIN_PARAMS = {
    "Ethereum":   {"supply_base": 40e9,  "vol_base": 2e9,   "price_base": 2000, "vol_mult": 1.0},
    "BSC":        {"supply_base": 5e9,   "vol_base": 800e6, "price_base": 300,  "vol_mult": 0.8},
    "Avalanche":  {"supply_base": 1.5e9, "vol_base": 200e6, "price_base": 25,   "vol_mult": 1.2},
    "Polygon":    {"supply_base": 1.2e9, "vol_base": 300e6, "price_base": 0.8,  "vol_mult": 1.1},
    "Fantom":     {"supply_base": 200e6, "vol_base": 50e6,  "price_base": 0.4,  "vol_mult": 1.5},
    "Cronos":     {"supply_base": 300e6, "vol_base": 30e6,  "price_base": 0.08, "vol_mult": 0.7},
    "Solana":     {"supply_base": 3e9,   "vol_base": 1.5e9, "price_base": 100,  "vol_mult": 1.8},
    "Tron":       {"supply_base": 50e9,  "vol_base": 500e6, "price_base": 0.10, "vol_mult": 0.3},
    "Near":       {"supply_base": 200e6, "vol_base": 30e6,  "price_base": 4,    "vol_mult": 1.0},
    "Sui":        {"supply_base": 300e6, "vol_base": 200e6, "price_base": 1.5,  "vol_mult": 2.0},
    "Aptos":      {"supply_base": 150e6, "vol_base": 50e6,  "price_base": 8,    "vol_mult": 1.3},
    "Arbitrum":   {"supply_base": 3e9,   "vol_base": 1e9,   "price_base": 1.2,  "vol_mult": 1.4},
    "Optimism":   {"supply_base": 800e6, "vol_base": 300e6, "price_base": 2.0,  "vol_mult": 1.2},
    "zkSync Era": {"supply_base": 200e6, "vol_base": 50e6,  "price_base": 0.15, "vol_mult": 1.6},
    "Mantle":     {"supply_base": 400e6, "vol_base": 80e6,  "price_base": 0.6,  "vol_mult": 1.0},
    "Starknet":   {"supply_base": 100e6, "vol_base": 20e6,  "price_base": 0.5,  "vol_mult": 1.3},
}


def generate_sample_data(
    start_date: str = "2022-06-01",
    end_date: str = "2025-12-31",
    velocity_alpha: float = 0.02,
):
    """
    Generate sample data with a built-in velocity->return signal.

    velocity_alpha: strength of the velocity->price signal.
      Higher = velocity is more predictive of returns.
      Set to 0 for pure noise (null hypothesis).
    """
    _ensure_cache()
    rng = np.random.RandomState(SEED)

    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)

    # Market-wide factor (crypto beta)
    market_drift = 0.0001  # slight upward drift
    market_vol = 0.025
    market_returns = rng.normal(market_drift, market_vol, n_days)
    # Add regime changes
    bull_periods = [
        (200, 350),   # late 2022 -> early 2023 recovery
        (550, 750),   # mid 2024 bull
        (900, 1050),  # 2025 bull
    ]
    for start, end in bull_periods:
        if end <= n_days:
            market_returns[start:end] += 0.002

    bear_periods = [
        (0, 150),     # mid-2022 bear
        (400, 500),   # correction
    ]
    for start, end in bear_periods:
        if end <= n_days:
            market_returns[start:end] -= 0.003

    for chain in CHAINS:
        params = CHAIN_PARAMS.get(chain.name)
        if params is None:
            continue

        print(f"  Generating data for {chain.name}...")

        # ── Supply: random walk with trend ───────────────────────────────
        supply_growth = rng.normal(0.0003, 0.01, n_days)  # slight growth
        supply_log = np.log(params["supply_base"]) + np.cumsum(supply_growth)
        supply = np.exp(supply_log)
        supply = np.maximum(supply, 1e6)  # floor

        # ── Volume: mean-reverting with spikes ───────────────────────────
        vol_base = params["vol_base"]
        vol_mult = params["vol_mult"]
        log_vol = np.zeros(n_days)
        log_vol[0] = np.log(vol_base)
        for i in range(1, n_days):
            mean_revert = 0.05 * (np.log(vol_base) - log_vol[i - 1])
            shock = rng.normal(0, 0.15)
            # Occasional volume spikes
            if rng.random() < 0.03:
                shock += rng.exponential(0.5)
            log_vol[i] = log_vol[i - 1] + mean_revert + shock

        volume = np.exp(log_vol) * vol_mult
        volume = np.maximum(volume, 0)

        # ── Velocity (derived) ───────────────────────────────────────────
        velocity = volume / np.maximum(supply, 1)

        # ── Price: market beta + idiosyncratic + velocity signal ─────────
        beta = 0.6 + rng.random() * 0.8  # 0.6 to 1.4
        idio_vol = 0.015 + rng.random() * 0.025  # 1.5-4% daily idio vol
        idio_returns = rng.normal(0, idio_vol, n_days)

        # Velocity signal: rolling velocity predicts forward returns
        vel_rolling = pd.Series(velocity).rolling(30, min_periods=10).mean().values
        vel_signal = np.zeros(n_days)
        for i in range(30, n_days):
            if not np.isnan(vel_rolling[i]):
                lookback = vel_rolling[max(0, i - 90):i]
                std = np.nanstd(lookback)
                if std > 1e-10:
                    z = (vel_rolling[i] - np.nanmean(lookback)) / std
                    vel_signal[i] = np.clip(z, -2, 2)  # cap z-scores

        price_returns = beta * market_returns + idio_returns + velocity_alpha * vel_signal
        # Clip extreme daily returns to keep prices realistic
        price_returns = np.clip(price_returns, -0.15, 0.15)
        price_log = np.log(params["price_base"]) + np.cumsum(price_returns)
        price = np.exp(price_log)

        # ── Save to cache ────────────────────────────────────────────────
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        supply_records = [{"date": d, "supply": float(s)} for d, s in zip(date_strs, supply)]
        volume_records = [{"date": d, "volume": float(v)} for d, s in zip(date_strs, volume) for v in [volume[date_strs.index(d)]]]
        price_records = [{"date": d, "price": float(p)} for d, p in zip(date_strs, price)]

        # Simpler record building
        volume_records = [{"date": d, "volume": float(volume[i])} for i, d in enumerate(date_strs)]

        _save(f"supply_{chain.defillama_id}", supply_records)
        _save(f"dex_volume_{chain.defillama_id}", volume_records)
        _save(f"price_{chain.coingecko_id}", price_records)

    print(f"\n  Sample data generated: {start_date} -> {end_date} ({n_days} days)")


def _save(key: str, data: list):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    print("Generating sample data...\n")
    generate_sample_data()
    print("\nDone! Data saved to cache/")
