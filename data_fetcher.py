"""Fetch stablecoin supply, DEX volume, and token prices from DefiLlama."""

import json
import os
import time
from datetime import datetime

import pandas as pd
import requests

from config import CHAINS, Chain

CACHE_DIR = "cache"
STABLECOIN_API = "https://stablecoins.llama.fi"
LLAMA_API = "https://api.llama.fi"
COINS_API = "https://coins.llama.fi"

REQUEST_DELAY = 0.4  # seconds between API calls


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  HTTP {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  Request error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(wait)
    return None


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_cache(key: str) -> list | None:
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(key: str, data: list) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


def _records_to_df(records: list, date_col: str = "date") -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Remove duplicate dates, keep last
    df = df[~df.index.duplicated(keep="last")]
    return df


# ── Stablecoin supply per chain ─────────────────────────────────────────────

def fetch_stablecoin_supply(chain: Chain) -> pd.DataFrame:
    """Daily total stablecoin circulating supply (USD) on a chain."""
    cache_key = f"supply_{chain.defillama_id}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return _records_to_df(cached)

    url = f"{STABLECOIN_API}/stablecoincharts/{chain.defillama_id}"
    data = _get(url)
    if not data:
        return pd.DataFrame()

    records = []
    for entry in data:
        ts = int(entry.get("date", 0))
        supply = 0.0
        circ = entry.get("totalCirculatingUSD", {})
        if isinstance(circ, dict):
            supply = circ.get("peggedUSD", 0) or 0
        elif isinstance(circ, (int, float)):
            supply = circ
        records.append({
            "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
            "supply": float(supply),
        })

    _save_cache(cache_key, records)
    return _records_to_df(records)


# ── DEX volume per chain ────────────────────────────────────────────────────

def fetch_dex_volume(chain: Chain) -> pd.DataFrame:
    """Daily DEX trading volume (USD) on a chain."""
    cache_key = f"dex_volume_{chain.defillama_id}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return _records_to_df(cached)

    url = (
        f"{LLAMA_API}/overview/dexs/{chain.defillama_id}"
        "?excludeTotalDataChart=false"
        "&excludeTotalDataChartBreakdown=true"
        "&dataType=dailyVolume"
    )
    data = _get(url)
    if not data or "totalDataChart" not in data:
        return pd.DataFrame()

    records = []
    for entry in data["totalDataChart"]:
        if isinstance(entry, list) and len(entry) >= 2:
            ts, vol = int(entry[0]), float(entry[1])
        elif isinstance(entry, dict):
            ts = int(entry.get("date", entry.get("timestamp", 0)))
            vol = float(entry.get("volume", entry.get("dailyVolume", 0)))
        else:
            continue
        records.append({
            "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
            "volume": vol,
        })

    _save_cache(cache_key, records)
    return _records_to_df(records)


# ── Token prices ─────────────────────────────────────────────────────────────

def fetch_token_price(chain: Chain) -> pd.DataFrame:
    """Daily price of chain's native token (USD)."""
    cache_key = f"price_{chain.coingecko_id}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return _records_to_df(cached)

    records = _fetch_price_defillama(chain)
    if not records:
        records = _fetch_price_coingecko(chain)
    if not records:
        return pd.DataFrame()

    _save_cache(cache_key, records)
    return _records_to_df(records)


def _fetch_price_defillama(chain: Chain) -> list:
    coin_id = f"coingecko:{chain.coingecko_id}"
    url = f"{COINS_API}/chart/{coin_id}?period=1d&span=5000"
    data = _get(url)
    if not data or "coins" not in data:
        return []

    coin_data = data["coins"].get(coin_id, {})
    prices = coin_data.get("prices", [])
    if not prices:
        return []

    records = []
    for item in prices:
        if isinstance(item, dict):
            ts = int(item.get("timestamp", 0))
            price = float(item.get("price", 0))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            ts, price = int(item[0]), float(item[1])
        else:
            continue
        records.append({
            "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
            "price": price,
        })
    return records


def _fetch_price_coingecko(chain: Chain) -> list:
    """Fallback: CoinGecko free API."""
    url = (
        f"https://api.coingecko.com/api/v3/coins/{chain.coingecko_id}"
        "/market_chart?vs_currency=usd&days=max&interval=daily"
    )
    data = _get(url)
    if not data or "prices" not in data:
        return []

    records = []
    for ts_ms, price in data["prices"]:
        records.append({
            "date": datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d"),
            "price": float(price),
        })
    return records


# ── Fetch everything ─────────────────────────────────────────────────────────

def fetch_all_data(chains: list[Chain] | None = None) -> dict:
    """
    Fetch supply, volume, price for all chains.
    Returns: {chain_name: {"supply": df, "volume": df, "price": df, "chain": Chain}}
    """
    if chains is None:
        chains = CHAINS

    result = {}
    total = len(chains)

    for i, chain in enumerate(chains, 1):
        print(f"[{i}/{total}] Fetching {chain.name} ({chain.symbol})...")

        supply = fetch_stablecoin_supply(chain)
        time.sleep(REQUEST_DELAY)

        volume = fetch_dex_volume(chain)
        time.sleep(REQUEST_DELAY)

        price = fetch_token_price(chain)
        time.sleep(REQUEST_DELAY)

        result[chain.name] = {
            "supply": supply,
            "volume": volume,
            "price": price,
            "chain": chain,
        }

        s_days = len(supply)
        v_days = len(volume)
        p_days = len(price)
        print(f"  -> supply: {s_days}d, volume: {v_days}d, price: {p_days}d")

        if s_days == 0 or v_days == 0 or p_days == 0:
            print(f"  !! Missing data for {chain.name}, will be excluded")

    return result
