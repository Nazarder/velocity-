"""Fetch stablecoin supply, on-chain transfer volume (Dune), and token prices."""

import json
import os
import time
from datetime import datetime

import pandas as pd
import requests

from config import CHAINS, CG_API_KEY, DUNE_API_KEY, DUNE_QUERY_ID, Chain

CACHE_DIR = "cache"
STABLECOIN_API = "https://stablecoins.llama.fi"
LLAMA_API = "https://api.llama.fi"
COINS_API = "https://coins.llama.fi"
DUNE_API = "https://api.dune.com/api/v1"

REQUEST_DELAY = 0.4  # seconds between API calls
DUNE_POLL_INTERVAL = 2  # seconds between status polls
DUNE_MAX_POLL = 300  # max seconds to wait for query execution


# ── Helpers ──────────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "velocity-strategy/1.0",
    "Accept": "application/json",
})


def _get(url: str, headers: dict | None = None, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 503):
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  HTTP {resp.status_code} for {url}")
            if resp.status_code in (400, 403):
                # Might be transient; retry once more with backoff
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
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


def _parse_defillama_chart(data: dict, coin_id: str) -> list:
    """Parse price records from DefiLlama chart response."""
    coin_data = data.get("coins", {}).get(coin_id, {})
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


def _fetch_price_defillama(chain: Chain) -> list:
    coin_id = f"coingecko:{chain.coingecko_id}"

    # Try with explicit start timestamp (Jan 1 2020) — more reliable
    start_ts = 1577836800  # 2020-01-01
    end_ts = int(time.time())
    span = min((end_ts - start_ts) // 86400, 3000)

    url = f"{COINS_API}/chart/{coin_id}?start={start_ts}&span={span}&period=1d"
    data = _get(url)
    if data and "coins" in data:
        records = _parse_defillama_chart(data, coin_id)
        if records:
            return records

    # Fallback: try without span/period params
    url = f"{COINS_API}/chart/{coin_id}?start={start_ts}"
    data = _get(url)
    if data and "coins" in data:
        records = _parse_defillama_chart(data, coin_id)
        if records:
            return records

    return []


def _fetch_price_coingecko(chain: Chain) -> list:
    """Fallback: CoinGecko API (free demo key required since 2024)."""
    # CoinGecko Pro vs Demo base URL
    if CG_API_KEY:
        base = "https://pro-api.coingecko.com" if CG_API_KEY.startswith("CG-") else "https://api.coingecko.com"
        headers = {"x-cg-demo-api-key": CG_API_KEY}
    else:
        base = "https://api.coingecko.com"
        headers = None
        print("  Warning: CG_API_KEY not set — CoinGecko may reject requests (HTTP 401)")
        print("  Get a free key at https://www.coingecko.com/en/api/pricing")

    url = (
        f"{base}/api/v3/coins/{chain.coingecko_id}"
        "/market_chart?vs_currency=usd&days=max&interval=daily"
    )
    data = _get(url, headers=headers)
    if not data or "prices" not in data:
        return []

    records = []
    for ts_ms, price in data["prices"]:
        records.append({
            "date": datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d"),
            "price": float(price),
        })
    return records


# ── Stablecoin transfer volume (Dune Analytics) ─────────────────────────────

def _dune_headers() -> dict:
    return {"X-Dune-API-Key": DUNE_API_KEY}


def _dune_execute_query(query_id: str) -> str | None:
    """Execute a saved Dune query. Returns execution_id or None."""
    url = f"{DUNE_API}/query/{query_id}/execute"
    try:
        resp = requests.post(url, headers=_dune_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json().get("execution_id")
        print(f"  Dune execute error: HTTP {resp.status_code}")
        return None
    except requests.RequestException as e:
        print(f"  Dune execute error: {e}")
        return None


def _dune_wait_for_results(execution_id: str) -> list | None:
    """Poll execution status, then fetch results."""
    elapsed = 0
    while elapsed < DUNE_MAX_POLL:
        url = f"{DUNE_API}/execution/{execution_id}/status"
        try:
            resp = requests.get(url, headers=_dune_headers(), timeout=30)
            if resp.status_code != 200:
                print(f"  Dune status error: HTTP {resp.status_code}")
                return None
            state = resp.json().get("state")
            if state == "QUERY_STATE_COMPLETED":
                break
            if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED",
                         "QUERY_STATE_EXPIRED"):
                print(f"  Dune query {state}")
                return None
        except requests.RequestException as e:
            print(f"  Dune poll error: {e}")
            return None
        time.sleep(DUNE_POLL_INTERVAL)
        elapsed += DUNE_POLL_INTERVAL
    else:
        print(f"  Dune query timed out after {DUNE_MAX_POLL}s")
        return None

    # Fetch results (paginated)
    all_rows = []
    url = f"{DUNE_API}/execution/{execution_id}/results"
    offset = 0
    limit = 32000
    while True:
        try:
            resp = requests.get(
                url, headers=_dune_headers(), timeout=60,
                params={"limit": limit, "offset": offset},
            )
            if resp.status_code != 200:
                print(f"  Dune results error: HTTP {resp.status_code}")
                break
            data = resp.json()
            rows = data.get("result", {}).get("rows", [])
            all_rows.extend(rows)
            if len(rows) < limit:
                break
            offset += limit
        except requests.RequestException as e:
            print(f"  Dune results error: {e}")
            break

    return all_rows if all_rows else None


def fetch_stablecoin_transfer_volume_all(
    chains: list[Chain] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily stablecoin transfer volume for all chains via a single Dune query.
    Returns {chain_name: DataFrame(date, transfer_volume)}.
    """
    if chains is None:
        chains = CHAINS

    # Build reverse map: dune_chain_id -> chain name
    dune_to_name = {}
    for c in chains:
        if c.dune_chain_id:
            dune_to_name[c.dune_chain_id] = c.name

    # Check if all chains are already cached
    all_cached = True
    result = {}
    for c in chains:
        if not c.dune_chain_id:
            continue
        cache_key = f"stablecoin_vol_{c.dune_chain_id}"
        cached = _load_cache(cache_key)
        if cached is not None:
            result[c.name] = _records_to_df(cached)
        else:
            all_cached = False

    if all_cached and result:
        print("  Stablecoin transfer volumes loaded from cache")
        return result

    # Execute Dune query
    print(f"  Executing Dune query {DUNE_QUERY_ID}...")
    execution_id = _dune_execute_query(DUNE_QUERY_ID)
    if not execution_id:
        return result  # return whatever was cached

    print(f"  Waiting for Dune results (execution: {execution_id})...")
    rows = _dune_wait_for_results(execution_id)
    if not rows:
        print("  No results from Dune query")
        return result

    print(f"  Got {len(rows)} rows from Dune")

    # Group rows by blockchain
    by_chain: dict[str, list] = {}
    for row in rows:
        blockchain = row.get("blockchain", "")
        if blockchain not in dune_to_name:
            continue
        chain_name = dune_to_name[blockchain]
        if chain_name not in by_chain:
            by_chain[chain_name] = []

        date_str = row.get("date", "")
        # Dune returns ISO timestamps like "2024-01-01T00:00:00Z"
        if "T" in str(date_str):
            date_str = str(date_str).split("T")[0]

        by_chain[chain_name].append({
            "date": date_str,
            "volume": float(row.get("transfer_volume", 0)),
        })

    # Cache and build DataFrames
    for c in chains:
        if not c.dune_chain_id:
            continue
        if c.name in by_chain:
            records = by_chain[c.name]
            cache_key = f"stablecoin_vol_{c.dune_chain_id}"
            _save_cache(cache_key, records)
            result[c.name] = _records_to_df(records)
            print(f"    {c.name}: {len(records)} days")

    return result


# ── Fetch everything ─────────────────────────────────────────────────────────

def fetch_all_data(
    chains: list[Chain] | None = None,
    volume_source: str = "transfer",
) -> dict:
    """
    Fetch supply, volume, price for all chains.

    volume_source: "transfer" = stablecoin transfer volume from Dune,
                   "dex" = DEX trading volume from DefiLlama.
    Returns: {chain_name: {"supply": df, "volume": df, "price": df, "chain": Chain}}
    """
    if chains is None:
        chains = CHAINS

    # Pre-fetch stablecoin transfer volumes from Dune (single query for all chains)
    transfer_vols: dict[str, pd.DataFrame] = {}
    use_dune = volume_source == "transfer" and DUNE_API_KEY and DUNE_QUERY_ID
    if use_dune:
        print("Fetching stablecoin transfer volumes from Dune Analytics...\n")
        transfer_vols = fetch_stablecoin_transfer_volume_all(chains)
        print()
    elif volume_source == "transfer":
        if not DUNE_API_KEY:
            print("Warning: DUNE_API_KEY not set. Falling back to DEX volume.\n")
        elif not DUNE_QUERY_ID:
            print("Warning: DUNE_QUERY_ID not set. Falling back to DEX volume.\n")

    result = {}
    total = len(chains)

    for i, chain in enumerate(chains, 1):
        print(f"[{i}/{total}] Fetching {chain.name} ({chain.symbol})...")

        supply = fetch_stablecoin_supply(chain)
        time.sleep(REQUEST_DELAY)

        # Use Dune transfer volume if available, otherwise fall back to DEX volume
        if use_dune and chain.name in transfer_vols:
            volume = transfer_vols[chain.name]
        else:
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
        vol_src = "dune" if (use_dune and chain.name in transfer_vols) else "dex"
        print(f"  -> supply: {s_days}d, volume({vol_src}): {v_days}d, price: {p_days}d")

        if s_days == 0 or v_days == 0 or p_days == 0:
            print(f"  !! Missing data for {chain.name}, will be excluded")

    return result
