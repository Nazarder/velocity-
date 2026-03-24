"""Chain definitions, hypothesis groups, and strategy parameters."""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Chain:
    name: str
    defillama_id: str   # Chain name in DefiLlama APIs
    coingecko_id: str   # CoinGecko ID for native token price
    symbol: str
    layer: str          # "L1" or "L2"
    evm: bool
    dune_chain_id: str = ""  # Dune Analytics blockchain identifier


# ── All chains with tradeable native tokens ──────────────────────────────────

CHAINS: List[Chain] = [
    # L1 EVM
    Chain("Ethereum",  "Ethereum",  "ethereum",          "ETH",  "L1", True,  "ethereum"),
    Chain("BSC",       "BSC",       "binancecoin",       "BNB",  "L1", True,  "bnb"),
    Chain("Avalanche", "Avalanche", "avalanche-2",       "AVAX", "L1", True,  "avalanche_c"),
    Chain("Polygon",   "Polygon",   "matic-network",     "POL",  "L1", True,  "polygon"),
    Chain("Fantom",    "Fantom",    "fantom",            "FTM",  "L1", True,  "fantom"),
    Chain("Cronos",    "Cronos",    "crypto-com-chain",  "CRO",  "L1", True,  ""),
    # L1 Non-EVM
    Chain("Solana",    "Solana",    "solana",            "SOL",  "L1", False, "solana"),
    Chain("Tron",      "Tron",      "tron",              "TRX",  "L1", False, "tron"),
    Chain("Near",      "Near",      "near",              "NEAR", "L1", False, "near"),
    Chain("Sui",       "Sui",       "sui",               "SUI",  "L1", False, "sui"),
    Chain("Aptos",     "Aptos",     "aptos",             "APT",  "L1", False, "aptos"),
    # L2 EVM
    Chain("Arbitrum",  "Arbitrum",  "arbitrum",          "ARB",  "L2", True,  "arbitrum"),
    Chain("Optimism",  "Optimism",  "optimism",          "OP",   "L2", True,  "optimism"),
    Chain("zkSync Era","zkSync Era","zksync",            "ZK",   "L2", True,  "zksync"),
    Chain("Mantle",    "Mantle",    "mantle",            "MNT",  "L2", True,  "mantle"),
    # L2 Non-EVM
    Chain("Starknet",  "Starknet",  "starknet",          "STRK", "L2", False, "starknet"),
]


# ── Hypothesis groups ────────────────────────────────────────────────────────

GROUPS: Dict[str, Callable[[Chain], bool]] = {
    "all_chains":      lambda c: True,
    "l1_only":         lambda c: c.layer == "L1",
    "l2_only":         lambda c: c.layer == "L2",
    "evm_only":        lambda c: c.evm,
    "l1_evm":          lambda c: c.layer == "L1" and c.evm,
    "l1_non_evm":      lambda c: c.layer == "L1" and not c.evm,
    "l2_evm":          lambda c: c.layer == "L2" and c.evm,
}


# ── Strategy parameters ─────────────────────────────────────────────────────

VELOCITY_WINDOW   = 30      # Rolling window for velocity (days)
REBALANCE_FREQ    = 7       # Rebalance every N days
TOP_QUANTILE      = 0.33    # Top 33% = long, bottom 33% = short
MIN_CHAINS        = 4       # Minimum chains required for a valid group
MIN_SUPPLY_USD    = 1_000_000  # Ignore chains with < $1M stablecoin supply

# ── Dune Analytics ─────────────────────────────────────────────────────────

DUNE_API_KEY  = os.environ.get("DUNE_API_KEY", "")
DUNE_QUERY_ID = os.environ.get("DUNE_QUERY_ID", "6899878")  # Stablecoin transfer volume query
