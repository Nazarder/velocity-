"""Chain definitions, hypothesis groups, and strategy parameters."""

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


# ── All chains with tradeable native tokens ──────────────────────────────────

CHAINS: List[Chain] = [
    # L1 EVM
    Chain("Ethereum",  "Ethereum",  "ethereum",          "ETH",  "L1", True),
    Chain("BSC",       "BSC",       "binancecoin",       "BNB",  "L1", True),
    Chain("Avalanche", "Avalanche", "avalanche-2",       "AVAX", "L1", True),
    Chain("Polygon",   "Polygon",   "matic-network",     "POL",  "L1", True),
    Chain("Fantom",    "Fantom",    "fantom",            "FTM",  "L1", True),
    Chain("Cronos",    "Cronos",    "crypto-com-chain",  "CRO",  "L1", True),
    # L1 Non-EVM
    Chain("Solana",    "Solana",    "solana",            "SOL",  "L1", False),
    Chain("Tron",      "Tron",      "tron",              "TRX",  "L1", False),
    Chain("Near",      "Near",      "near",              "NEAR", "L1", False),
    Chain("Sui",       "Sui",       "sui",               "SUI",  "L1", False),
    Chain("Aptos",     "Aptos",     "aptos",             "APT",  "L1", False),
    # L2 EVM
    Chain("Arbitrum",  "Arbitrum",  "arbitrum",          "ARB",  "L2", True),
    Chain("Optimism",  "Optimism",  "optimism",          "OP",   "L2", True),
    Chain("zkSync Era","zkSync Era","zksync",            "ZK",   "L2", True),
    Chain("Mantle",    "Mantle",    "mantle",            "MNT",  "L2", True),
    # L2 Non-EVM
    Chain("Starknet",  "Starknet",  "starknet",          "STRK", "L2", False),
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
