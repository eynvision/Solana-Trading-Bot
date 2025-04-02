
#!/usr/bin/env python3

import os
import sys
import requests
import time
import json
import base64
import traceback
from datetime import datetime, timezone, timedelta
from dateutil import parser  # For parsing pool_created_at
from geckoterminal_py import GeckoTerminalSyncClient

# Solana imports
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.keypair import Keypair

from solana.publickey import PublicKey
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.client import Token

# -------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------

SOLANA_RPC_URL = "https://attentive-nameless-thunder.solana-mainnet.quiknode.pro/36057343869a97df17621e986257497cad16d87c"
# SOLANA_RPC_URL = "https://api.devnet.solana.com"

# Jupiter Swap endpoint (publicly available):
JUPITER_API = "https://quote-api.jup.ag/v6"


# GeckoTerminal endpoint:
GECKO_API_URL = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"

# File Paths
LOG_FILE = "trading_log.json"
HISTORY_FILE = "token_history.json"

# Trading Parameters
MIN_LIQUIDITY = 10000           # Minimum pool liquidity in USD
MIN_VOLUME = 1000              # Minimum 24h (or short timeframe) trading volume in USD
BUY_AMOUNT_SOL = 0.00001       # SOL amount per token position
MAX_POSITIONS = 1              # Maximum simultaneous token positions
SLIPPAGE_TOLERANCE = 10000        # Slippage tolerance in basis points (0.5% = 50 bps)
PRICE_CHECK_INTERVAL = 30     # Seconds between trading iterations
POST_BUY_DELAY = 60            # Seconds to wait after a buy before evaluating for sell

# Risk Management Parameters
STOP_LOSS = -0.01              # Stop loss at -1%
PROFIT_TARGETS = [
    (0.001, 0.50),  # 0.1% profit: sell 50%
    (1.0, 1.00),    # 100% profit: sell 100%
    (2.0, 1.00),    # 200% profit: sell 100%
    (3.0, 1.00),    # 300% profit: sell 100%
    (4.0, 1.00)     # 400% profit: sell 100%
]

# Wallet Configuration
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")
if not WALLET_PRIVATE_KEY:
    print("❌ Error: WALLET_PRIVATE_KEY environment variable not set")
    sys.exit(1)

# Constant Mint Addresses
SOL_MINT = "So11111111111111111111111111111111111111112"

# API Headers
API_HEADERS = {
    'Accept': 'application/json',
    'User-Agent': 'Solana Trading Bot'
}

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def get_sol_price():
    """Fetch the current price of SOL in USD from CoinGecko."""
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "solana", "vs_currencies": "usd"},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return data.get("solana", {}).get("usd", None)
    except Exception as e:
        print(f"⚠️ Error fetching SOL price from CoinGecko: {e}")
    return None

def log_transaction(action, data, log_file=LOG_FILE):
    """Log trading transactions to a JSON file with timestamp."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        **data
    }
    try:
        with open(log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        print(f"📝 Logged {action} transaction")
    except Exception as e:
        print(f"⚠️ Logging error: {e}")

def load_token_history(history_file=HISTORY_FILE):
    """Load token trading history from JSON."""
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"⚠️ Error loading token history: {e}")
        return {}

def save_token_history(history, history_file=HISTORY_FILE):
    """Save token trading history to JSON."""
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving token history: {e}")

def get_token_info(token_address):
    """Retrieve token name and symbol from GeckoTerminal."""
    try:
        url = f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_address}/info"
        response = requests.get(url, headers=API_HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'attributes' in data['data']:
                attributes = data['data']['attributes']
                name = attributes.get('name') or token_address[:8]
                symbol = attributes.get('symbol') or name
                return name, symbol
    except Exception as e:
        print(f"Error getting token info: {e}")
    # Fallback
    return token_address[:8], token_address[:8]

def is_valid_mint(address):
    """Validate Solana mint address format by attempting to parse as Pubkey."""
    try:
        # Strip any known prefixes
        clean_address = address.replace('solana_', '')
        Pubkey.from_string(clean_address)
        return True
    except Exception as e:
        print(f"⚠️ Invalid mint address {address}: {e}")
        return False

def get_current_price(token_address):
    """
    Get current token price in USD from the 'new_pools' endpoint.
    If multiple pools exist, takes the first returned pool's base_token_price_usd.
    """
    try:
        response = requests.get(
            f"{GECKO_API_URL}?base_token={token_address}",
            headers=API_HEADERS,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']:
                # We assume the first pool in 'data' is relevant
                return float(data['data'][0]['attributes']['base_token_price_usd'])
    except Exception as e:
        print(f"Error getting price for {token_address}: {str(e)}")
    return None

def get_token_decimals(token_address, rpc_url=SOLANA_RPC_URL):
    """
    Fetch the number of decimals for a given SPL token.
    Falls back to 6 decimals if fetching fails.
    """
    try:
        # Clean the token address
        clean_address = token_address.replace('solana_', '')
        client = Client(rpc_url)
        token = Token(client, PublicKey(clean_address), TOKEN_PROGRAM_ID, None)
        decimals = token.get_decimals()
        return decimals
    except Exception as e:
        print(f"⚠️ Error fetching decimals for {token_address}: {e}. Using fallback value (6 decimals).")
        return 6  # Fallback to 6 decimals

# -------------------------------------------------------------------
# Swap Module
# -------------------------------------------------------------------

class SolanaSwapper:
    """Solana token swapping utility using Jupiter Aggregator."""
    def __init__(self, rpc_url, wallet_keypair):
        self.rpc_url = rpc_url
        self.wallet_keypair = wallet_keypair
        self.solana_client = Client(rpc_url)
        self.jupiter_api = JUPITER_API
        self._non_tradable_tokens = set()
        self._invalid_tokens = set()
        self.gecko_client = GeckoTerminalSyncClient()
        self._token_decimals_cache = {}  # Cache for token decimals

    def get_token_decimals(self, token_address):
        """Fetch and cache token decimals."""
        clean_address = token_address.replace('solana_', '')
        if clean_address in self._token_decimals_cache:
            return self._token_decimals_cache[clean_address]
        decimals = get_token_decimals(clean_address, self.rpc_url)
        if decimals is not None:
            self._token_decimals_cache[clean_address] = decimals
        return decimals

    def get_token_dexes(self, token_address):
        """Get list of DEXes where token is trading using geckoterminal_py client."""
        try:
            # Clean token address used by geckoterminal client
            clean_address = token_address.replace('solana_', '')

            # Get top pools for the token
            pools = self.gecko_client.get_top_pools_by_network_token(
                network_id="solana",
                token_id=clean_address
            )

            if pools.empty:
                # If no top pools found, check 'new_pools'
                print("🔍 Checking new pools via geckoterminal...")
                new_pools = self.gecko_client.get_new_pools_by_network("solana")
                pools = new_pools[
                    (new_pools['base_token_id'] == clean_address) |
                    (new_pools['quote_token_id'] == clean_address)
                ]

            if pools.empty:
                print(f"🔍 No liquidity pools found for token {token_address}")
                return []

            # Gather DEX info
            dexes = set()
            total_volume = 0
            total_liquidity = 0

            for _, pool in pools.iterrows():
                dex_id = pool.get('dex_id')
                if dex_id:
                    dexes.add(dex_id)
                    liquidity = float(pool.get('reserve_in_usd', 0) or 0)
                    volume = float(pool.get('volume_usd_h24', 0) or 0)
                    total_volume += volume
                    total_liquidity += liquidity

                    # Additional logs
                    buys_24h = int(pool.get('transactions_h24_buys', 0) or 0)
                    sells_24h = int(pool.get('transactions_h24_sells', 0) or 0)
                    price_change = float(pool.get('price_change_percentage_h24', 0) or 0)
                    print(f"  💱 DEX: {dex_id}")
                    print(f"    💰 Liquidity: ${liquidity:,.2f}")
                    print(f"    📊 24h Volume: ${volume:,.2f}")
                    print(f"    🔄 24h Trades: {buys_24h} buys, {sells_24h} sells")
                    print(f"    📈 24h Price Change: {price_change:,.2f}%")

            print(f"\n📊 Aggregate Stats for {token_address}:")
            print(f"  💰 Total Liquidity: ${total_liquidity:,.2f}")
            print(f"  📊 Total 24h Volume: ${total_volume:,.2f}")
            print(f"  🏦 DEXes: {', '.join(dexes)}")

            if total_liquidity < MIN_LIQUIDITY:
                print(f"⚠️ Liquidity (${total_liquidity:,.2f}) below {MIN_LIQUIDITY}")
                return []

            if total_volume < MIN_VOLUME:
                print(f"⚠️ Volume (${total_volume:,.2f}) below {MIN_VOLUME}")
                return []

            return list(dexes)

        except Exception as e:
            print(f"🤦‍♂️ Error gathering liquidity data for token {token_address}: {e}")
            return []

    def is_token_tradable_on_jupiter(self, input_mint, output_mint, lamports, slippage_bps=50):
        """
        Ping Jupiter for a swap quote to see if there's a route.
        If we get a valid 200 response with a route, it's tradable.
        """
        try:
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(lamports),  # int
                "slippageBps": slippage_bps,
                "swapMode": "ExactIn"
            }
            response = requests.get(
                f"{self.jupiter_api}/quote",
                params=params,
                headers=API_HEADERS,
                timeout=10
            )
            if response.status_code == 200:
                return True
            else:
                data = {}
                try:
                    data = response.json()
                except:
                    pass

                if data.get("errorCode") == "TOKEN_NOT_TRADABLE":
                    self._non_tradable_tokens.add(output_mint)
                    print(f"❌ Token not supported by Jupiter: {output_mint}")
                else:
                    print(f"❌ Jupiter quote error: {data.get('error', 'Unknown error')}")
            return False

        except Exception as e:
            print(f"❌ is_token_tradable_on_jupiter error: {str(e)}")
            return False

    def is_token_tradable(self, token_address):
        """
        1) Check if a token is on at least one DEX with enough liquidity (via get_token_dexes).
        2) Check if Jupiter has a route: SOL -> token.
        3) Check if Jupiter also has a route: token -> SOL (ensures sellability).
        """
        if token_address in self._non_tradable_tokens:
            print(f"⚠️ Token {token_address} previously marked as non-tradable")
            return False

        # 1) Check DEX availability
        print(f"\n🔍 Checking DEX availability for {token_address} ...")
        dexes = self.get_token_dexes(token_address)
        if not dexes:
            print("❌ Token not found on any suitable DEX or insufficient liquidity.")
            self._non_tradable_tokens.add(token_address)
            return False
        print(f"✅ Token is on DEXes: {dexes}")

        # 2) Check buy route on Jupiter with a small SOL input => token out
        print(f"🔄 Checking Jupiter route SOL -> {token_address} ...")
        # E.g. 0.001 SOL => 1000000 lamports
        if not self.is_token_tradable_on_jupiter(SOL_MINT, token_address, 1000000, SLIPPAGE_TOLERANCE):
            print(f"❌ No route to buy {token_address} found on Jupiter aggregator.")
            return False

        # 3) Check sell route on Jupiter with a small token input => SOL out
        print(f"🔄 Checking Jupiter route {token_address} -> SOL ...")
        # We'll do a "small" amount in token lamports. We don't know token decimals from the snippet.
        # We can attempt 1000000 "lamports" as well, or even less. The aggregator won't fill if decimals mismatch, 
        # but we at least see if it returns 200. 
        if not self.is_token_tradable_on_jupiter(token_address, SOL_MINT, 1000000, SLIPPAGE_TOLERANCE):
            print(f"❌ No route to sell {token_address} found on Jupiter aggregator.")
            return False

        print("✅ Token is tradable and sellable on Jupiter.")
        return True

    def check_transfer_success(self, error_data):
        """
        Extract logs from a Jupiter error to detect if partial transfer succeeded.
        """
        try:
            if not isinstance(error_data, str):
                error_data = str(error_data)

            if "logs: Some([" in error_data:
                logs_start = error_data.find("logs: Some([") + 11
                logs_end = error_data.rfind("])")
                logs_str = error_data[logs_start:logs_end]
                logs = [line.strip().strip('"') for line in logs_str.split('", "')]
            else:
                return False

            successful_transfers = []
            last_operation = None

            for log in logs:
                if "Instruction:" in log:
                    last_operation = log
                elif "success" in log and last_operation:
                    if "TransferChecked" in last_operation or "Transfer" in last_operation:
                        successful_transfers.append(last_operation)
                    last_operation = None

            if successful_transfers:
                print("✅ Completed transfers (partial success):")
                for op in successful_transfers:
                    print(f"  - {op}")
                return True

            return False

        except Exception as e:
            print(f"Error checking transfer success: {e}")
            return False

    def execute_swap(self, input_mint, output_mint, amount_sol_or_tokens, slippage_bps=50):
        """
        Execute a token swap using Jupiter aggregator.
          - input_mint: mint of token you are giving
          - output_mint: mint of token you want
          - amount_sol_or_tokens: in "units" (SOL or token). We'll convert to lamports based on decimals.
        """
        try:
            # Skip known invalid tokens
            if output_mint in self._invalid_tokens:
                print(f"⚠️ Skipping known invalid token: {output_mint}")
                return None

            # Validate mint addresses
            if not all(is_valid_mint(m) for m in [input_mint, output_mint]):
                print("❌ Invalid mint address(es).")
                return None

            # Check if the token is indeed tradable
            if not self.is_token_tradable(output_mint) and input_mint == SOL_MINT:
                print(f"⚠️ {output_mint} is not tradable/sellable. Aborting buy.")
                return None

            # Get the Jupiter quote
            quote = self.get_swap_quote(input_mint, output_mint, amount_sol_or_tokens, slippage_bps)
            if not quote:
                print("❌ No quote available. Aborting swap.")
                return None

            # Create the transaction
            swap_transaction_raw = self.create_swap_transaction(quote, str(self.wallet_keypair.pubkey()))
            if not swap_transaction_raw:
                print("❌ Could not create swap transaction. Aborting swap.")
                return None

            # Decode base64 transaction
            swap_transaction_buf = base64.b64decode(swap_transaction_raw)

            # Build an unsigned versioned transaction
            unsigned_tx = VersionedTransaction.from_bytes(swap_transaction_buf)

            # Sign the transaction (only the fee-payer / user)
            signed_tx = VersionedTransaction(unsigned_tx.message, [self.wallet_keypair])

            # Send transaction
            tx_hash = self.solana_client.send_raw_transaction(bytes(signed_tx))
            print(f"📝 Transaction sent: {tx_hash.value}")

            # Confirm transaction
            try:
                # Specify commitment level if needed, e.g., 'confirmed'
                status = self.solana_client.confirm_transaction(tx_hash.value, commitment="confirmed")
                if status.value and len(status.value) > 0:
                    tx_status = status.value[0]
                    if tx_status.err:
                        print(f"❌ Transaction failed: {tx_status.err}")
                        return None
                    else:
                        print(f"✅ Transaction confirmed: {tx_hash.value}")
                else:
                    print("⚠️ No status returned for the transaction.")
                    return None

            except Exception as e:
                error_str = str(e)
                print(f"⚠️ Transaction warning: {error_str} ({type(e).__name__})")
                import traceback
                traceback.print_exc()

                if "custom program error: 6001" in error_str:
                    logs = getattr(e, 'data', {})
                    if self.check_transfer_success(logs):
                        return {
                            "success": True,
                            "transaction_hash": str(tx_hash.value),
                            "amount": quote.get('outAmount', 0),
                            "warning": "Partial success - router error"
                        }
                if "blockhash" in error_str.lower():
                    print("💡 Possibly a blockhash expiration. Try again quickly.")
                return None

            # Calculate the output amount based on decimals
            output_decimals = self.get_token_decimals(output_mint)
            if output_decimals is None:
                print(f"⚠️ Unable to fetch decimals for output token: {output_mint}")
                return None
            out_amount_units = float(quote.get('outAmount', 0)) / 10**output_decimals

            return {
                "success": True,
                "transaction_hash": str(tx_hash.value),
                "amount": out_amount_units
            }

        except Exception as e:
            print(f"❌ Swap execution error: {str(e)}")
            return None


    def get_swap_quote(self, input_mint, output_mint, amount_units, slippage_bps=50):
        """
        Get swap quote from Jupiter.
          - If input is SOL, multiply by 1e9 for lamports
          - If input is a token, use its actual decimals
        """
        try:
            # Fetch input token decimals
            input_decimals = 9 if input_mint == SOL_MINT else self.get_token_decimals(input_mint)
            if input_decimals is None:
                print(f"⚠️ Unable to fetch decimals for input token: {input_mint}")
                return None

            # Convert amount_units to raw units based on decimals
            lamports = int(amount_units * 10**input_decimals)

            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(lamports),
                "slippageBps": slippage_bps,
                "swapMode": "ExactIn"
            }

            headers = {
                **API_HEADERS,
                "Authorization": f"Bearer {os.getenv('JUPITER_API_KEY', '')}"
            }

            resp = requests.get(
                f"{self.jupiter_api}/quote",
                params=params,
                headers=headers,
                timeout=10
            )

            if resp.status_code == 200:
                data = resp.json()
                return data
            else:
                data = {}
                try:
                    data = resp.json()
                except:
                    pass
                if data.get("errorCode") == "TOKEN_NOT_TRADABLE":
                    self._non_tradable_tokens.add(output_mint)
                print(f"❌ Jupiter quote error: {data.get('error', 'Unknown error')}")
                return None

        except Exception as e:
            print(f"❌ Swap Quote Error: {str(e)}")
            return None

    def create_swap_transaction(self, quote, user_public_key):
        """
        Create the Jupiter swap transaction from the quote response.
        This hits the /swap endpoint with the 'quoteResponse' + userPublicKey.
        """
        try:
            swap_params = {
                "quoteResponse": quote,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": True
            }
            resp = requests.post(
                f"{self.jupiter_api}/swap",
                json=swap_params,
                headers=API_HEADERS,
                timeout=10
            )

            if resp.status_code == 200:
                return resp.json().get('swapTransaction')

            elif resp.status_code == 500:
                error_data = resp.json()
                error_msg = error_data.get('error', '')
                if "Missing token program" in error_msg:
                    token_address = error_msg.split()[-1]
                    self._invalid_tokens.add(token_address)
                    print(f"❌ Invalid token (missing program): {token_address}")
                else:
                    print(f"❌ Swap Transaction Creation Error: {resp.status_code}")
                    print(f"Response: {resp.text}")
                return None
            else:
                print(f"❌ Swap Transaction Creation Error: {resp.status_code}")
                print(f"Response: {resp.text}")
                return None

        except Exception as e:
            print(f"❌ Swap Transaction Creation Error: {str(e)}")
            return None

    def buy_token(self, token_address, sol_amount):
        """Convenience function: Buy token_address using sol_amount SOL."""
        return self.execute_swap(SOL_MINT, token_address, sol_amount, SLIPPAGE_TOLERANCE)

    def sell_token(self, token_address, token_amount):
        """Convenience function: Sell token_address for SOL."""
        return self.execute_swap(token_address, SOL_MINT, token_amount, SLIPPAGE_TOLERANCE)

# -------------------------------------------------------------------
# Main Trading Bot
# -------------------------------------------------------------------

class SolanaTrader:
    """Main trading bot class for Solana token trading."""

    def __init__(self, rpc_url, wallet_private_key):
        # Create keypair from private key
        try:
            self.wallet_keypair = Keypair.from_bytes(bytes.fromhex(wallet_private_key))
        except Exception as e:
            print(f"❌ Invalid wallet private key: {e}")
            sys.exit(1)

        # Initialize Solana client
        self.solana_client = Client(rpc_url)

        # Initialize swapper
        self.swapper = SolanaSwapper(rpc_url, self.wallet_keypair)

        # Trading state
        self.wallet_tokens = {}
        self.token_history = load_token_history()

    def get_wallet_balance(self):
        """Get wallet SOL balance."""
        try:
            balance = self.solana_client.get_balance(self.wallet_keypair.pubkey())
            return float(balance.value) / 10**9  # lamports -> SOL
        except Exception as e:
            print(f"⚠️ Error getting wallet balance: {e}")
            return 0

    def get_new_solana_pools(self):
        """Fetch new Solana pools from GeckoTerminal's new_pools endpoint."""
        try:
            response = requests.get(GECKO_API_URL, headers=API_HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            print(f"API Error: {str(e)}")
            return []

    def evaluate_pool(self, pool):
        """
        Evaluate if a pool meets:
          - Liquidity >= MIN_LIQUIDITY
          - volume (24h, or short timeframe) >= MIN_VOLUME
          - pool age < 1 hour
          - has at least 1$ in last trades and at least 10$ (approx check with 'm5' volume)
        """
        attributes = pool.get('attributes', {})
        # Check liquidity
        liquidity = float(attributes.get('reserve_in_usd', 0) or 0

                         )

        # We also want to check if there's a 5-min or 1-hour volume that meets threshold
        # Sample structure: "volume_usd":{"m5":"1190.29515","h1":"1190.29515", ...}
        volume_usd_obj = attributes.get('volume_usd', {})
        vol_5m = float(volume_usd_obj.get('m5', 0) or 0)
        vol_1h = float(volume_usd_obj.get('h1', 0) or 0)
        vol_24h = float(attributes.get('volume_usd_h24', 0) or 0)  # sometimes top-level?

        # For simplicity, we check 24h and the short timeframe to ensure there's some *recent* activity
        combined_volume = max(vol_24h, vol_1h, vol_5m)

        if liquidity < MIN_LIQUIDITY:
            return False
        if combined_volume < MIN_VOLUME:
            return False

        # 4) "Check latest transactions in token if it's above 1$ and at least one of them above 10$."
        # We interpret: we want to see a 5-min volume > 1$ AND > 10$. Let's do a minimal check:
        if vol_5m < 1:
            return False
        if vol_5m < 10:
            # We can't confirm there's a *single* trade above $10, but the user specifically asked.
            # We'll do an approximate check: if the 5-min volume is < 10, skip.
            return False

        # 7) Pool age < 1 hour
        pool_created_str = attributes.get('pool_created_at')
        if not pool_created_str:
            return False
        try:
            pool_created_time = parser.isoparse(pool_created_str)
            now_utc = datetime.now(timezone.utc)
            age_seconds = (now_utc - pool_created_time).total_seconds()
            if age_seconds > 3600:  # older than 1 hour
                return False
        except Exception as e:
            print(f"⚠️ Could not parse pool_created_at: {e}")
            return False

        return True

    def calculate_token_performance(self, token_address, current_price_usd):
        """
        Calculate token performance (P/L) from token_history purchases.
        - `current_price_usd` = how much 1 token is worth in USD
        - Returns performance dict or None if no recorded purchases
        """
        token_data = self.token_history.get(token_address, {})
        if not token_data.get('purchases'):
            return None

        # Summation
        total_sol_invested = sum(p['sol_spent'] for p in token_data['purchases'])
        total_tokens_bought = sum(p['amount'] for p in token_data['purchases'])

        # Fetch token decimals
        token_decimals = token_data.get('decimals')
        if token_decimals is None:
            token_decimals = self.swapper.get_token_decimals(token_address)
            if token_decimals is None:
                print(f"⚠️ Unable to fetch decimals for token: {token_address}")
                return None
            # Update token history with decimals
            token_data['decimals'] = token_decimals
            self.token_history[token_address] = token_data
            save_token_history(self.token_history)

        # Current value in SOL
        sol_price = get_sol_price() or 0
        if sol_price <= 0:
            return None

        # 1 token = current_price_usd USD
        # 1 SOL   = sol_price USD
        # => 1 token = (current_price_usd / sol_price) SOL
        token_price_in_sol = current_price_usd / sol_price

        # Use the new method to get the actual token balance
        current_token_balance = self.get_wallet_token_balance(token_address)
        if current_token_balance == 0:
            print(f"⚠️ Token {token_address} balance is zero.")
            return None

        # Convert token balance from raw units to actual tokens
        actual_token_balance = current_token_balance / 10**token_decimals

        # Our holdings in SOL
        current_value_in_sol = actual_token_balance * token_price_in_sol

        # Our P/L in SOL
        profit_loss_sol = current_value_in_sol - total_sol_invested
        profit_loss_percentage = 0.0
        if total_sol_invested > 0:
            profit_loss_percentage = (profit_loss_sol / total_sol_invested) * 100.0

        return {
            "total_sol_invested": total_sol_invested,
            "total_tokens": actual_token_balance,
            "current_value_sol": current_value_in_sol,
            "profit_loss_percentage": profit_loss_percentage,
            "average_purchase_price_sol": total_sol_invested / actual_token_balance
            if actual_token_balance > 0 else 0
        }

    def should_sell_token(self, token_address, current_price_usd):
        """
        Determine selling strategy based on profit/loss.
        Stop-loss or partial sells per profit targets.
        Returns a dict or None if hold.
        """
        performance = self.calculate_token_performance(token_address, current_price_usd)
        if not performance:
            return None

        pnl_pct = performance['profit_loss_percentage']

        # Stop loss
        if pnl_pct <= STOP_LOSS * 100:  # e.g. STOP_LOSS = -0.01 => -1%
            return {
                "action": "sell_all",
                "reason": "Stop Loss Triggered",
                "current_pnl": pnl_pct
            }

        # Profit-taking
        for target, sell_percentage in PROFIT_TARGETS:
            if pnl_pct >= target * 100:
                return {
                    "action": "sell_partial",
                    "percentage": sell_percentage,
                    "reason": f"Profit Target {target*100:.1f}% Reached",
                    "current_pnl": pnl_pct
                }

        return None  # No action (hold)

    def get_wallet_token_balance(self, token_address):
        """
        Fetch the correct token balance from associated token accounts.
        """
        try:
            token_accounts = self.solana_client.get_token_accounts_by_owner(
                self.wallet_keypair.pubkey(),
                {"mint": token_address},
                commitment=Confirmed
            )
            
            if not token_accounts['result']['value']:
                print(f"⚠️ No associated token account found for {token_address}")
                return 0

            total_balance = 0
            for account in token_accounts['result']['value']:
                balance = self.solana_client.get_token_account_balance(account['pubkey'])
                token_balance = int(balance['result']['value']['amount'])
                print(f"🔎 Found token balance: {token_balance} for {token_address} in account {account['pubkey']}")
                total_balance += token_balance

            return total_balance

        except Exception as e:
            print(f"⚠️ Error fetching token balance for {token_address}: {e}")
            return 0

    def monitor_wallet_transactions(self, token_address, last_checked_time):
        """
        Check the wallet for new purchases of the specified token after the last_checked_time.
        Returns True if a new purchase is detected, False otherwise.
        """
        try:
            # Fetch all token accounts for the wallet
            token_accounts = self.solana_client.get_token_accounts_by_owner(
                self.wallet_keypair.pubkey(),
                {"mint": token_address},
                commitment=Confirmed
            )
            if not token_accounts['result']['value']:
                return False

            # For each token account, fetch the balance and recent transactions
            for account in token_accounts['result']['value']:
                pubkey = account['pubkey']
                balance = self.solana_client.get_token_account_balance(pubkey)
                token_balance = int(balance['result']['value']['amount'])

                # Check recent transactions involving this token account
                transactions = self.solana_client.get_signatures_for_address(pubkey, limit=10)
                for tx in transactions['result']:
                    tx_time = self.solana_client.get_transaction(tx['signature'])
                    if tx_time['result'] and tx_time['result']['blockTime']:
                        tx_block_time = datetime.fromtimestamp(tx_time['result']['blockTime'], tz=timezone.utc)
                        if tx_block_time > last_checked_time:
                            # Further verification can be done here
                            return True
            return False

        except Exception as e:
            print(f"⚠️ Error monitoring wallet transactions: {e}")
            return False

    def main_trading_loop(self):
        """Main trading loop for monitoring and executing trades."""

        print("🚀 Solana Token Trading Bot Started")
        print(f"💳 Wallet Address: {self.wallet_keypair.pubkey()}")

        # Initialize last_checked_time to current time minus a buffer to capture recent transactions
        last_checked_time = datetime.now(timezone.utc) - timedelta(seconds=PRICE_CHECK_INTERVAL)

        while True:
            try:
                # Update wallet balance
                wallet_balance = self.get_wallet_balance()
                sol_price = get_sol_price() or 0
                print(f"\n💰 Wallet Balance: {wallet_balance:.6f} SOL (≈ ${wallet_balance * sol_price:.2f})")
                print(f"📊 Active Positions: {len(self.wallet_tokens)}")

                # Check existing positions for possible sells
                for token_address in list(self.wallet_tokens.keys()):
                    try:
                        current_price = get_current_price(token_address)
                        if not current_price:
                            continue
                        sell_decision = self.should_sell_token(token_address, current_price)
                        if sell_decision:
                            token_info = self.token_history.get(token_address, {})
                            name = token_info.get('name', token_address[:8])
                            symbol = token_info.get('symbol', token_address[:8])

                            if sell_decision['action'] == 'sell_all':
                                # Sell entire position
                                sell_amount = token_info.get('total_tokens', 0)
                                if sell_amount > 0:
                                    result = self.swapper.sell_token(token_address, sell_amount)
                                    if result and result.get('success'):
                                        print(f"💸 Emergency Sell (Stop-Loss): {name} ({symbol}) @ {sell_decision['current_pnl']:.2f}% P/L")
                                        # Remove from active positions and history
                                        del self.wallet_tokens[token_address]
                                        if token_address in self.token_history:
                                            del self.token_history[token_address]
                                        save_token_history(self.token_history)
                                        log_transaction("sell_all", {
                                            "token_address": token_address,
                                            "name": name,
                                            "symbol": symbol,
                                            "amount_tokens": sell_amount,
                                            "reason": sell_decision['reason'],
                                            "pnl": sell_decision['current_pnl']
                                        })

                            elif sell_decision['action'] == 'sell_partial':
                                sell_percentage = sell_decision['percentage']
                                total_amount = token_info.get('total_tokens', 0)
                                partial_amount = total_amount * sell_percentage
                                if partial_amount > 0:
                                    result = self.swapper.sell_token(token_address, partial_amount)
                                    if result and result.get('success'):
                                        print(f"💰 Partial Sell for Profit: {name} ({symbol})")
                                        print(f"    Sold {sell_percentage*100:.1f}% @ {sell_decision['current_pnl']:.2f}% P/L")
                                        # Adjust position
                                        remaining = total_amount - partial_amount
                                        token_info['total_tokens'] = remaining
                                        self.token_history[token_address] = token_info
                                        save_token_history(self.token_history)
                                        log_transaction("sell_partial", {
                                            "token_address": token_address,
                                            "name": name,
                                            "symbol": symbol,
                                            "amount_tokens": partial_amount,
                                            "reason": sell_decision['reason'],
                                            "pnl": sell_decision['current_pnl']
                                        })

                                        # If user sold everything, remove entirely
                                        if remaining <= 0:
                                            del self.wallet_tokens[token_address]
                                            del self.token_history[token_address]
                                            save_token_history(self.token_history)

                    except Exception as e:
                        print(f"❌ Error managing token {token_address}: {e}")

                # Fetch new pools to see if we can buy new tokens (only if we haven't reached MAX_POSITIONS)
                if len(self.wallet_tokens) < MAX_POSITIONS:
                    try:
                        pools = self.get_new_solana_pools()
                        for pool in pools:
                            # Evaluate the pool's metrics & age
                            if not self.evaluate_pool(pool):
                                continue

                            # Identify the token address
                            # By convention, 'relationships.base_token' often is the "project" side
                            # and 'quote_token' is SOL. We check if base_token is not the SOL mint.
                            base_token_data = pool['relationships']['base_token']['data']
                            quote_token_data = pool['relationships']['quote_token']['data']
                            base_id = base_token_data['id']  # e.g. "solana_3j3rqH..."
                            quote_id = quote_token_data['id']
                            # We want to buy the "base" if the "quote" is "solana_So1111..."
                            # If it's reversed, just skip or handle differently.

                            if SOL_MINT in base_id:
                                # That means base_token is actually SOL => we skip, because it's the other side
                                continue

                            # Now strip "solana_" prefix
                            token_address = base_id.replace("solana_", "")
                            if not is_valid_mint(token_address):
                                print(f"⚠️ Invalid token address: {token_address}")
                                continue

                            # 8) If we have seen (bought) this token before, skip
                            if token_address in self.token_history:
                                continue

                            # 5) Double-check if the token is sellable (done inside is_token_tradable)
                            #    but let's do an extra layer
                            if not self.swapper.is_token_tradable(token_address):
                                continue

                            # Get token info & metrics
                            name, symbol = get_token_info(token_address)
                            attributes = pool.get('attributes', {})
                            try:
                                price_usd = float(attributes.get('base_token_price_usd', 0))
                            except (ValueError, TypeError):
                                price_usd = 0

                            if price_usd <= 0:
                                continue

                            # Decide how much SOL to spend
                            # We won't exceed 90% of wallet balance
                            buy_amount_sol = min(BUY_AMOUNT_SOL, wallet_balance * 0.9)
                            if buy_amount_sol <= 0:
                                continue

                            # Execute buy
                            buy_result = self.swapper.buy_token(token_address, buy_amount_sol)
                            if buy_result and buy_result.get('success'):
                                token_amount = buy_result['amount']
                                # Fetch token decimals
                                token_decimals = self.swapper.get_token_decimals(token_address)
                                if token_decimals is None:
                                    print(f"⚠️ Unable to fetch decimals for token: {token_address}. Skipping.")
                                    continue

                                # Record in token_history
                                self.token_history[token_address] = {
                                    "purchases": [{
                                        "timestamp": time.time(),
                                        "amount": token_amount,
                                        "purchase_price_usd": price_usd,
                                        "sol_spent": buy_amount_sol
                                    }],
                                    "total_tokens": token_amount,
                                    "total_sol_invested": buy_amount_sol,
                                    "decimals": token_decimals,  # Store decimals
                                    "name": name,
                                    "symbol": symbol
                                }
                                # Add to active positions
                                self.wallet_tokens[token_address] = {
                                    "amount": token_amount,
                                    "purchase_time": time.time(),
                                    "symbol": symbol,
                                    "name": name,
                                    "decimals": token_decimals  # Store decimals
                                }
                                save_token_history(self.token_history)

                                # Log the purchase
                                log_transaction("buy", {
                                    "token_address": token_address,
                                    "name": name,
                                    "symbol": symbol,
                                    "amount_sol": buy_amount_sol,
                                    "amount_tokens": token_amount,
                                    "price_usd": price_usd,
                                    "decimals": token_decimals,
                                    "transaction_hash": buy_result.get('transaction_hash')
                                })
                                print(f"\n✨ Bought New Token => {name} ({symbol})")
                                print(f"   🔑 Address: {token_address}")
                                print(f"   💰 Spent: {buy_amount_sol} SOL, Received ~{token_amount:.6f} tokens.")

                                # 🛠 Debug: Check wallet balance after buy
                                wallet_balance = self.get_wallet_balance()
                                print(f"🛠 Debug: Wallet Balance After Buy: {wallet_balance:.6f} SOL")
                                
                                # 🛠 Debug: Check if the token account exists
                                token_accounts = self.solana_client.get_token_accounts_by_owner(
                                    self.wallet_keypair.pubkey(),
                                    {"mint": token_address},
                                    commitment=Confirmed
                                )
                                if not token_accounts['result']['value']:
                                    print(f"⚠️ Warning: No token accounts found for {token_address} after buy!")
                                else:
                                    print(f"✅ Token {token_address} successfully detected in wallet after buy!")

                                # Implement delay after buying
                                print(f"⏳ Waiting {POST_BUY_DELAY} seconds before evaluating for sell...")
                                time.sleep(POST_BUY_DELAY)

                            # If we reached max positions, break early
                            if len(self.wallet_tokens) >= MAX_POSITIONS:
                                break

                    except Exception as e:
                        print(f"⚠️ Error fetching or processing pools: {e}")

                # Print summary
                print(f"\n📊 Active Positions: {len(self.wallet_tokens)}")
                print(f"🔄 Updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
                print(f"⏳ Waiting {PRICE_CHECK_INTERVAL} seconds...")
                print("-" * 50)

                # Update last_checked_time
                last_checked_time = datetime.now(timezone.utc)

                # Wait
                time.sleep(PRICE_CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                save_token_history(self.token_history)
                break

            except Exception as e:
                print(f"⚠️ Critical error in main loop: {e}")
                log_transaction("system_error", {"error": str(e)})
                time.sleep(5)

# -------------------------------------------------------------------
# Script entry point
# -------------------------------------------------------------------

def main():
    """Entry point for the trading bot."""
    # Validate the wallet private key
    try:
        Keypair.from_bytes(bytes.fromhex(WALLET_PRIVATE_KEY))
    except Exception as e:
        print(f"❌ Invalid wallet private key: {e}")
        sys.exit(1)

    # Initialize trader
    trader = SolanaTrader(SOLANA_RPC_URL, WALLET_PRIVATE_KEY)

    # Print startup info
    print("🚀 Solana Token Trading Bot Initializing...")
    print(f"💳 Wallet Address: {trader.wallet_keypair.pubkey()}")
    print(f"⚖️ Min Liquidity: ${MIN_LIQUIDITY:,}")
    print(f"📈 Min Volume: ${MIN_VOLUME:,}")
    print(f"💸 Buy Amount: {BUY_AMOUNT_SOL} SOL")
    print(f"🎯 Max Positions: {MAX_POSITIONS}")
    print(f"📉 Stop Loss: {STOP_LOSS * 100:.2f}%")
    print(f"📈 Profit Targets: {PROFIT_TARGETS}")
    print(f"⏰ Post-Buy Delay: {POST_BUY_DELAY} seconds")
    print("-" * 50)

    # Start trading loop
    trader.main_trading_loop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        # Attempt logging
        try:
            with open(LOG_FILE, "a") as f:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "runtime_error",
                    "error": str(e)
                }, f)
                f.write("\n")
        except:
            pass
        sys.exit(1)


