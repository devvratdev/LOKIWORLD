"""Deterministic quantitative trading simulation environment.
 
All behaviour is pure programmatic math. Given the same seed and the same
sequence of actions, the environment produces identical states and hashes.
No LLM or network calls are made anywhere in this module.
"""
 
from __future__ import annotations
 
import copy
import hashlib
import json
import random
from typing import Any, Dict, List, Optional
 
# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
INITIAL_CASH: float = 100_000.0
INITIAL_HOLDINGS: int = 0
BOOK_LEVELS: int = 5
 
TICK_SIZE: float = 0.05           # price spacing between book levels
HALF_SPREAD: float = 0.05         # distance from mid to best bid / best ask
MID_PRICE_RANGE: tuple = (90.0, 110.0)
LEVEL_QTY_RANGE: tuple = (100, 1000)
 
# Deterministic slippage: execution price moves against the trader by
# BASE_SLIPPAGE_BPS + SLIPPAGE_BPS_PER_LEVEL * (levels already walked).
BASE_SLIPPAGE_BPS: float = 1.0
SLIPPAGE_BPS_PER_LEVEL: float = 2.0
 
PRICE_DECIMALS: int = 4
VALID_SIDES = ("buy", "sell")
 
 
class TradingEnv:
    """Seeded, deterministic limit-order-book trading environment."""
 
    def __init__(self) -> None:
        self.step_count: int = 0
        self.cash: float = INITIAL_CASH
        self.holdings: int = INITIAL_HOLDINGS
        self.order_book: Dict[str, List[Dict[str, float]]] = {"bids": [], "asks": []}
 
        # Resting limit orders (placeholder bookkeeping; deliberately kept out
        # of the hashed state, which covers only the core simulation state).
        self._resting_orders: Dict[int, Dict[str, Any]] = {}
        self._next_order_id: int = 1
        self._initialized: bool = False
 
    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def get_state(self) -> Dict[str, Any]:
        """Return a deep copy of the current environment state."""
        return {
            "step_count": self.step_count,
            "cash": self.cash,
            "holdings": self.holdings,
            "order_book": copy.deepcopy(self.order_book),
            "_resting_orders": copy.deepcopy(self._resting_orders)
        }
 
    @staticmethod
    def hash_state(state: Dict[str, Any]) -> str:
        """SHA-256 of the canonical JSON encoding of ``state``."""
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
 
    def compute_state_hash(self) -> str:
        """SHA-256 hash of the current state."""
        return self.hash_state(self.get_state())
 
    def _require_reset(self) -> None:
        if not self._initialized:
            raise RuntimeError("Environment not initialized: call reset(seed) first.")
 
    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed: int) -> Dict[str, Any]:
        """Seed ``random``, rebuild the initial state, return state + hash."""
        random.seed(seed)
 
        self.step_count = 0
        self.cash = INITIAL_CASH
        self.holdings = INITIAL_HOLDINGS
        self._resting_orders = {}
        self._next_order_id = 1
        self.order_book = self._generate_order_book()
        self._initialized = True
 
        state = self.get_state()
        return {"state": state, "state_hash": self.hash_state(state)}
 
    @staticmethod
    def _generate_order_book() -> Dict[str, List[Dict[str, float]]]:
        """Build a 5-level book around a seeded mid price."""
        mid = round(random.uniform(*MID_PRICE_RANGE), 2)
        best_bid = round(mid - HALF_SPREAD, 2)
        best_ask = round(mid + HALF_SPREAD, 2)
 
        bids = [
            {
                "price": round(best_bid - i * TICK_SIZE, 2),
                "quantity": random.randint(*LEVEL_QTY_RANGE),
            }
            for i in range(BOOK_LEVELS)
        ]
        asks = [
            {
                "price": round(best_ask + i * TICK_SIZE, 2),
                "quantity": random.randint(*LEVEL_QTY_RANGE),
            }
            for i in range(BOOK_LEVELS)
        ]
        return {"bids": bids, "asks": asks}
 
    # ------------------------------------------------------------------ #
    # Tool methods
    # ------------------------------------------------------------------ #
    def get_market_depth(self) -> Dict[str, List[Dict[str, float]]]:
        """Return the current order book (bids descending, asks ascending)."""
        self._require_reset()
        return copy.deepcopy(self.order_book)
 
    def submit_order(
        self,
        side: str,
        quantity: int,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Match an order against the book with deterministic slippage.
 
        ``price=None`` submits a market order (any unfilled remainder is
        discarded). A numeric ``price`` submits a limit order that is matched
        against the *slipped* execution price; any unfilled remainder is
        recorded as a resting order (placeholder, not matched later).
        Shorting is not allowed and buys are capped by available cash.
        """
        self._require_reset()
        self.step_count += 1
 
        side_norm = side.lower().strip() if isinstance(side, str) else None
        qty = self._coerce_int(quantity)
        limit = self._coerce_price(price)
 
        if side_norm not in VALID_SIDES:
            return self._reject("side must be 'buy' or 'sell'")
        if qty is None or qty <= 0:
            return self._reject("quantity must be a positive integer")
        if price is not None and (limit is None or limit <= 0):
            return self._reject("price must be a positive number or None")
        if side_norm == "sell" and qty > self.holdings:
            return self._reject("insufficient holdings (short selling not allowed)")
 
        book_side = self.order_book["asks" if side_norm == "buy" else "bids"]
        direction = 1.0 if side_norm == "buy" else -1.0
 
        fills: List[Dict[str, float]] = []
        remaining = qty
        levels_walked = 0
        slippage_cost = 0.0
 
        for level in book_side:
            if remaining <= 0:
                break
 
            quoted = level["price"]
            slip_bps = BASE_SLIPPAGE_BPS + SLIPPAGE_BPS_PER_LEVEL * levels_walked
            exec_price = round(quoted * (1.0 + direction * slip_bps / 10_000.0), PRICE_DECIMALS)
 
            if limit is not None:
                if side_norm == "buy" and exec_price > limit:
                    break
                if side_norm == "sell" and exec_price < limit:
                    break
 
            fill_qty = min(remaining, int(level["quantity"]))
            if side_norm == "buy":
                fill_qty = min(fill_qty, int(self.cash // exec_price))
            if fill_qty <= 0:
                break
 
            notional = round(fill_qty * exec_price, PRICE_DECIMALS)
            if side_norm == "buy":
                self.cash = round(self.cash - notional, PRICE_DECIMALS)
                self.holdings += fill_qty
            else:
                self.cash = round(self.cash + notional, PRICE_DECIMALS)
                self.holdings -= fill_qty
 
            level["quantity"] -= fill_qty
            remaining -= fill_qty
            levels_walked += 1
            slippage_cost += abs(exec_price - quoted) * fill_qty
            fills.append(
                {
                    "quoted_price": quoted,
                    "price": exec_price,
                    "quantity": fill_qty,
                }
            )
 
        # Drop fully consumed levels.
        self.order_book["asks"] = [lv for lv in self.order_book["asks"] if lv["quantity"] > 0]
        self.order_book["bids"] = [lv for lv in self.order_book["bids"] if lv["quantity"] > 0]
 
        filled_qty = qty - remaining
        total_notional = sum(f["price"] * f["quantity"] for f in fills)
        avg_price = round(total_notional / filled_qty, PRICE_DECIMALS) if filled_qty else 0.0
 
        order_id = self._next_order_id
        self._next_order_id += 1
 
        resting = False
        if remaining > 0 and limit is not None:
            self._resting_orders[order_id] = {
                "order_id": order_id,
                "side": side_norm,
                "quantity": remaining,
                "price": limit,
            }
            resting = True
 
        if filled_qty == qty:
            status = "filled"
        elif filled_qty > 0:
            status = "partially_filled"
        else:
            status = "unfilled"
 
        return {
            "order_id": order_id,
            "status": status,
            "side": side_norm,
            "requested_quantity": qty,
            "filled_quantity": filled_qty,
            "remaining_quantity": remaining,
            "average_price": avg_price,
            "slippage_cost": round(slippage_cost, PRICE_DECIMALS),
            "resting": resting,
            "fills": fills,
            "cash": self.cash,
            "holdings": self.holdings,
            "step_count": self.step_count,
            "state_hash": self.compute_state_hash(),
        }
 
    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """Cancel a resting limit order (placeholder implementation)."""
        self._require_reset()
        self.step_count += 1
 
        oid = self._coerce_int(order_id)
        order = self._resting_orders.pop(oid, None) if oid is not None else None
 
        if order is None:
            return {
                "success": False,
                "order_id": order_id,
                "error": "order_id not found among resting orders",
                "step_count": self.step_count,
                "state_hash": self.compute_state_hash(),
            }
        return {
            "success": True,
            "order_id": oid,
            "cancelled_order": order,
            "step_count": self.step_count,
            "state_hash": self.compute_state_hash(),
        }
 
    def get_portfolio(self) -> Dict[str, Any]:
        """Return cash, holdings and mark-to-market value (marked at mid)."""
        self._require_reset()
        mark = self._mark_price()
        holdings_value = round(self.holdings * mark, PRICE_DECIMALS)
        return {
            "cash": self.cash,
            "holdings": self.holdings,
            "mark_price": mark,
            "holdings_value": holdings_value,
            "total_value": round(self.cash + holdings_value, PRICE_DECIMALS),
            "step_count": self.step_count,
        }
 
    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _mark_price(self) -> float:
        """Mid price; falls back to the available side, or 0.0 if book is empty."""
        bids, asks = self.order_book["bids"], self.order_book["asks"]
        if bids and asks:
            return round((bids[0]["price"] + asks[0]["price"]) / 2.0, PRICE_DECIMALS)
        if bids:
            return bids[0]["price"]
        if asks:
            return asks[0]["price"]
        return 0.0
 
    def _reject(self, reason: str) -> Dict[str, Any]:
        return {
            "order_id": None,
            "status": "rejected",
            "reason": reason,
            "filled_quantity": 0,
            "cash": self.cash,
            "holdings": self.holdings,
            "step_count": self.step_count,
            "state_hash": self.compute_state_hash(),
        }
 
    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None
 
    @staticmethod
    def _coerce_price(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None
 