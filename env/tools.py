"""OpenAI-compatible tool schemas for the TradingEnv hackathon environment.
 
Every schema is strict: ``additionalProperties`` is false and every property is
listed in ``required`` (as OpenAI strict mode demands). Optional arguments are
expressed as a nullable type, so ``price`` must always be sent -- use ``null``
for a market order.
"""
 
from __future__ import annotations
 
from typing import Any, Dict, List
 
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_market_depth",
            "description": (
                "Return the current limit order book: up to 5 price levels per side. "
                "'bids' are sorted best (highest) first, 'asks' best (lowest) first. "
                "Each level is {price, quantity}. Read-only; does not consume a step."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_order",
            "description": (
                "Submit a buy or sell order that is matched immediately against the "
                "order book, best price first. Execution prices include deterministic "
                "slippage that grows with each book level walked within a single order. "
                "Set price to null for a market order, or to a number for a limit order "
                "(fills are only taken while the slipped execution price is no worse than "
                "the limit). Short selling is not allowed; buys are capped by available "
                "cash. Returns status, fills, average price, cash and holdings. "
                "Consumes one step."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "side": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Order direction.",
                    },
                    "quantity": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of shares; a positive whole number.",
                    },
                    "price": {
                        "type": ["number", "null"],
                        "exclusiveMinimum": 0,
                        "description": (
                            "Limit price in dollars (> 0), or null for a market order."
                        ),
                    },
                },
                "required": ["side", "quantity", "price"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": (
                "Cancel a resting limit order by its order_id (the id returned by "
                "submit_order). Returns success=false if the id is not a resting order. "
                "Consumes one step."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Identifier of the resting order to cancel.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": (
                "Return cash, share holdings, the mark price (book mid), holdings value "
                "and total mark-to-market portfolio value. Read-only; does not consume a step."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]
 
TOOL_NAMES: List[str] = [tool["function"]["name"] for tool in TOOLS]