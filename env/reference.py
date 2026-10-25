"""Reference (oracle) solutions for the TradingEnv hackathon tasks.
 
The harness calls ``run_reference(env, task_config)`` on a freshly reset
environment (setup actions already executed). The function dispatches on
``task_config["task_id"]`` and plays a fixed, known-good plan using only
``env.submit_order`` for state-changing steps. ``get_market_depth`` and
``get_portfolio`` are read-only and consume no steps.
 
Plans (all taken from the ``reference`` notes in ``tasks.json``):
 
* ``demo_task``                      one market buy of 100 (best ask holds 220).
* ``t1_block_buy_500``               one limit order per book level.
* ``t1_limit_discipline_buy_800``    one limit order per book level.
* ``t2_inventory_capped_churn``      buy 400, sell 400, buy 100, sell 100,
                                     each leg as one limit order per level.
* ``t2_volatility_shock_liquidation`` two market sells of half the inventory
                                     each; done in 2 steps, before the hidden
                                     shock (which fires after agent step 3).
 
Why "one limit order per level": slippage grows with every level a *single*
order walks (1 bp + 2 bp per level already walked). An order that only ever
touches the top level always pays the minimum 1 bp, so splitting a block into
level-sized orders is the cheapest way to cross a multi-level book. The limit
price is set just above (buy) / below (sell) that first-level execution price,
so an order can never spill into a deeper, more slipped level.
"""
 
from __future__ import annotations
 
from typing import Any, Callable, Dict, List, Optional
 
try:  # keep the limit-price guard in sync with the environment's slippage model
    from env import BASE_SLIPPAGE_BPS
except ImportError:  # pragma: no cover - standalone fallback
    BASE_SLIPPAGE_BPS = 1.0
 
PRICE_DECIMALS = 4
LIMIT_BUFFER = 0.0001     # absorbs float rounding in the slipped execution price
MAX_ORDERS_PER_LEG = 6    # safety net: a 5-level book never needs more
 
 
# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
class _Runner:
    """Thin wrapper that forwards orders to the env and records an episode log."""
 
    def __init__(self, env: Any) -> None:
        self.env = env
        self.log: List[Dict[str, Any]] = []
 
    def order(self, side: str, quantity: int, price: Optional[float]) -> Dict[str, Any]:
        args = {"side": side, "quantity": int(quantity), "price": price}
        result = self.env.submit_order(**args)
        self.log.append({"tool": "submit_order", "arguments": args, "result": result})
        return result
 
    def best_level(self, side: str) -> Optional[Dict[str, float]]:
        """Top-of-book level to *hit*: asks when buying, bids when selling."""
        book = self.env.get_market_depth()
        levels = book["asks" if side == "buy" else "bids"]
        return levels[0] if levels else None
 
    def holdings(self) -> int:
        return int(self.env.get_portfolio()["holdings"])
 
 
def _first_level_limit(side: str, quoted: float) -> float:
    """Limit price that admits the top level (1 bp slip) but nothing deeper."""
    bps = BASE_SLIPPAGE_BPS / 10_000.0
    if side == "buy":
        return round(quoted * (1.0 + bps), PRICE_DECIMALS) + LIMIT_BUFFER
    return round(quoted * (1.0 - bps), PRICE_DECIMALS) - LIMIT_BUFFER
 
 
def _trade_per_level(runner: _Runner, side: str, quantity: int) -> int:
    """Trade ``quantity`` shares as one limit order per book level.
 
    Each order is sized to the top level's resting quantity (or the remainder)
    and priced so it can only fill at that level. Returns the shares filled.
    """
    remaining = int(quantity)
    for _ in range(MAX_ORDERS_PER_LEG):
        if remaining <= 0:
            break
        level = runner.best_level(side)
        if level is None:
            break
        chunk = min(remaining, int(level["quantity"]))
        result = runner.order(side, chunk, _first_level_limit(side, level["price"]))
        filled = int(result.get("filled_quantity", 0))
        if filled <= 0:
            break
        remaining -= filled
    return int(quantity) - remaining
 
 
# --------------------------------------------------------------------------- #
# Per-task plans
# --------------------------------------------------------------------------- #
def _demo_task(runner: _Runner) -> None:
    # The best ask holds 220 shares, so a single market order stays on level 1.
    runner.best_level("buy")  # inspect the book first (read-only, free)
    runner.order("buy", 100, None)
 
 
def _block_buy_500(runner: _Runner) -> None:
    _trade_per_level(runner, "buy", 500)
 
 
def _limit_discipline_buy_800(runner: _Runner) -> None:
    _trade_per_level(runner, "buy", 800)
 
 
def _inventory_capped_churn(runner: _Runner) -> None:
    # Peak |position| = 400, total volume = 400 + 400 + 100 + 100 = 1,000, flat at the end.
    for side, qty in (("buy", 400), ("sell", 400), ("buy", 100), ("sell", 100)):
        _trade_per_level(runner, side, qty)
 
 
def _volatility_shock_liquidation(runner: _Runner) -> None:
    # Two market sells keep us at 2 agent steps, i.e. flat before the shock
    # (which fires after step 3). Size the legs from the actual inventory.
    inventory = runner.holdings()
    if inventory <= 0:
        return
    first = max(1, inventory // 2)
    runner.order("sell", first, None)
    left = runner.holdings()
    if left > 0:
        runner.order("sell", left, None)
 
 
_PLANS: Dict[str, Callable[[_Runner], None]] = {
    "demo_task": _demo_task,
    "t1_block_buy_500": _block_buy_500,
    "t1_limit_discipline_buy_800": _limit_discipline_buy_800,
    "t2_inventory_capped_churn": _inventory_capped_churn,
    "t2_volatility_shock_liquidation": _volatility_shock_liquidation,
}
 
 
# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_reference(env: Any, task_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Solve ``task_config["task_id"]`` on ``env`` and return the episode log.
 
    ``env`` must already be reset (and any setup actions executed). The returned
    log is a list of ``{"tool", "arguments", "result"}`` dicts that can be fed
    straight to ``verifier.verify``.
    """
    task_id = task_config.get("task_id")
    plan = _PLANS.get(task_id)
    if plan is None:
        raise ValueError(
            f"No reference solution for task_id={task_id!r}; known: {sorted(_PLANS)}"
        )
    runner = _Runner(env)
    plan(runner)
    return runner.log
 
 
# --------------------------------------------------------------------------- #
# Self-check: python reference.py [path/to/tasks.json]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    import sys
 
    from env import TradingEnv
    from verifier import verify_detailed
 
    path = sys.argv[1] if len(sys.argv) > 1 else "tasks.json"
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
 
    for task in spec["tasks"]:
        cfg = {
            **task,
            "metric_definitions": spec["metric_definitions"],
            "initial_state": spec["initial_state"],
        }
        env = TradingEnv()
        env.reset(task["seed"])
        setup_log = []
        for action in task["setup_actions"]:
            res = getattr(env, action["tool"])(**action["arguments"])
            setup_log.append({"tool": action["tool"], "arguments": action["arguments"], "result": res})
        agent_log = run_reference(env, cfg)
        detail = verify_detailed(cfg, setup_log + agent_log, env.get_portfolio())
        print(f"{task['task_id']:<36} reward={detail['reward']:.4f} "
              f"steps={int(detail['metrics'].get('agent_steps', 0))}")
 