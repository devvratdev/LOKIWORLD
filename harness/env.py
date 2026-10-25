"""The environment: reset / step / score, backed by the in-process TradingEnv (env/env.py).
 
TradingEnv only knows the order book, cash and holdings. Everything else a task needs is done
here, because tasks.json describes it and the harness is responsible for it:
 
  * setup_actions   run right after reset(seed); they are not agent steps (flagged "setup").
  * scenario_events the hidden volatility shock fires once the agent has taken the configured
                    number of steps, or at the end of the episode if it is still pending.
  * episode_log     every call and result, in order. It is exactly what verify() receives.
 
Env also exposes the four tool methods (get_market_depth, submit_order, cancel_order,
get_portfolio) so env/reference.py can drive it like a TradingEnv, with events still applied.
"""
import copy
import json
 
from env.env import TradingEnv
from env.tools import TOOL_NAMES as ENV_TOOL_NAMES
from env.verifier import verify
 
try:
    from env.verifier import verify_detailed
except ImportError:                     # only used for extra diagnostics
    verify_detailed = None
 
from .tasks import build_task
 
PASS_SCORE = 0.95                       # a run counts as a success at or above this score
STEP_TOOLS = ("submit_order", "cancel_order")   # the calls that consume an agent step
 
 
class Env:
    def __init__(self):
        self.task = None
        self.log = []                   # the episode_log handed to verify()
        self.listener = None            # optional callback(entry) after each agent tool call
        self._env = None
        self._events = []
        self._n_setup = 0
 
    # ---------------------------------------------------------------- reset
    def reset(self, task_id, seed=None):
        """Start a clean episode. `seed=None` uses the seed from tasks.json for this task."""
        self.task = build_task(task_id, seed)
        config = self.task.config
        self._env = TradingEnv()
        reset_out = self._env.reset(config["seed"])         # {"state": ..., "state_hash": ...}
        # First log entry: the post-reset book, from which the verifier reads arrival_mid.
        self.log = [{"event": "reset", "seed": config["seed"], "result": copy.deepcopy(reset_out)}]
        self._events = [{**copy.deepcopy(event), "fired": False}
                        for event in config.get("scenario_events") or []]
        setup = config.get("setup_actions") or []
        self._n_setup = len(setup)
        for action in setup:                                # harness-run, not counted as agent steps
            self.call(action["tool"], action["arguments"], setup=True)
        return (f"Task: {self.task.instructions}\n\n"
                "You are trading a single stock through the provided tools. get_market_depth and "
                "get_portfolio are free; every submit_order or cancel_order call uses one step of "
                f"your budget ({self.task.max_steps} steps). Call `finish` when you are done.")
 
    # ----------------------------------------------------------------- step
    def step(self, name, args):
        """Run one tool call. Returns (text, is_error)."""
        result = self.call(name, args)
        is_error = result.get("status") == "rejected" or result.get("success") is False
        return json.dumps(result), is_error
 
    def call(self, name, args, setup=False):
        """Run one environment tool, log it, and apply any scenario event that became due."""
        if self._env is None:
            raise RuntimeError("call reset() before using the environment")
        if name not in ENV_TOOL_NAMES:
            raise ValueError(f"unknown environment tool '{name}'")
        result = getattr(self._env, name)(**args)
        entry = {"tool": name, "arguments": copy.deepcopy(args), "result": copy.deepcopy(result)}
        if setup:
            entry["setup"] = True
        self.log.append(entry)
        if not setup:
            if name in STEP_TOOLS:
                self._fire_events("steps")
            if self.listener:
                self.listener(entry)
        return result
 
    # The same four methods as TradingEnv, so reference/baseline code can use this object directly.
    def get_market_depth(self):
        return self.call("get_market_depth", {})
 
    def submit_order(self, side, quantity, price=None):
        return self.call("submit_order", {"side": side, "quantity": quantity, "price": price})
 
    def cancel_order(self, order_id):
        return self.call("cancel_order", {"order_id": order_id})
 
    def get_portfolio(self):
        return self.call("get_portfolio", {})
 
    @property
    def step_count(self):
        return self._env.step_count
 
    @property
    def agent_steps(self):
        return self._env.step_count - self._n_setup
 
    def state_hash(self):
        return self._env.compute_state_hash()[:8]
 
    # ------------------------------------------------------- scenario events
    def _fire_events(self, reason):
        steps = self.agent_steps
        for event in self._events:
            if event["fired"]:
                continue
            trigger = event.get("trigger") or {}
            after = trigger.get("after_agent_steps")
            due = ((reason == "steps" and after is not None and steps >= after)
                   or (reason == "end" and trigger.get("fire_at_episode_end_if_pending", False)))
            if due:
                self._apply_event(event, reason)
 
    def _apply_event(self, event, reason):
        kind = event.get("type")
        if kind == "volatility_shock":
            self._volatility_shock(event["effect"])
        else:
            raise ValueError(f"unsupported scenario event type '{kind}'")
        event["fired"] = True
        self.log.append({"event": kind, "fired_on": reason, "agent_steps": self.agent_steps,
                         "disclosed_to_agent": bool(event.get("disclosed_to_agent", False)),
                         "state_hash": self._env.compute_state_hash()})
 
    def _volatility_shock(self, effect):
        """Gap every level's price and thin its quantity, as described in tasks.json."""
        price_mult = effect["price_multiplier"]
        qty_mult = effect["quantity_multiplier"]
        decimals = effect.get("price_rounding_decimals", 2)
        min_qty = effect.get("min_level_quantity", 1)
        book = self._env.order_book
        for side in effect.get("apply_to", ["bids", "asks"]):
            for level in book[side]:
                level["price"] = round(level["price"] * price_mult, decimals)
                level["quantity"] = max(min_qty, int(level["quantity"] * qty_mult))
 
    def finalize(self):
        """End of episode: fire events that are still pending. Safe to call more than once."""
        self._fire_events("end")
 
    # ---------------------------------------------------------------- score
    def score(self):
        """Reward from env/verifier.py. Returns {"total": float, ...diagnostics}.
 
        Reading it mid-episode is a running score and changes nothing. Call finalize() first
        for the final score, so a still-pending end-of-episode event is included.
        """
        config = self.task.config
        portfolio = self._env.get_portfolio()               # read-only, not logged, no step
        out = {"total": round(float(verify(config, self.log, portfolio)), 4)}
        if verify_detailed is not None:
            detail = verify_detailed(config, self.log, portfolio)
            out["criteria_passed"] = f"{sum(c['passed'] for c in detail['criteria'])}/{len(detail['criteria'])}"
            out["metrics"] = detail["metrics"]
        return out
 