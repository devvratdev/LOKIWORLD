"""Deterministic, LLM-free verifier for the TradingEnv hackathon tasks.
 
Public API
----------
``verify(task_config, episode_log, final_state) -> float``
    Reward in ``[0.0, 1.0]``.
``verify_detailed(task_config, episode_log, final_state) -> dict``
    Same computation, but also returns the metrics and per-criterion results
    (useful for debugging and logging).
 
Inputs
------
task_config
    A task from ``tasks.json`` merged with the file-level ``metric_definitions``
    and ``initial_state`` (the task's own ``success_criteria``, ``seed``,
    ``setup_actions`` and ``reference`` are used as well).
episode_log
    Ordered list of dicts. Each entry should carry the tool call and the
    environment response, e.g.
    ``{"tool": "submit_order", "arguments": {...}, "result": {...}}``.
    Common variants are tolerated: OpenAI-style ``function`` blocks, a call
    entry followed by a separate result entry, and bare result dicts.
    Harness-run setup actions may be included in the log; they are excluded
    from every agent metric (flagged with ``"setup": true`` or recognised by
    ``step_count <= len(setup_actions)``).
final_state
    The environment state at episode end -- the ``get_portfolio()`` output
    (preferred) or ``TradingEnv.get_state()``. If a scenario event fires at
    episode end (e.g. the volatility shock), the harness must apply it before
    capturing this state, because ``equity_loss_usd`` is marked at book mid.
 
Reward design
-------------
* All criteria met                          -> 1.0
* No agent step taken (and not all met)     -> 0.0
* Otherwise every criterion earns a score in [0, 1]:
    - passed                                -> 1.0
    - missed "budget" criteria (<=, <)      -> decays linearly to 0 as the metric
                                               overshoots its limit by 50%
    - missed "progress" criteria (>=, >, ==)-> proportional to progress made
  The reward is ``0.8 * weighted_mean(scores) * completion`` where
  ``completion`` is the mean score of the primary objectives (fill quantity,
  final holdings, traded volume). Any failed run is therefore capped at 0.8,
  a run that finishes the job but breaches a budget lands in roughly 0.5-0.8,
  and a run that fails a primary objective is capped at 0.5.
"""
 
from __future__ import annotations
 
import json
import random
from typing import Any, Dict, List, Optional, Tuple
 
# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_INITIAL_CASH: float = 100_000.0
DEFAULT_INITIAL_HOLDINGS: float = 0.0
ORDER_TOOLS: Tuple[str, ...] = ("submit_order", "cancel_order")
 
METRIC_DECIMALS: int = 4
EPS: float = 1e-9
 
# Metrics that represent the actual job to be done; the rest are constraints.
PRIMARY_METRICS = frozenset({"filled_buy_quantity", "total_filled_volume", "final_holdings"})
PRIMARY_WEIGHT: float = 2.0
CONSTRAINT_WEIGHT: float = 1.0
 
OVERSHOOT_TOLERANCE: float = 0.5   # budget overshoot (relative) at which score hits 0
MAX_PARTIAL_REWARD: float = 0.8    # ceiling for any run that misses a criterion
PRIMARY_FAILURE_CAP: float = 0.5   # ceiling when a primary objective is missed
 
_SETUP_MARKERS = ("setup", "harness")
_NAME_KEYS = ("tool", "tool_name", "name", "action")
_ARG_KEYS = ("arguments", "args", "parameters", "input")
_CALL_KEYS = ("action", "tool_call", "function_call", "call")
_RESULT_KEYS = (
    "result", "response", "observation", "output",
    "env_response", "tool_result", "content",
)
_OPERATORS = ("<=", ">=", "==", "<", ">", "!=")
 
 
# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _as_obj(value: Any) -> Any:
    """Decode JSON strings; return anything else unchanged."""
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value
 
 
def _num(value: Any) -> Optional[float]:
    """Finite float or None (bools and NaN/inf are rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out
 
 
def _book_mid(book: Any, require_both: bool = True) -> Optional[float]:
    """Mid of the best bid and best ask (optionally falls back to one side)."""
    if not isinstance(book, dict):
        return None
 
    def prices(side: str) -> List[float]:
        levels = book.get(side) or []
        vals = [_num(lv.get("price")) for lv in levels if isinstance(lv, dict)]
        return [v for v in vals if v is not None]
 
    bids, asks = prices("bids"), prices("asks")
    if bids and asks:
        return round((max(bids) + min(asks)) / 2.0, METRIC_DECIMALS)
    if require_both:
        return None
    if bids:
        return max(bids)
    if asks:
        return min(asks)
    return None
 
 
# --------------------------------------------------------------------------- #
# Episode-log parsing
# --------------------------------------------------------------------------- #
def _extract_call(entry: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = [entry]
    for key in _CALL_KEYS:
        nested = entry.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for cand in candidates:
        fn = cand.get("function")
        if isinstance(fn, dict):
            cand = fn
        name = next((cand[k] for k in _NAME_KEYS if isinstance(cand.get(k), str)), None)
        if name:
            args = next((_as_obj(cand[k]) for k in _ARG_KEYS if k in cand), None)
            if args is None:
                args = next((_as_obj(entry[k]) for k in _ARG_KEYS if k in entry), {})
            return name, args if isinstance(args, dict) else {}
    return None, {}
 
 
def _extract_result(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in _RESULT_KEYS:
        if key in entry:
            value = _as_obj(entry[key])
            if isinstance(value, dict):
                return value
    looks_like_result = (
        any(k in entry for k in ("fills", "filled_quantity", "success", "total_value"))
        or ("status" in entry and "order_id" in entry)
        or ("cash" in entry and "holdings" in entry)
        or ("bids" in entry and "asks" in entry)
    )
    return entry if looks_like_result else None
 
 
def _infer_tool(result: Dict[str, Any]) -> Optional[str]:
    if "fills" in result or "requested_quantity" in result or result.get("status") in (
        "rejected", "filled", "partially_filled", "unfilled"
    ):
        return "submit_order"
    if "success" in result and "order_id" in result:
        return "cancel_order"
    if "bids" in result and "asks" in result:
        return "get_market_depth"
    if "total_value" in result:
        return "get_portfolio"
    return None
 
 
def _parse_log(episode_log: Any) -> List[Dict[str, Any]]:
    """Normalise the log into ``{tool, args, result, setup}`` records."""
    records: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
 
    for entry in episode_log or []:
        if not isinstance(entry, dict):
            continue
        name, args = _extract_call(entry)
        result = _extract_result(entry)
        if name is None and result is None:
            continue
 
        setup = bool(
            entry.get("setup")
            or entry.get("is_setup")
            or entry.get("phase") in _SETUP_MARKERS
            or entry.get("source") in _SETUP_MARKERS
        )
 
        # A bare result directly after a call-only entry completes that call.
        if name is None and result is not None and pending is not None:
            pending["result"] = result
            pending["setup"] = pending["setup"] or setup
            pending = None
            continue
 
        rec = {"tool": name, "args": args, "result": result, "setup": setup}
        if rec["tool"] is None and result is not None:
            rec["tool"] = _infer_tool(result)
        records.append(rec)
        pending = rec if (name is not None and result is None) else None
 
    return records
 
 
def _find_arrival_mid_in_log(episode_log: Any) -> Optional[float]:
    """Look for an explicit ``arrival_mid`` or the post-reset book in the log."""
    for entry in episode_log or []:
        if not isinstance(entry, dict):
            continue
        containers = [entry] + [
            _as_obj(entry.get(k)) for k in ("result", "response", "observation", "output")
        ]
        for c in containers:
            if not isinstance(c, dict):
                continue
            explicit = _num(c.get("arrival_mid"))
            if explicit is not None:
                return explicit
            state = c.get("state") if isinstance(c.get("state"), dict) else c
            if state.get("step_count") == 0 and "order_book" in state:
                mid = _book_mid(state["order_book"])
                if mid is not None:
                    return mid
    return None
 
 
def _replay_arrival_mid(seed: Any) -> Optional[float]:
    """Recompute the post-reset mid by replaying the seed through TradingEnv."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        return None
    saved_state = random.getstate()
    try:
        from env import TradingEnv  # local import: the verifier works without it
 
        reset = TradingEnv().reset(seed)
        return _book_mid(reset["state"]["order_book"])
    except Exception:  # noqa: BLE001 - any failure just means "fall back"
        return None
    finally:
        random.setstate(saved_state)
 
 
# --------------------------------------------------------------------------- #
# Episode model + metric computation
# --------------------------------------------------------------------------- #
class _Episode:
    """Everything the metrics need, extracted once from log + final state."""
 
    def __init__(self, task_config: Dict[str, Any], episode_log: Any, final_state: Any) -> None:
        initial = task_config.get("initial_state") or {}
        self.initial_cash = _num(initial.get("cash"))
        if self.initial_cash is None:
            self.initial_cash = DEFAULT_INITIAL_CASH
        initial_holdings = _num(initial.get("holdings"))
        if initial_holdings is None:
            initial_holdings = DEFAULT_INITIAL_HOLDINGS
 
        n_setup = len(task_config.get("setup_actions") or [])
        records = _parse_log(episode_log)
 
        # ---- final state --------------------------------------------------
        outer = final_state if isinstance(final_state, dict) else {}
        view = outer
        if "cash" not in outer:
            for key in ("state", "portfolio", "final_state"):
                if isinstance(outer.get(key), dict):
                    view = outer[key]
                    break
        self.final_view: Dict[str, Any] = view
        state_steps = _num(outer.get("step_count"))
        if state_steps is None:
            state_steps = _num(view.get("step_count"))
 
        # ---- split setup / agent records ---------------------------------
        order_recs = [r for r in records if r["tool"] in ORDER_TOOLS]
        for r in order_recs:
            step_no = _num(r["result"].get("step_count")) if r["result"] else None
            r["is_setup"] = r["setup"] or (
                n_setup > 0 and step_no is not None and step_no <= n_setup
            )
        if (
            n_setup > 0
            and state_steps is not None
            and not any(r["is_setup"] for r in order_recs)
            and len(order_recs) == int(state_steps)
        ):
            # The log contains the setup actions but did not flag them.
            for r in order_recs[:n_setup]:
                r["is_setup"] = True
 
        setup_recs = [r for r in order_recs if r["is_setup"]]
        agent_recs = [r for r in order_recs if not r["is_setup"]]
 
        # ---- agent_steps ----------------------------------------------------
        steps = len(agent_recs)
        if state_steps is not None:
            steps = max(steps, max(0, int(state_steps) - n_setup))
        self.agent_steps: int = steps
 
        # ---- fills and holdings trace -------------------------------------
        self.buy_fills: List[Tuple[float, int]] = []
        self.sell_fills: List[Tuple[float, int]] = []
        self.holdings_trace: List[float] = []
        first_before: Optional[float] = None
 
        for r in agent_recs:
            res = r["result"]
            if r["tool"] != "submit_order" or not isinstance(res, dict):
                continue
            side = str(res.get("side") or r["args"].get("side") or "").lower().strip()
            bought = sold = 0
            for fill in res.get("fills") or []:
                price, qty = _num(fill.get("price")), _num(fill.get("quantity"))
                if price is None or qty is None or qty <= 0:
                    continue
                if side == "buy":
                    self.buy_fills.append((price, int(qty)))
                    bought += int(qty)
                elif side == "sell":
                    self.sell_fills.append((price, int(qty)))
                    sold += int(qty)
            held = _num(res.get("holdings"))
            if held is not None:
                self.holdings_trace.append(held)
                if first_before is None:
                    first_before = held - bought + sold
 
        # ---- inherited holdings (position handed over after setup) ---------
        inherited: Optional[float] = None
        for r in setup_recs:
            if r["tool"] == "submit_order" and isinstance(r["result"], dict):
                held = _num(r["result"].get("holdings"))
                if held is not None:
                    inherited = held
        if inherited is None:
            inherited = first_before if first_before is not None else initial_holdings
        self.inherited_holdings: float = inherited
 
        # ---- final holdings / equity ----------------------------------------
        fh = _num(view.get("holdings"))
        if fh is None:
            fh = self.holdings_trace[-1] if self.holdings_trace else (
                inherited if self.agent_steps == 0 else None
            )
        self.final_holdings: Optional[float] = fh
 
        total_value = _num(view.get("total_value"))
        if total_value is None:
            cash, held = _num(view.get("cash")), _num(view.get("holdings"))
            if cash is not None and held is not None:
                mark = _num(view.get("mark_price"))
                if mark is None:
                    mark = _book_mid(view.get("order_book"), require_both=False)
                total_value = cash + held * (mark if mark is not None else 0.0)
        self.total_value: Optional[float] = total_value
 
        # ---- arrival mid ----------------------------------------------------
        arrival = _find_arrival_mid_in_log(episode_log)
        if arrival is None:
            arrival = _replay_arrival_mid(task_config.get("seed"))
        if arrival is None:
            arrival = _num((task_config.get("reference") or {}).get("arrival_mid"))
        self.arrival_mid: Optional[float] = arrival
 
    # ------------------------------------------------------------------ #
    @property
    def peak_abs_holdings(self) -> Optional[float]:
        values = list(self.holdings_trace)
        if self.final_holdings is not None:
            values.append(self.final_holdings)
        return max(abs(v) for v in values) if values else None
 
    def compute_metrics(self) -> Dict[str, Optional[float]]:
        buy_qty = sum(q for _, q in self.buy_fills)
        sell_qty = sum(q for _, q in self.sell_fills)
 
        shortfall: Optional[float]
        max_over: Optional[float]
        if not self.buy_fills:
            shortfall, max_over = 0.0, 0.0
        elif self.arrival_mid is None:
            shortfall, max_over = None, None
        else:
            paid = sum(p * q for p, q in self.buy_fills)
            shortfall = paid - buy_qty * self.arrival_mid
            max_over = max(p for p, _ in self.buy_fills) - self.arrival_mid
 
        equity_loss = (
            None if self.total_value is None else self.initial_cash - self.total_value
        )
        return {
            "arrival_mid": self.arrival_mid,
            "filled_buy_quantity": float(buy_qty),
            "total_filled_volume": float(buy_qty + sell_qty),
            "implementation_shortfall_usd": shortfall,
            "max_fill_price_over_arrival_mid": max_over,
            "peak_abs_holdings": self.peak_abs_holdings,
            "final_holdings": self.final_holdings,
            "equity_loss_usd": equity_loss,
            "agent_steps": float(self.agent_steps),
        }
 
 
# --------------------------------------------------------------------------- #
# Criterion evaluation
# --------------------------------------------------------------------------- #
def _passes(value: float, op: str, threshold: float) -> bool:
    if op == "<=":
        return value <= threshold + EPS
    if op == ">=":
        return value >= threshold - EPS
    if op == "==":
        return abs(value - threshold) <= EPS
    if op == "<":
        return value < threshold - EPS
    if op == ">":
        return value > threshold + EPS
    return abs(value - threshold) > EPS  # "!="
 
 
def _scale(metric: str, threshold: float, episode: _Episode) -> float:
    """Reference magnitude used to turn a miss into a relative error."""
    if metric == "final_holdings":
        peak = episode.peak_abs_holdings or 0.0
        return max(peak, abs(episode.inherited_holdings), 1.0)
    return abs(threshold) if abs(threshold) > EPS else 1.0
 
 
def _closeness(value: float, op: str, threshold: float, scale: float) -> float:
    """Score in [0, 1] for a criterion that was *not* met."""
    if op in ("<=", "<"):
        overshoot = (value - threshold) / scale
        return max(0.0, 1.0 - overshoot / OVERSHOOT_TOLERANCE)
    if op in (">=", ">"):
        return max(0.0, min(1.0, 1.0 - (threshold - value) / scale))
    if op == "==":
        return max(0.0, 1.0 - abs(value - threshold) / scale)
    return 0.0
 
 
def _evaluate_criterion(
    criterion: Dict[str, Any], metrics: Dict[str, Optional[float]], episode: _Episode
) -> Dict[str, Any]:
    metric = criterion.get("metric")
    op = criterion.get("operator")
    threshold = _num(criterion.get("value"))
    if op not in _OPERATORS:
        raise ValueError(f"Unsupported operator {op!r} in success_criteria")
    if threshold is None:
        raise ValueError(f"Non-numeric threshold for metric {metric!r}")
 
    raw = metrics.get(metric)
    value = round(raw, METRIC_DECIMALS) if raw is not None else None
    primary = metric in PRIMARY_METRICS
 
    if value is None:  # metric could not be computed -> criterion not met
        passed, score = False, 0.0
    else:
        passed = _passes(value, op, threshold)
        score = 1.0 if passed else _closeness(value, op, threshold, _scale(metric, threshold, episode))
 
    return {
        "metric": metric,
        "operator": op,
        "threshold": threshold,
        "value": value,
        "passed": passed,
        "score": round(score, 6),
        "primary": primary,
        "weight": PRIMARY_WEIGHT if primary else CONSTRAINT_WEIGHT,
    }
 
 
# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def verify_detailed(
    task_config: Dict[str, Any], episode_log: Any, final_state: Any
) -> Dict[str, Any]:
    """Compute metrics, evaluate success criteria and derive the reward."""
    criteria = task_config.get("success_criteria")
    if not criteria:
        raise ValueError("task_config must contain a non-empty 'success_criteria' list")
 
    episode = _Episode(task_config, episode_log, final_state)
    all_metrics = episode.compute_metrics()
 
    wanted = set(task_config.get("metric_definitions") or {}) | {c.get("metric") for c in criteria}
    metrics = {
        name: (round(all_metrics[name], METRIC_DECIMALS) if all_metrics.get(name) is not None else None)
        for name in sorted(n for n in wanted if n)
    }
 
    results = [_evaluate_criterion(c, all_metrics, episode) for c in criteria]
    all_passed = all(r["passed"] for r in results)
 
    if all_passed:
        reward = 1.0
    elif episode.agent_steps == 0:
        reward = 0.0  # no attempt was made
    else:
        total_weight = sum(r["weight"] for r in results)
        weighted = sum(r["weight"] * r["score"] for r in results) / total_weight
        primaries = [r for r in results if r["primary"]]
        completion = sum(r["score"] for r in primaries) / len(primaries) if primaries else 1.0
        reward = MAX_PARTIAL_REWARD * weighted * completion
        if any(not r["passed"] for r in primaries):
            reward = min(reward, PRIMARY_FAILURE_CAP)
        reward = min(reward, MAX_PARTIAL_REWARD)
 
    reward = round(max(0.0, min(1.0, reward)), 4)
    return {
        "reward": reward,
        "all_passed": all_passed,
        "metrics": metrics,
        "criteria": results,
        "task_id": task_config.get("task_id"),
    }
 
 
def verify(task_config: Dict[str, Any], episode_log: Any, final_state: Any) -> float:
    """Return the episode reward in ``[0.0, 1.0]`` (1.0 = all criteria met)."""
    return float(verify_detailed(task_config, episode_log, final_state)["reward"])
 