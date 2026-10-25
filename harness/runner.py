"""Episode loop, baseline runner, and telemetry."""
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
 
from .env import PASS_SCORE, STEP_TOOLS
from .tools import check_args, tools_for
 
SYSTEM_PROMPT = (
    "You are an autonomous quantitative trading agent working in a simulated limit-order-book market. "
    "Use the provided tools to inspect the market and place orders. "
    "Always respond with tool calls. When the task is complete, call the `finish` tool."
)
TRIMMED = "[older tool output removed to save context]"
TRIMMED_ARGS = '{"note": "arguments removed to save context"}'
 
 
@dataclass
class Limits:
    max_steps: int = 40            # one step = one model reply
    max_seconds: int = 600         # clock starts at the first model request
    max_tool_calls: int = 10       # per reply; extra calls are not run
    context_budget_tokens: int = 48000
    max_output_chars: int = 4000
 
 
class JsonlLog:
    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
 
    def write(self, record):
        self.file.write(json.dumps(record, default=str) + "\n")
 
    def close(self):
        self.file.close()
 
 
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
 
 
def show(record):
    """One readable line per step on the terminal (live telemetry)."""
    tools = ",".join(a.get("tool", "?") for a in record["parsed_actions"]) or "-"
    reward = record["reward"].get("total") if record.get("reward") else None
    print(f"[{record['agent']:<6} step {record['step']:>2}] tools={tools:<40} "
          f"reward={reward if reward is not None else '-':<6} state={record['state']['hash']} "
          f"errors={len(record['errors'])}", flush=True)
 
 
# --------------------------------------------------------- context trimming
def estimate_tokens(messages):
    return len(json.dumps(messages, default=str)) // 4
 
 
def trim_history(messages, budget):
    """Shrink old content until the conversation fits the token budget.
 
    First old tool outputs are replaced by a short note, then the (possibly large)
    arguments of old tool calls. The latest assistant turn and its results are
    never touched, so the agent always sees what it just did.
    """
    while estimate_tokens(messages) > budget:
        last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=0)
        victim = next((m for i, m in enumerate(messages)
                       if m["role"] == "tool" and m["content"] != TRIMMED and i < last_assistant), None)
        if victim is not None:
            victim["content"] = TRIMMED
            continue
        shrunk = False
        for m in messages[:last_assistant]:
            for call in m.get("tool_calls", []):
                if call["function"]["arguments"] != TRIMMED_ARGS:
                    call["function"]["arguments"] = TRIMMED_ARGS
                    shrunk = True
                    break
            if shrunk:
                break
        if not shrunk:
            return          # nothing older left to shrink
 
 
def _is_rejected(result):
    return result.get("status") == "rejected" or result.get("success") is False
 
 
# ------------------------------------------------------------ baselines
def run_scripted(env, task_id, seed, script, agent, log):
    """Run a scripted agent (oracle or no-op) through the same Env code path as a model.
 
    `script` is a function(env, task) that plays the episode by calling the env's tool methods.
    One telemetry record is written per agent step (submit_order / cancel_order).
    """
    env.reset(task_id, seed)
    counter = {"step": 0}
 
    def on_entry(entry):
        if entry["tool"] not in STEP_TOOLS:
            return
        counter["step"] += 1
        result = entry["result"]
        record = {"timestamp": now(), "agent": agent, "task": task_id, "step": counter["step"], "seed": env.task.seed,
                  "model_input": None, "model_output_raw": None,
                  "parsed_actions": [{"tool": entry["tool"], "args": entry["arguments"]}],
                  "tool_result": [json.dumps(result)],
                  "errors": [{"kind": "tool_error", "detail": json.dumps(result)}] if _is_rejected(result) else [],
                  "state": {"hash": env.state_hash()}, "reward": env.score(), "tokens": 0, "latency_ms": 0}
        log.write(record)
        show(record)
 
    env.listener = on_entry
    try:
        script(env, env.task)
    finally:
        env.listener = None
    env.finalize()                  # fire any end-of-episode scenario event before the final score
    final = env.score()
    log.write({"timestamp": now(), "agent": agent, "task": task_id, "event": "episode_end",
               "seed": env.task.seed, "steps": counter["step"], "reward": final, "episode_log": env.log})
    return final
 
 
# ---------------------------------------------------------- model episode
def run_episode(env, task_id, seed, client, append_text, limits, log):
    prompt = env.reset(task_id, seed)
    client.start(env.task)
    tools = tools_for(env.task)
    allowed = [t["function"]["name"] for t in tools]
    system = SYSTEM_PROMPT + (("\n\n" + append_text) if append_text else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    started, termination, finished, steps = None, None, False, 0
 
    for step in range(1, limits.max_steps + 1):
        if started is not None and time.monotonic() - started >= limits.max_seconds:
            termination = "time_cap"
            break
        if started is None:
            started = time.monotonic()                      # the clock starts here
        trim_history(messages, limits.context_budget_tokens)
        sent = copy.deepcopy(messages)
        remaining = limits.max_seconds - (time.monotonic() - started)
        call_start = time.monotonic()
        steps = step
        record = {"timestamp": now(), "agent": "model", "task": task_id, "step": step, "seed": env.task.seed,
                  "model_input": sent, "model_output_raw": None, "parsed_actions": [], "tool_result": [],
                  "errors": [], "state": {"hash": env.state_hash()}, "reward": None, "tokens": 0, "latency_ms": 0}
 
        try:
            reply = client.complete(sent, tools, timeout=max(5, min(120, remaining)))
        except Exception as exc:    # includes provider errors and context-window overflow
            record["errors"].append({"kind": "model_call_failed", "detail": f"{type(exc).__name__}: {exc}"})
            log.write(record)
            show(record)
            termination = "crash"
            break
        record["latency_ms"] = int((time.monotonic() - call_start) * 1000)
        record["model_output_raw"], record["tokens"] = reply.raw, reply.tokens
 
        # keep every tool call in the history so each one gets a matching result
        assistant = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": c.history_arguments()}}
                                       for c in reply.tool_calls]
        messages.append(assistant)
 
        if not reply.tool_calls:
            record["errors"].append({"kind": "no_tool_call", "detail": "reply contained no tool call"})
            messages.append({"role": "user", "content": "No tool call was received. Call one of the "
                             "available tools, or call `finish` if you are done."})
 
        for index, call in enumerate(reply.tool_calls):
            args, parse_error = call.parse()
            record["parsed_actions"].append({"tool": call.name, "args": args, "parse_error": parse_error})
            if finished:
                result = "not run: the episode already finished"
            elif index >= limits.max_tool_calls:
                result = f"not run: at most {limits.max_tool_calls} tool calls are allowed per reply"
                record["errors"].append({"kind": "too_many_tool_calls", "detail": call.name})
            elif parse_error:
                result = f"error: {parse_error}"
                record["errors"].append({"kind": "unparsable_arguments", "detail": call.name})
            elif call.name not in allowed:
                result = f"error: unknown tool '{call.name}'. Available tools: {', '.join(allowed)}"
                record["errors"].append({"kind": "unknown_tool", "detail": call.name})
            elif (problem := check_args(call.name, args)):
                result = f"error: {problem}"
                record["errors"].append({"kind": "invalid_arguments", "detail": f"{call.name}: {problem}"})
            elif call.name == "finish":
                finished, result = True, "Episode finished."
            else:
                result, is_error = env.step(call.name, args)
                if is_error:
                    record["errors"].append({"kind": "tool_error", "detail": result})
            result = result[:limits.max_output_chars]
            record["tool_result"].append(result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
 
        record["state"] = {"hash": env.state_hash()}
        record["reward"] = env.score()                      # running score from verify()
        log.write(record)
        show(record)
        if finished:
            break
    else:
        termination = "step_cap"
 
    try:
        env.finalize()                                      # pending end-of-episode events fire here
        final = env.score()
    except Exception as exc:
        final = {"total": None, "error": str(exc)}
        termination = termination or "crash"
    if termination is None:
        termination = "success" if (final["total"] or 0) >= PASS_SCORE else "failure"
    log.write({"timestamp": now(), "agent": "model", "task": task_id, "event": "episode_end", "seed": env.task.seed,
               "steps": steps, "termination": termination, "reward": final, "episode_log": env.log,
               "elapsed_s": round(time.monotonic() - started, 1) if started else 0})
    return {"termination": termination, "steps": steps, "reward": final,
            "elapsed_s": round(time.monotonic() - started, 1) if started else 0}
 