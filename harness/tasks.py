"""Task registry, loaded from env/tasks.json.
 
Each task carries its own seed, so the same task_id always starts from the same order book.
`build_task` merges the file-level `metric_definitions` and `initial_state` into the task, which
gives the `task_config` that `env/verifier.py` and `env/reference.py` expect.
 
Set TASKS_FILE to read the definitions from somewhere other than <repo>/env/tasks.json.
"""
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
 
from env.tools import TOOL_NAMES as ENV_TOOL_NAMES
 
ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = Path(os.environ.get("TASKS_FILE") or ROOT / "env" / "tasks.json")
 
 
@dataclass
class TaskInstance:
    task_id: str
    title: str
    difficulty: str      # "tier N - tier name", shown in the checks table
    instructions: str    # the task prompt given to the model
    seed: int            # passed to TradingEnv.reset(seed)
    max_steps: int       # order budget stated in the prompt (submit_order / cancel_order calls)
    tool_names: list     # environment tools this task allows
    config: dict         # merged task_config for verify() and run_reference()
 
 
def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"task file not found: {path} (set TASKS_FILE or put it in env/tasks.json)")
    for key in ("tasks", "metric_definitions", "initial_state"):
        if key not in spec:
            raise ValueError(f"{path} is missing the top-level key '{key}'")
    return spec
 
 
_SPEC = _load(TASKS_FILE)
TASKS = {task["task_id"]: task for task in _SPEC["tasks"]}
DEMO_TASK = "demo_task" if "demo_task" in TASKS else next(iter(TASKS))
 
 
def build_task(task_id, seed=None):
    """Return a TaskInstance. `seed=None` uses the seed written in tasks.json."""
    if task_id not in TASKS:
        raise KeyError(f"unknown task '{task_id}'. Available: {', '.join(TASKS)}")
    task = copy.deepcopy(TASKS[task_id])
    if seed is not None:
        task["seed"] = int(seed)
    unknown = [name for name in task["tools"] if name not in ENV_TOOL_NAMES]
    if unknown:
        raise ValueError(f"task '{task_id}' lists tools that env/tools.py does not define: {unknown}")
    config = {**task,
              "metric_definitions": copy.deepcopy(_SPEC["metric_definitions"]),
              "initial_state": copy.deepcopy(_SPEC["initial_state"])}
    return TaskInstance(task_id=task_id, title=task["title"],
                        difficulty=f"tier {task['tier']} - {task['tier_name']}",
                        instructions=task["prompt"], seed=task["seed"], max_steps=task["max_steps"],
                        tool_names=list(task["tools"]), config=config)
 