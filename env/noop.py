from typing import Any, Dict, List

def run_noop(env: Any, task_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A baseline agent that gives up immediately, returning an empty log."""
    # The agent does not call env.submit_order at all.
    return []