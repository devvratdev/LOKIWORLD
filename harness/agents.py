"""Agents: the model client, two offline mock models, and the baselines."""
import json
from dataclasses import dataclass, field
 
from env.reference import run_reference
 
try:
    from env.noop import run_noop as _env_noop      # optional: your own no-op baseline
except ImportError:
    _env_noop = None
 
 
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw text exactly as the model sent it
 
    def parse(self):
        """Return (args, error). `args` is a dict when the arguments are valid JSON."""
        try:
            args = json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError as exc:
            return None, f"arguments are not valid JSON ({exc.msg})"
        if not isinstance(args, dict):
            return None, "arguments must be a JSON object"
        return args, None
 
    def history_arguments(self):
        """Arguments as stored in the conversation. Invalid JSON is replaced with {} so
        the next request is still accepted by every provider."""
        args, error = self.parse()
        return json.dumps(args) if error is None else "{}"
 
 
@dataclass
class Reply:
    text: str | None
    tool_calls: list = field(default_factory=list)
    raw: str = ""
    tokens: int = 0
 
 
# ------------------------------------------------------------ real models
class OpenAIClient:
    """Calls any model through an OpenAI-compatible endpoint.
 
    Judges and teams are given a base_url, an api_key and a model name; the harness reads
    them from the environment variables BASE_URL, API_KEY and MODEL. The same call is used
    for every model, so switching models only means changing those values.
    """
 
    def __init__(self, model, api_key, base_url=None, max_tokens=None):
        import openai
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url or None, max_retries=2)
        self.model, self.max_tokens = model, max_tokens
 
    def start(self, task):
        pass
 
    def complete(self, messages, tools, timeout):
        kwargs = dict(model=self.model, messages=messages, tools=tools, timeout=timeout)
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens      # optional: set MAX_TOKENS to cap each reply
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = []
        for index, call in enumerate(message.tool_calls or []):
            function = getattr(call, "function", None)
            calls.append(ToolCall(id=call.id or f"call_{index}",
                                  name=(function.name if function else "") or "",
                                  arguments=(function.arguments if function else "") or ""))
        usage = getattr(response, "usage", None)
        return Reply(text=message.content, tool_calls=calls, raw=response.model_dump_json(),
                     tokens=getattr(usage, "total_tokens", 0) or 0)
 
 
# ------------------------------------------------------------ mock models
MOCK_CALLS_PER_REPLY = 8        # stays below Limits.max_tool_calls
 
 
def oracle_calls(task):
    """The reference solution as the (tool, args) order calls it makes.
 
    run_reference reads the book, so it is dry-run on a scratch Env (same task, same seed,
    same setup and events) and the orders it placed are recorded. Replaying them on a fresh
    episode gives the identical result, because reads never change the state.
    """
    from .env import Env, STEP_TOOLS                # local import: env.py does not import agents
    scratch = Env()
    scratch.reset(task.task_id, task.seed)
    run_reference(scratch, task.config)
    return [(e["tool"], e["arguments"]) for e in scratch.log
            if e.get("tool") in STEP_TOOLS and not e.get("setup")]
 
 
class MockClient:
    """Offline stand-ins so the template runs with no API key.
 
    mock/solver  replays the reference solution as tool calls, then finishes.
    mock/idle    finishes immediately without doing anything.
    """
 
    def __init__(self, kind):
        self.kind = kind
        self.turn = 0
        self.task = None
        self.queue = []
 
    def start(self, task):
        self.task, self.turn = task, 0
        self.queue = oracle_calls(task) if self.kind == "solver" else []
 
    def complete(self, messages, tools, timeout):
        self.turn += 1
        calls = []
        if self.kind == "solver":
            if self.turn == 1:
                calls.append(("get_portfolio", {}))
            room = MOCK_CALLS_PER_REPLY - len(calls)
            calls += self.queue[:room]
            self.queue = self.queue[room:]
        if not self.queue:
            calls.append(("finish", {}))
        tool_calls = [ToolCall(f"mock-{self.turn}-{i}", name, json.dumps(args))
                      for i, (name, args) in enumerate(calls)]
        return Reply(text=None, tool_calls=tool_calls, raw=json.dumps({"mock": self.kind, "turn": self.turn}))
 
 
def make_client(model, api_key, base_url, max_tokens=None):
    if model.startswith("mock/"):
        kind = model.split("/", 1)[1]
        if kind not in ("solver", "idle"):
            raise ValueError("mock models are 'mock/solver' and 'mock/idle'")
        return MockClient(kind)
    if not api_key:
        raise ValueError("API_KEY is not set")
    return OpenAIClient(model, api_key, base_url, max_tokens)
 
 
# -------------------------------------------------------------- baselines
# A baseline is a function(env, task) that plays the whole episode through the Env wrapper.
def run_oracle(env, task):
    """The reference solution (env/reference.py). Must score full marks."""
    return run_reference(env, task.config)
 
 
def run_noop(env, task):
    """An agent that gives up immediately. Must score about zero.
 
    Uses env/noop.py's run_noop(env, task_config) when that module exists, otherwise does nothing.
    """
    if _env_noop is not None:
        return _env_noop(env, task.config)
    return None
 