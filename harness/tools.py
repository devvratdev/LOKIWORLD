"""Tool definitions shown to the model, plus argument checking.
 
The four trading tools come straight from env/tools.py. The harness only adds `finish`,
which lets the model end the episode itself.
"""
from env.tools import TOOLS as ENV_TOOLS
 
FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call this when you are done trading and the task is complete. Ends the episode.",
        "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}
 
TOOLS = [*ENV_TOOLS, FINISH_TOOL]
SPECS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TOOL_NAMES = list(SPECS)
 
 
def tools_for(task):
    """The tools passed to the model for one task: its allowed env tools, then `finish`."""
    allowed = set(task.tool_names)
    return [t for t in ENV_TOOLS if t["function"]["name"] in allowed] + [FINISH_TOOL]
 
 
_PY_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}
 
 
def _type_ok(value, type_name):
    if type_name == "null":
        return value is None
    if isinstance(value, bool):                     # bool is an int in Python; JSON keeps them apart
        return type_name == "boolean"
    return isinstance(value, _PY_TYPES[type_name])
 
 
def _check_value(key, value, schema):
    types = schema.get("type")
    types = [types] if isinstance(types, str) else list(types or [])
    if types and not any(_type_ok(value, t) for t in types):
        return f"argument '{key}' must be of type {' or '.join(types)}"
    if "enum" in schema and value not in schema["enum"]:
        return f"argument '{key}' must be one of {schema['enum']}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"argument '{key}' must be >= {schema['minimum']}"
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return f"argument '{key}' must be > {schema['exclusiveMinimum']}"
    return None
 
 
def check_args(name, args):
    """Return an error message if `args` don't match the tool's schema, else None."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    spec = SPECS[name]
    properties = spec.get("properties", {})
    for key in spec.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    for key, value in args.items():
        if key not in properties:
            if spec.get("additionalProperties") is False:
                return f"unexpected argument '{key}'"
            continue
        problem = _check_value(key, value, properties[key])
        if problem:
            return problem
    return None
 