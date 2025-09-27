import json, os, time, uuid
from typing import Any, Dict, List, Optional

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",",":"))

def load_schema(path: str = "data/schema.json") -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)

def timestamp_id(prefix: str = "session") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

def deep_get(d: Dict[str, Any], dotted: str) -> Any:
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def deep_set(d: Dict[str, Any], dotted: str, value: Any):
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def save_session(messages: List[Dict[str, str]], report_json_str: str, out_dir: str = "outputs/sessions"):
    ensure_dir(out_dir)
    row = {
        "id": timestamp_id(),
        "messages": messages,
        "report_json": canonical_json(json.loads(report_json_str))
    }
    with open(os.path.join(out_dir, "latest.jsonl"), "a") as f:
        f.write(canonical_json(row) + "\n")
