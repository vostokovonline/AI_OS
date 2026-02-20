import json
from pathlib import Path

BASE = Path(__file__).parent.parent / "schemas"

def load_schema(name: str):
    path = BASE / name
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    with open(path) as f:
        return json.load(f)
