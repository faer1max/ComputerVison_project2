"""Accept only explicit recorded-to-published source mappings; never ignore a mismatch."""
from pathlib import Path
import json

def matches_hashes(recorded, current, root):
    if set(recorded) != set(current):
        return False
    path = Path(root) / "assets/source_translation_bridge.json"
    bridge = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": []}
    for key, value in current.items():
        if recorded[key] == value:
            continue
        candidates = [f for f in bridge["files"] if f["path"] == key or ("/" not in key and Path(f["path"]).name == key)]
        if len(candidates) != 1:
            return False
        entry = candidates[0]
        if entry["published_sha256"] != value or recorded[key] not in entry["recorded_sha256"]:
            return False
    return True
