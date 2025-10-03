import json, os

def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
