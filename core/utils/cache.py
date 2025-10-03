import os, json
from .hashing import sha256_of_str

CACHE_DIR = "artifacts/cache"

def _key_path(agent_name: str, key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{agent_name}_{key}.json")

def get_cached(agent_name: str, prompt: str):
    key = sha256_of_str(prompt)
    path = _key_path(agent_name, key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def set_cached(agent_name: str, prompt: str, data: dict):
    key = sha256_of_str(prompt)
    path = _key_path(agent_name, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
