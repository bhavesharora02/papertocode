import hashlib, json

def sha256_of_obj(obj) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def sha256_of_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
