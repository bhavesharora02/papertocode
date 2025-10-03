import sys, json
from core.ir.build_ir import build_ir
from core.agents.interpreter import run_interpreter
from core.agents.mapper import run_mapper

def load_parsed(parsed_json_path: str):
    with open(parsed_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def guess_meta(parsed: dict):
    # Minimal meta using parsed fields. Improve later by proper title detection.
    title = parsed.get("sections", {}).get("title") or "Unknown Title"
    domain = "NLP" if "token" in parsed.get("raw_text","").lower() else "CV"
    tasks = ["text_classification"] if domain=="NLP" else ["image_classification"]
    return {"title": title, "arxiv_id": None, "tasks": tasks, "domain": domain}

if __name__ == "__main__":
    parsed_path = sys.argv[1]  # e.g., artifacts/parsed/bert.json
    parsed = load_parsed(parsed_path)
    meta = guess_meta(parsed)
    method_text = parsed.get("sections", {}).get("method") or parsed.get("raw_text", "")[:5000]

    a1 = run_interpreter(meta, method_text)
    a2 = run_mapper({"domain": meta["domain"], **a1})

    ir = build_ir(meta, a1, a2)
    out = parsed_path.replace("artifacts/parsed", "artifacts/ir").replace(".json", "_ir.json")

    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(ir.model_dump_json(indent=2))
    print(f"✅ IR saved at {out}")
