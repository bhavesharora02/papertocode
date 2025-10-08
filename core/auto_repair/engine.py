# core/auto_repair/engine.py
import json, os
from .rules import RULES
from .utils import run_cmd

FIX_DB = "artifacts/auto_fixes.json"

class AutoRepairEngine:
    def __init__(self, debug=False):
        self.debug = debug
        os.makedirs(os.path.dirname(FIX_DB), exist_ok=True)
        if not os.path.exists(FIX_DB):
            with open(FIX_DB, "w", encoding="utf-8") as f:
                json.dump({"history": []}, f)

    def repair(self, *, error_text, ir_path, repo_dir):
        # 1) try rule-based fixes
        for rule in RULES:
            if rule.matches(error_text):
                res = rule.run(ir_path=ir_path, repo_dir=repo_dir)
                if res.get("applied"):
                    self._log_fix(rule.name, error_text)
                    print(f"🔧 Applied rule: {rule.name}")
                    return {"regenerate_repo": res.get("regenerate_repo", False)}
                else:
                    # Some rules return True regardless (e.g., force regenerate)
                    self._log_fix(rule.name, error_text)
                    print(f"🔧 Triggered rule (no file change needed): {rule.name}")
                    return {"regenerate_repo": res.get("regenerate_repo", False)}

        # 2) (optional) fall back to LLM-based suggestion (stub)
        # NOTE: You can wire your Gemini API here to produce a patch.
        print("ℹ️ No rule matched. (Optional) call Gemini here for a patch suggestion.")
        return None

    def _log_fix(self, rule_name, error_text):
        with open(FIX_DB, "r+", encoding="utf-8") as f:
            db = json.load(f)
            db["history"].append({
                "rule": rule_name,
                "error_tail": "\n".join(error_text.splitlines()[-10:])
            })
            f.seek(0)
            json.dump(db, f, indent=2)
            f.truncate()
