# core/auto_repair/rules.py
import re, os
from .utils import patch_model_insert_flatten, map_ir_loss_optimizer

def remove_unicode_from_verify_repo(ir_path, repo_dir):
    """Remove problematic Unicode symbols (⚠️, ✅, →, etc.) from all scripts including generate_repo.py."""
    changed = False
    paths_to_clean = []

    # 1️⃣ Always include key scripts
    for base in ["scripts/verify_repo.py", "scripts/generate_repo.py", "scripts/codegen.py"]:
        if os.path.exists(base):
            paths_to_clean.append(base)

    # 2️⃣ Include all other .py scripts under /scripts
    script_dir = "scripts"
    if os.path.exists(script_dir):
        for fname in os.listdir(script_dir):
            if fname.endswith(".py"):
                paths_to_clean.append(os.path.join(script_dir, fname))

    # 3️⃣ Include model file (some may log emojis)
    model_py = os.path.join(repo_dir, "model.py")
    if os.path.exists(model_py):
        paths_to_clean.append(model_py)

    # 4️⃣ Perform cleaning
    for path in set(paths_to_clean):
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()

            replacements = {
                "⚠️": "Warning:",
                "✅": "OK:",
                "▶️": "Run:",
                "→": "->",
                "➜": "->",
                "↠": "->",
                "🔍": "DEBUG:",
                "❌": "Error:",
                "✔️": "OK:",
                "💡": "Note:",
            }
            for bad, good in replacements.items():
                src = src.replace(bad, good)

            cleaned = ''.join(ch if ord(ch) < 128 else '?' for ch in src)

            with open(path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            changed = True
        except Exception as e:
            print(f"Warning: Could not clean Unicode in {path}: {e}")

    return changed



class RepairActionResult(dict):
    # convenience
    pass

class RepairRule:
    def __init__(self, name, pattern, action, regenerate_repo=False):
        self.name = name
        self.pattern = re.compile(pattern, re.I | re.M | re.S)
        self.action = action
        self.regenerate_repo = regenerate_repo

    def matches(self, error_text):
        return bool(self.pattern.search(error_text))

    def run(self, *, ir_path, repo_dir):
        changed = self.action(ir_path=ir_path, repo_dir=repo_dir)
        return RepairActionResult({
            "rule": self.name,
            "applied": bool(changed),
            "regenerate_repo": self.regenerate_repo
        })

# ——— actions ———

def fix_loss_and_optimizer(ir_path, repo_dir):
    # map odd losses/optimizers, and enforce CE for multi-class
    return map_ir_loss_optimizer(ir_path, prefer_ce_when_multiclass=True)

def insert_flatten_before_linear(ir_path, repo_dir):
    model_py = os.path.join(repo_dir, "model.py")
    if not os.path.exists(model_py):
        return False
    return patch_model_insert_flatten(model_py)

def force_tree_fallback_regenerate(ir_path, repo_dir):
    # we rely on your codegen.py already replacing tree models → Sequential
    # triggering regeneration is enough
    return True

# ——— rule set ———
# ——— rule set ———
RULES = [
    # Unicode encoding error (Windows console)
    RepairRule(
        name="remove_unicode_from_verify_repo",
        pattern=r"UnicodeEncodeError.*charmap",
        action=remove_unicode_from_verify_repo,
        regenerate_repo=False,
    ),

    # shape mismatch for Linear
    RepairRule(
        name="insert_flatten_for_linear",
        pattern=r"mat1 and mat2 shapes cannot be multiplied|Expected 2D tensor|Expected (?:3D|4D) input to conv2d, but got",
        action=insert_flatten_before_linear,
        regenerate_repo=False,
    ),

    # MSE vs CrossEntropy issues (target/input size mismatch or warning)
    RepairRule(
        name="map_loss_optimizer_and_force_ce",
        pattern=r"Using a target size.*different to the input size|must match the size of tensor|no attribute 'SecondOrderApproximatedLoss'",
        action=fix_loss_and_optimizer,
        regenerate_repo=True,  # IR changed => regenerate repo
    ),

    # unknown tree/boosting layer → ensure generate does fallback
    RepairRule(
        name="tree_model_fallback",
        pattern=r"has no attribute 'XGBoost'|DecisionTree|RandomForest|GradientBoosting|CatBoost|LightGBM",
        action=force_tree_fallback_regenerate,
        regenerate_repo=True,
    ),
]
