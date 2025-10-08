# core/auto_repair/utils.py
import subprocess, json, re, os

def run_cmd(cmd):
    cmd = [c for c in cmd if c != ""]
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = p.communicate()
    return p.returncode, out, err

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path, s):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def patch_model_insert_flatten(model_py_path):
    """
    Insert nn.Flatten() before the first nn.Linear(...) if not already present.
    Simple text patch, robust enough for our generated template.
    """
    src = read_text(model_py_path)
    if "nn.Flatten()" in src:
        return False

    # find first occurrence of ".Linear(" creation
    lines = src.splitlines()
    new_lines = []
    injected = False
    for i, line in enumerate(lines):
        new_lines.append(line)
        # insert module attribute creation just before first Linear in __init__
        if (not injected) and ("= nn.Linear(" in line or "= nn.Sequential(" in line and "Linear(" in line):
            # add a dedicated flatten module at top-level to keep naming simple
            new_lines.append("        self._auto_flatten = nn.Flatten()")
            injected = True

    src2 = "\n".join(new_lines)

    # now pipe the forward() to call self._auto_flatten right before first layer usage
    if injected and "def forward(" in src2 and "_auto_flatten" in src2:
        src2 = src2.replace("out = x", "out = x\n        out = self._auto_flatten(out)")
        write_text(model_py_path, src2)
        return True
    return False

def map_ir_loss_optimizer(ir_json_path, prefer_ce_when_multiclass=True):
    """
    Load IR, map odd losses/optimizers, and optionally force CE for multi-class.
    Returns True if IR changed.
    """
    ir = load_json(ir_json_path)
    changed = False

    # loss map
    loss_map = {
        "secondorderapproximatedloss": "CrossEntropyLoss",
        "squarederror": "MSELoss",
        "l2loss": "MSELoss",
        "l1loss": "L1Loss",
        "huberloss": "SmoothL1Loss",
        "crossentropy": "CrossEntropyLoss",
        "logloss": "CrossEntropyLoss",
        "binarycrossentropy": "BCELoss",
        "hingeloss": "HingeEmbeddingLoss"
    }
    loss = (ir.get("training", {}).get("loss") or "").strip()
    if loss:
        norm = loss.lower().replace("_", "")
        if norm in loss_map and loss != loss_map[norm]:
            ir["training"]["loss"] = loss_map[norm]
            changed = True

    # force CE when multi-class
    if prefer_ce_when_multiclass:
        num_labels = ir.get("model", {}).get("num_labels", None)
        if isinstance(num_labels, int) and num_labels > 1:
            if ir.get("training", {}).get("loss") != "CrossEntropyLoss":
                ir["training"]["loss"] = "CrossEntropyLoss"
                changed = True

    # optimizer map
    opt_map = {
        "gradientboosting": "Adam", "xgboost": "Adam", "adaboost": "Adam",
        "decisiontree": "Adam", "randomforest": "Adam", "lightgbm": "Adam",
        "catboost": "Adam", "boosting": "Adam", "sgdclassifier": "SGD",
        "gdoptimizer": "SGD"
    }
    opt = (ir.get("training", {}).get("optimizer", {}).get("name") or "").strip()
    if opt:
        norm = opt.lower().replace("_", "")
        if norm in opt_map and opt != opt_map[norm]:
            ir["training"]["optimizer"]["name"] = opt_map[norm]
            changed = True

    if changed:
        save_json(ir_json_path, ir)
    return changed
