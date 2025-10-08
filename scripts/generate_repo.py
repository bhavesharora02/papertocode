import sys, json, os, torch
from core.ir.schema import IR
from core.codegen.codegen import generate_repo

def safe_print(msg):
    """Ensure all prints are ASCII-safe (avoid UnicodeEncodeError on Windows)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="ignore").decode())

def make_synthetic_dataset():
    """Fallback synthetic dataset with 10 random samples."""
    X = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 2, (10,))
    safe_print("? Using synthetic random dataset (offline fallback).")
    return {"X": X, "y": y}


if __name__ == "__main__":
    ir_path = sys.argv[1]
    with open(ir_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- ? Sanitize nested target_metrics ---
    if "training" in data and isinstance(data["training"].get("target_metrics"), dict):
        fixed_metrics = {}
        for k, v in data["training"]["target_metrics"].items():
            if isinstance(v, dict):
                # flatten nested metrics like {"top-1 accuracy": 0.843}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        key_name = sub_k.lower().replace(" ", "_").replace("-", "_")
                        fixed_metrics[key_name] = sub_v
            elif isinstance(v, (int, float)):
                key_name = k.lower().replace(" ", "_").replace("-", "_")
                fixed_metrics[key_name] = v
        data["training"]["target_metrics"] = fixed_metrics

    # --- Ensure model structure exists ---
    data.setdefault("model", {})
    data["model"].setdefault("init", {})

    # --- Ensure pretrained field is defined ---
    if not data["model"]["init"].get("pretrained"):
        data["model"]["init"]["pretrained"] = "bert-base-uncased"

    # --- Auto-detect num_classes if dataset info exists ---
    num_labels = 2  # safe default
    if "dataset" in data and isinstance(data["dataset"], dict):
        label_key = data["dataset"].get("features", {}).get("label")
        name = data["dataset"].get("source", "dummy")

        if label_key:
            try:
                from datasets import load_dataset
                clean_name = name.replace("hf://", "")
                parts = clean_name.split("/")

                try:
                    if len(parts) == 2:
                        ds = load_dataset(parts[0], parts[1], split="train[:1]")
                    else:
                        ds = load_dataset(clean_name, split="train[:1]")
                except Exception as e:
                    safe_print(f"Warning: Could not load dataset '{name}'. Error: {e}")
                    safe_print("?? Falling back to CIFAR-10 for dummy verification...")
                    from datasets import load_dataset as _ld
                    try:
                        ds = _ld("cifar10", split="train[:1]")
                    except Exception:
                        ds = make_synthetic_dataset()

                # Check label availability
                if hasattr(ds, "features") and label_key in ds.features:
                    label_feature = ds.features[label_key]
                    if hasattr(label_feature, "num_classes"):
                        num_labels = label_feature.num_classes
                    else:
                        num_labels = len(set(ds[label_key]))
                else:
                    safe_print(f"Warning: Label '{label_key}' missing in fallback dataset. Defaulting num_classes=2.")
                    num_labels = 2

                safe_print(f"OK: Auto-detected num_classes = {num_labels}")
            except Exception as e:
                safe_print(f"Warning: Could not auto-detect num_classes, defaulting to 2. Error: {e}")
        else:
            safe_print("Warning: No label feature found, defaulting num_classes = 2.")
    else:
        safe_print("Warning: Dataset info missing, defaulting num_classes = 2.")

    data["model"]["num_labels"] = num_labels

    # --- Training defaults ---
    if "training" not in data:
        data["training"] = {}

    if not data["training"].get("loss"):
        safe_print("Warning: Missing loss, defaulting to 'MSELoss'")
        data["training"]["loss"] = "MSELoss"

    if "optimizer" not in data["training"]:
        data["training"]["optimizer"] = {"name": "Adam", "lr": 0.001}
    else:
        opt = data["training"]["optimizer"]
        if not opt.get("name"):
            safe_print("Warning: Missing optimizer name, defaulting to 'Adam'")
            opt["name"] = "Adam"
        if not opt.get("lr"):
            safe_print("Warning: Missing learning rate, defaulting to 0.001")
            opt["lr"] = 0.001

    # --- Build IR object and generate repo ---
    ir = IR(**data)

    if len(sys.argv) > 2:
        outdir = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(ir_path))[0]
        outdir = os.path.join("artifacts", "repos", base)

    generate_repo(ir, outdir)
    safe_print(f"OK: Repo generated at {outdir}")
