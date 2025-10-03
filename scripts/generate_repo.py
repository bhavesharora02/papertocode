import sys, json, os
from core.ir.schema import IR
from core.codegen.codegen import generate_repo

if __name__ == "__main__":
    ir_path = sys.argv[1]  # e.g. artifacts/ir/bert_ir.json
    with open(ir_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Patch: Ensure model dict exists
    if "model" not in data:
        data["model"] = {}

    # Patch: Ensure pretrained is never blank
    if "init" not in data["model"]:
        data["model"]["init"] = {}
    if not data["model"]["init"].get("pretrained"):
        data["model"]["init"]["pretrained"] = "bert-base-uncased"

    # 🔥 Auto-detect num_classes if dataset + label present
    if "dataset" in data and "features" in data["dataset"]:
        label_key = data["dataset"]["features"].get("label")
        if label_key:
            try:
                from datasets import load_dataset
                name = data["dataset"]["source"].replace("hf://", "")
                parts = name.split("/")
                if len(parts) == 2:
                    ds = load_dataset(parts[0], parts[1], split="train[:1]")
                else:
                    ds = load_dataset(name, split="train[:1]")

                label_feature = ds.features[label_key]
                if hasattr(label_feature, "num_classes"):
                    num_labels = label_feature.num_classes
                else:
                    num_labels = len(set(ds[label_key]))

                print(f"✅ Auto-detected num_classes = {num_labels}")
                data["model"]["num_labels"] = num_labels
            except Exception as e:
                print(f"⚠️ Could not auto-detect num_classes, defaulting to 2. Error: {e}")
                data["model"]["num_labels"] = 2
        else:
            data["model"]["num_labels"] = 2
    else:
        data["model"]["num_labels"] = 2

    ir = IR(**data)

    # ✅ Output repo path fix
    if len(sys.argv) > 2:
        outdir = sys.argv[2]  # manual override
    else:
        base = os.path.splitext(os.path.basename(ir_path))[0]
        outdir = os.path.join("artifacts", "repos", base)

    generate_repo(ir, outdir)
