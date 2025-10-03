import sys, json, os
from core.ir.schema import IR
from core.verify.verifier import verify_repo
from datasets import load_dataset

def patch_num_classes(ir_dict):
    """
    Replace 'num_classes' placeholder in IR with actual dataset label count.
    If dataset is dummy (or LLaMA pretraining mix), skip auto-detect.
    """
    try:
        if ir_dict["dataset"]["source"] == "hf://dummy":
            print("⚠️ Dummy dataset detected, skipping num_classes auto-patch. Forcing = 2")
            ir_dict["model"]["num_labels"] = 2
            return

        if ir_dict["dataset"]["source"].startswith("hf://"):
            name = ir_dict["dataset"]["source"].replace("hf://", "")
            parts = name.split("/")
            if len(parts) == 2:
                ds = load_dataset(parts[0], parts[1], split="train[:1]")  # just 1 sample
            else:
                ds = load_dataset(name, split="train[:1]")

            label_col = ir_dict["dataset"]["features"]["label"]
            num_classes = len(ds.features[label_col].names)

            # patch model layers
            for layer in ir_dict["model"]["layers"]:
                if "params" in layer and "out_features" in layer["params"]["params"]:
                    if layer["params"]["params"]["out_features"] == "num_classes":
                        layer["params"]["params"]["out_features"] = num_classes
            ir_dict["model"]["num_labels"] = num_classes
            print(f"✅ Patched num_classes = {num_classes}")
    except Exception as e:
        print(f"⚠️ Could not patch num_classes automatically: {e}")
        ir_dict["model"]["num_labels"] = 2  # safe fallback

if __name__ == "__main__":
    ir_path = sys.argv[1]   # e.g. artifacts/ir/bert_ir.json
    repo_dir = sys.argv[2]  # e.g. artifacts/repos/bert

    with open(ir_path, "r") as f:
        data = json.load(f)

    # patch before creating IR object
    patch_num_classes(data)

    ir = IR(**data)

    # ⚡ Ensure reports dir always exists
    os.makedirs("artifacts/reports", exist_ok=True)

    if len(sys.argv) > 3:
        report_path = sys.argv[3]
    else:
        base = os.path.splitext(os.path.basename(ir_path))[0]
        report_path = os.path.join("artifacts/reports", base + "_verify.json")

    print(f"⚡ Starting verification for {repo_dir}, report → {report_path}")

    # Force dummy dataset for LLaMA/dummy
    if ir.dataset.source == "hf://dummy" or ir.model.family.lower() == "llama":
        print("⚠️ Forcing dummy dataset for verification (LLaMA/dummy case).")

    verify_repo(ir, repo_dir, report_path)
    print("✅ Verification finished")
