import sys, os, json

def patch_ir(ir_path: str, out_path: str = None):
    with open(ir_path, "r") as f:
        ir = json.load(f)

    # -------- Dataset --------
    if not ir.get("dataset"):
        ir["dataset"] = {}

    # Splits
    if not ir["dataset"].get("splits") or not sum(ir["dataset"]["splits"].values()):
        ir["dataset"]["splits"] = {"train": 0.8, "val": 0.1, "test": 0.1}

    # Features
    feats = ir["dataset"].get("features", {})
    if not feats.get("text_a"):
        feats["text_a"] = "text"
    if feats.get("text_b") is None:
        feats["text_b"] = ""   # always a string
    if not feats.get("label"):
        feats["label"] = "label"
    ir["dataset"]["features"] = feats

    # Dataset source mapping
    name = (ir["dataset"].get("name") or "").lower()
    src = str(ir["dataset"].get("source", "")).lower()
    if not src.startswith("hf://"):
        print(f"⚠️ Invalid dataset source '{ir['dataset'].get('source')}', patching to dummy")
        ir["dataset"]["source"] = "hf://dummy"

    # -------- Model --------
    if not ir.get("model"):
        ir["model"] = {}

    fam = ir["model"].get("family", "").lower()

    # Init
    if not ir["model"].get("init") or not ir["model"]["init"].get("pretrained"):
        if "bert" in fam:
            ir["model"]["init"] = {"pretrained": "bert-base-uncased"}
        elif "resnet" in fam:
            ir["model"]["init"] = {"pretrained": "resnet50"}
        elif "llama" in fam:
            ir["model"]["init"] = {"pretrained": "meta-llama/Llama-2-7b-hf"}
        else:
            ir["model"]["init"] = {"pretrained": None}

    # Num labels
    if "num_labels" not in ir["model"]:
        if "imagenet" in ir["dataset"]["source"]:
            ir["model"]["num_labels"] = 1000
        else:
            ir["model"]["num_labels"] = 2

    # -------- Training --------
    training = ir.get("training", {})
    if not training.get("loss"):
        training["loss"] = "CrossEntropyLoss"
    if not training.get("optimizer"):
        training["optimizer"] = {"name": "AdamW", "lr": 2e-5}
    if not training.get("batch_size") or training["batch_size"] > 1024:
        print(f"⚠️ Resetting batch_size → 32 (was {training.get('batch_size')})")
        training["batch_size"] = 32
    if not training.get("epochs"):
        training["epochs"] = 3
    if not training.get("metrics"):
        training["metrics"] = ["accuracy"]
    if not training.get("tolerance"):
        training["tolerance"] = 0.05
    ir["training"] = training

    # -------- Preprocessing --------
    prep = ir.get("preprocessing", {})
    if not prep.get("tokenizer"):
        if "bert" in fam:
            prep["tokenizer"] = "bert-base-uncased"
        elif "llama" in fam:
            prep["tokenizer"] = "meta-llama/Llama-2-7b-hf"
        else:
            prep["tokenizer"] = ""  # no tokenizer needed (CV models)
    if not prep.get("max_len"):
        prep["max_len"] = 128
    ir["preprocessing"] = prep

    # -------- Clean unsupported params --------
    for layer in ir.get("model", {}).get("layers", []):
        params = layer.get("params", {}).get("params", {})
        for bad_key in [
            "normalization", "pre_normalization", "activation",
            "positional_embeddings", "ffn_activation"
        ]:
            if bad_key in params:
                print(f"⚠️ Removing unsupported param '{bad_key}': {params[bad_key]}")
                params.pop(bad_key, None)

    # -------- Save --------
    if not out_path:
        out_path = ir_path.replace(".json", "_patched.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ir, f, indent=2)

    print(f"✅ Patched IR saved at {out_path}")
    return out_path


if __name__ == "__main__":
    ir_path = sys.argv[1]
    patch_ir(ir_path)
