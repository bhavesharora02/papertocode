import os, json
from jinja2 import Environment, FileSystemLoader
from core.ir.schema import IR
from datasets import load_dataset

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

def detect_num_classes(ir: IR, default=2):
    try:
        if isinstance(ir.dataset.source, str) and ir.dataset.source.startswith("hf://"):
            name = ir.dataset.source.replace("hf://", "")
            parts = name.split("/")
            if len(parts) == 2:
                ds = load_dataset(parts[0], parts[1], split="train[:1]")
            else:
                ds = load_dataset(name, split="train[:1]")
            label_key = ir.dataset.features.get("label", "label")
            return len(ds.features[label_key].names)
    except Exception as e:
        print(f"⚠️ Could not auto-detect num_classes, defaulting to {default}. Error: {e}")
    return default


def sanitize_layer(layer_type, params, num_classes):
    clean = dict(params)
    key = layer_type.strip().lower()

    # --- Transformer ---
    if key == "transformer":
        mapping = {
            "dim": "d_model",
            "embedding_dim": "d_model",
            "n_heads": "nhead",
            "n_layers": "num_encoder_layers",
            "ffn_activation": "activation"
        }
        for old, new in mapping.items():
            if old in clean:
                clean[new] = clean.pop(old)
        for bad in ["normalization", "pre_normalization", "positional_embeddings", "positional_embedding"]:
            if bad in clean:
                print(f"⚠️ Ignoring unsupported param '{bad}': {clean[bad]}")
                clean.pop(bad)
        if "activation" in clean:
            act = str(clean["activation"]).lower()
            if act not in ["relu", "gelu"]:
                print(f"⚠️ Replacing unsupported activation '{act}' with 'gelu'")
                clean["activation"] = "gelu"

    # --- CNN/ResNet mappings ---
    mapping_layer = {
        "convstem": "Conv2d",
        "conv": "Conv2d",
        "maxpool": "MaxPool2d",
        "maxpool2d": "MaxPool2d",
        "avgpool": "AvgPool2d",
        "avgpool2d": "AvgPool2d",
        "batchnorm": "BatchNorm2d",
        "batchnorm2d": "BatchNorm2d",
        "linear": "Linear"
    }
    if key in mapping_layer:
        layer_type = mapping_layer[key]

    # --- Fix kernel param naming ---
    if "kernel" in clean and "kernel_size" not in clean:
        clean["kernel_size"] = clean.pop("kernel")

    # --- Linear num_classes patch ---
    if "out_features" in clean and clean["out_features"] == "num_classes":
        clean["out_features"] = num_classes

    # --- ResNetStage → Sequential dummy block ---
    if key == "resnetstage":
        num_blocks = clean.get("num_blocks", 2)
        in_channels = clean.get("in_channels", 64)
        out_channels = clean.get("out_channels", 64)
        stride = clean.get("stride", 1)

        print("⚠️ Converting ResNetStage → Sequential dummy block")
        modules = []
        for i in range(num_blocks):
            modules.append(f"nn.Conv2d({in_channels if i==0 else out_channels}, {out_channels}, kernel_size=3, stride={stride if i==0 else 1}, padding=1, bias=False)")
            modules.append(f"nn.BatchNorm2d({out_channels})")
            modules.append("nn.ReLU(inplace=True)")
        layer_type = "Sequential"
        clean = {"_modules": modules}

    # --- ClassificationHead → AdaptiveAvgPool2d + Flatten + Linear ---
    if key == "classificationhead":
        print("⚠️ Converting ClassificationHead → AdaptivePool + Linear")
        in_features = clean.get("in_features", 512)
        out_features = clean.get("num_classes", num_classes)
        layer_type = "Sequential"
        clean = {
            "_modules": [
                "nn.AdaptiveAvgPool2d((1,1))",
                "nn.Flatten()",
                f"nn.Linear({in_features}, {out_features})"
            ]
        }

    return layer_type, clean



def generate_repo(ir: IR, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    env.filters["py"] = lambda x: repr(x)

    # --- Patch pretrained fallback safely ---
    pretrained = None
    if isinstance(ir.model.init, dict):
        pretrained = ir.model.init.get("pretrained")
        if not pretrained or str(pretrained).strip() == "":
            pretrained = "bert-base-uncased"
            ir.model.init["pretrained"] = pretrained
    else:
        pretrained = getattr(ir.model.init, "pretrained", None)
        if not pretrained or str(pretrained).strip() == "":
            pretrained = "bert-base-uncased"
            ir.model.init.pretrained = pretrained

    # --- Detect num_classes ---
    num_classes = detect_num_classes(ir, default=2)
    print(f"✅ Auto-detected num_classes = {num_classes}")

    # --- Collect layers safely ---
    layers = []
    for l in (ir.model.layers if ir.model.layers else []):
        layer = l if isinstance(l, dict) else l.model_dump()
        raw_params = layer.get("params", {})
        if isinstance(raw_params, dict) and "params" in raw_params and isinstance(raw_params["params"], dict):
            raw_params = raw_params["params"]

        layer_type, clean_params = sanitize_layer(layer["type"], raw_params, num_classes)
        layer["type"] = layer_type
        layer["params"] = clean_params
        layers.append(layer)

    if not layers:
        layers = [{"type": "Linear", "params": {"in_features": 10, "out_features": num_classes}}]

    # --- Render templates ---
    model_tpl = env.get_template("model.py.j2")
    model_code = model_tpl.render(layers=layers, pretrained=pretrained, num_labels=num_classes)
    with open(os.path.join(outdir, "model.py"), "w") as f:
        f.write(model_code)

    train_tpl = env.get_template("train.py.j2")
    train_code = train_tpl.render(
        optimizer=ir.training.optimizer.model_dump() if hasattr(ir.training.optimizer, "model_dump") else ir.training.optimizer,
        loss=ir.training.loss,
        epochs=ir.training.epochs
    )
    with open(os.path.join(outdir, "train.py"), "w") as f:
        f.write(train_code)

    eval_tpl = env.get_template("evaluate.py.j2")
    with open(os.path.join(outdir, "evaluate.py"), "w") as f:
        f.write(eval_tpl.render())

    print(f"✅ Repo generated at {outdir}")
