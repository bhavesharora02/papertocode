# core/codegen/codegen.py
import os, json
from jinja2 import Environment, FileSystemLoader
from core.ir.schema import IR
from core.utils.dataset_utils import safe_load_dataset

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# ============================================================
# Auto-detect num_classes
# ============================================================
def detect_num_classes(ir: IR, default=2):
    """Detect number of output classes (ASCII-safe)."""
    try:
        if isinstance(ir.dataset.source, str):
            if ir.dataset.source.startswith("hf://"):
                name = ir.dataset.source.replace("hf://", "")
            else:
                name = ir.dataset.source

            try:
                ds = safe_load_dataset(name, split="train[:1]")
            except Exception:
                print(f"Warning: Dataset {name} gated -> falling back to cifar10")
                ds = safe_load_dataset("cifar10", split="train[:1]")

            label_key = ir.dataset.features.get("label", "label")
            if hasattr(ds, "features") and label_key in ds.features:
                feat = ds.features[label_key]
                if hasattr(feat, "names") and isinstance(feat.names, list):
                    num_classes = len(feat.names)
                    print(f"OK: Auto-detected num_classes = {num_classes}")
                    return num_classes
    except Exception as e:
        print(f"Warning: Could not auto-detect num_classes, defaulting to {default}. Error: {e}")
    return default


# ============================================================
# Layer Sanitizer
# ============================================================
def sanitize_layer(layer_type, params, num_classes):
    """Clean, normalize, and standardize layer parameters for code generation."""
    clean = dict(params)
    key = str(layer_type).strip().lower()

    # ---------------- Remove non-param keys ----------------
    bad_keys = ["comment", "comments", "description", "desc",
                "note", "notes", "name", "id", "type", "layer_name"]
    for bad_key in bad_keys:
        if bad_key in clean:
            print(f"Warning: Removing non-param key '{bad_key}' from layer {layer_type}")
            clean.pop(bad_key, None)

    # ============================================================
    # ✅ Pooling layers
    # ============================================================
    if key in ["adaptiveavgpool2d", "avgpool2d", "avgpool"]:
        if not clean or "output_size" not in clean:
            clean["output_size"] = (1, 1)
            print(f"Warning: Applied default output_size=(1,1) for {layer_type}")

    if key in ["maxpooling", "maxpool", "pooling", "pool", "maxpool2d"]:
        print(f"Warning: Converting {layer_type} -> MaxPool2d (paper-style normalization)")
        kernel_size = clean.get("kernel_size", 2)
        stride = clean.get("stride", 2)
        padding = clean.get("padding", 0)
        layer_type = "MaxPool2d"
        clean = {"kernel_size": kernel_size, "stride": stride, "padding": padding}

    if key in ["averagepooling", "avgpooling", "avgpool", "avgpoollayer"]:
        print(f"Warning: Converting {layer_type} -> AvgPool2d (paper-style normalization)")
        kernel_size = clean.get("kernel_size", 2)
        stride = clean.get("stride", 2)
        padding = clean.get("padding", 0)
        layer_type = "AvgPool2d"
        clean = {"kernel_size": kernel_size, "stride": stride, "padding": padding}

    # ============================================================
    # ✅ Convolutional layers
    # ============================================================
    if key in ["conv", "conv2d", "convstem"]:
        src_name = layer_type
        if "kernel_size" not in clean: clean["kernel_size"] = 3
        if "stride" not in clean: clean["stride"] = 1
        if "padding" not in clean: clean["padding"] = 1

        if "in_channels" not in clean:
            if "out_channels" not in clean: clean["out_channels"] = 16
            layer_type = "LazyConv2d"
            print(f"Warning: Using LazyConv2d (auto-infer in_channels) for {src_name}")
        else:
            if "out_channels" not in clean: clean["out_channels"] = 16
            print(f"Warning: Applied safe Conv2d params for {src_name}")

    if key in ["convolutional", "convblock", "conv_layer", "conv3x3"]:
        print(f"Warning: Converting {layer_type} -> Conv2d + ReLU (paper-style normalization)")
        in_channels = clean.get("in_channels", 3)
        out_channels = clean.get("out_channels", 16)
        kernel_size = clean.get("kernel_size", 3)
        stride = clean.get("stride", 1)
        padding = clean.get("padding", 1)
        layer_type = "Sequential"
        clean = {"_modules": [
            f"nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, bias=False)",
            "nn.ReLU(inplace=True)"
        ]}

    # ============================================================
    # ✅ Linear / Dense / FullyConnected
    # ============================================================
    if key in ["linear", "dense", "fullyconnected", "hiddenlayer", "fc"]:
        out_features = clean.get("out_features", num_classes)
        in_features = clean.get("in_features", None)
        layer_type = "Sequential"
        if in_features is None:
            clean = {"_modules": ["nn.Flatten()", f"nn.LazyLinear({out_features})"]}
            print(f"Warning: Using LazyLinear (auto-infer in_features) for {layer_type}")
        else:
            clean = {"_modules": ["nn.Flatten()", f"nn.Linear({in_features}, {out_features})"]}
            print(f"Warning: Wrapping {layer_type} with Flatten to ensure 2D input")

    # ============================================================
    # ✅ Dropout / Flatten / BatchNorm / Activation
    # ============================================================
    if key in ["dropout", "dropoutlayer"]:
        print(f"Warning: Normalizing {layer_type} -> Dropout(p=0.5)")
        layer_type = "Dropout"
        clean = {"p": clean.get("p", 0.5)}

    if key in ["flatten", "reshape"]:
        print(f"Warning: Normalizing {layer_type} -> Flatten()")
        layer_type = "Flatten"
        clean = {}

    if key in ["batchnormalization", "batchnorm", "batchnorm2d"]:
        print(f"Warning: Normalizing {layer_type} -> BatchNorm2d")
        layer_type = "BatchNorm2d"
        clean = {"num_features": clean.get("num_features", 64)}

    if key in ["activation", "relu", "gelu", "sigmoid", "tanh"]:
        act = key.lower()
        act_map = {
            "relu": "ReLU",
            "gelu": "GELU",
            "sigmoid": "Sigmoid",
            "tanh": "Tanh"
        }
        layer_type = act_map.get(act, "ReLU")
        clean = {}
        print(f"Warning: Normalizing {layer_type} activation")

    # ============================================================
    # ✅ MBConv, ConvBNReLU, ResidualStack, Transformer
    # ============================================================
    if key == "mbconv":
        print("Warning: Converting MBConv -> Conv2d + BatchNorm + ReLU block")
        in_channels = clean.get("in_channels")
        out_channels = clean.get("out_channels", 16)
        kernel_size = clean.get("kernel_size", 3)
        layer_type = "Sequential"
        conv = (f"nn.LazyConv2d({out_channels}, kernel_size={kernel_size}, padding=1, bias=False)"
                if in_channels is None else
                f"nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, padding=1, bias=False)")
        clean = {"_modules": [conv, f"nn.BatchNorm2d({out_channels})", "nn.ReLU(inplace=True)"]}

    if key in ["convbnrelu", "conv_bn_relu"]:
        print(f"Warning: Expanding {layer_type} -> Conv2d + BatchNorm2d + ReLU block")
        in_channels = clean.get("in_channels", 3)
        out_channels = clean.get("out_channels", 64)
        kernel_size = clean.get("kernel_size", 3)
        stride = clean.get("stride", 1)
        padding = clean.get("padding", 1)
        layer_type = "Sequential"
        clean = {"_modules": [
            f"nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, bias=False)",
            f"nn.BatchNorm2d({out_channels})",
            "nn.ReLU(inplace=True)"
        ]}

    if key in ["residualstack", "residualblock", "resnetstage"]:
        num_blocks = int(clean.get("num_blocks", 2))
        in_channels = int(clean.get("in_channels", 64))
        out_channels = int(clean.get("out_channels", max(in_channels, 64)))
        stride = int(clean.get("stride", 1))
        kernel_size = int(clean.get("kernel_size", 3))
        print(f"Warning: Expanding {layer_type} -> {num_blocks}x [Conv2d+BN+ReLU] (no skip-conn)")
        modules = []
        for i in range(num_blocks):
            ic = in_channels if i == 0 else out_channels
            st = stride if i == 0 else 1
            modules += [
                f"nn.Conv2d({ic}, {out_channels}, kernel_size={kernel_size}, stride={st}, padding=1, bias=False)",
                f"nn.BatchNorm2d({out_channels})",
                "nn.ReLU(inplace=True)"
            ]
        layer_type = "Sequential"
        clean = {"_modules": modules}

    if key == "transformer":
        mapping = {"dim": "d_model", "embedding_dim": "d_model", "n_heads": "nhead", "n_layers": "num_encoder_layers", "ffn_activation": "activation"}
        for old, new in mapping.items():
            if old in clean: clean[new] = clean.pop(old)
        for bad in ["normalization", "pre_normalization", "positional_embeddings", "positional_embedding"]:
            clean.pop(bad, None)
        if "activation" in clean:
            act = str(clean["activation"]).lower()
            if act not in ["relu", "gelu"]:
                print(f"Warning: Replacing unsupported activation '{act}' with 'gelu'")
                clean["activation"] = "gelu"

    # ============================================================
    # ✅ Tree / Ensemble fallback
    # ============================================================
    if "kernel" in clean and "kernel_size" not in clean:
        clean["kernel_size"] = clean.pop("kernel")
    if "out_features" in clean and clean["out_features"] == "num_classes":
        clean["out_features"] = num_classes

    tree_keywords = ["xgboost", "lightgbm", "catboost", "randomforest", "gbdt", "adaboost", "extratrees"]
    if any(k in key for k in tree_keywords):
        print(f"Warning: Replacing {layer_type} with Dummy Linear Wrapper (detected ensemble keyword: {key})")
        layer_type = "Sequential"
        clean = {"_modules": ["nn.Flatten()", "nn.LazyLinear(128)", "nn.ReLU()", f"nn.Linear(128, {num_classes})"]}

    return layer_type, clean


# ============================================================
# Repo Generator
# ============================================================
def generate_repo(ir: IR, outdir: str):
    """Generate model, train, and evaluate scripts from IR safely."""
    os.makedirs(outdir, exist_ok=True)
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    env.filters["py"] = lambda x: repr(x)

    pretrained = None
    if isinstance(ir.model.init, dict):
        pretrained = ir.model.init.get("pretrained") or "bert-base-uncased"
        ir.model.init["pretrained"] = pretrained
    else:
        pretrained = getattr(ir.model.init, "pretrained", None) or "bert-base-uncased"
        ir.model.init.pretrained = pretrained

    num_classes = detect_num_classes(ir, default=2)
    print(f"OK: Auto-detected num_classes = {num_classes}")

    layers = []
    for l in (ir.model.layers or []):
        layer = l if isinstance(l, dict) else l.model_dump()
        raw_params = layer.get("params", {})
        if isinstance(raw_params, dict) and "params" in raw_params and isinstance(raw_params["params"], dict):
            raw_params = raw_params["params"]
        layer_type, clean_params = sanitize_layer(layer["type"], raw_params, num_classes)
        layer["type"], layer["params"] = layer_type, clean_params
        layers.append(layer)

    if not layers:
        layers = [{"type": "Linear", "params": {"in_features": 10, "out_features": num_classes}}]

    # Normalize loss
    loss_map = {
        "squarederror": "MSELoss", "l2loss": "MSELoss", "l1loss": "L1Loss",
        "huberloss": "SmoothL1Loss", "crossentropy": "CrossEntropyLoss",
        "binarycrossentropy": "BCELoss", "hingeloss": "HingeEmbeddingLoss",
        "crossentropyloss": "CrossEntropyLoss", "mseloss": "MSELoss"
    }
    if hasattr(ir.training, "loss") and isinstance(ir.training.loss, str):
        normalized = ir.training.loss.strip().lower().replace("_", "")
        ir.training.loss = loss_map.get(normalized, "MSELoss")
        print(f"Warning: Mapping non-Torch loss '{normalized}' -> '{ir.training.loss}'")

    # Write templates
    model_tpl = env.get_template("model.py.j2")
    with open(os.path.join(outdir, "model.py"), "w") as f:
        f.write(model_tpl.render(layers=layers, pretrained=pretrained, num_labels=num_classes))

    train_tpl = env.get_template("train.py.j2")
    with open(os.path.join(outdir, "train.py"), "w") as f:
        f.write(train_tpl.render(
            optimizer=ir.training.optimizer.model_dump() if hasattr(ir.training.optimizer, "model_dump") else ir.training.optimizer,
            loss=ir.training.loss, epochs=ir.training.epochs
        ))

    eval_tpl = env.get_template("evaluate.py.j2")
    with open(os.path.join(outdir, "evaluate.py"), "w") as f:
        f.write(eval_tpl.render())

    print(f"OK: Repo generated at {outdir}")
