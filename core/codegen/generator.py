import os
import json
from pathlib import Path

def generate_repo(ir_path: str, out_dir: str):
    """Generate a runnable repo (model.py, train.py, evaluate.py) from IR."""
    ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === Prepare imports dynamically ===
    mapping = ir.get("mapping", {}).get("nn_modules", {})
    import_lines = ["import torch", "import torch.nn as nn", "import torch.optim as optim"]

    for fqcn in mapping.values():
        if fqcn.startswith("transformers."):
            cls_name = fqcn.split(".")[-1]
            import_lines.append(f"from transformers import {cls_name}")

    # === model.py ===
    model_code = [
        "\n".join(import_lines),
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]

    layers = ir["model"].get("layers", [])
    for i, layer in enumerate(layers):
        ltype = layer["type"]
        params = layer.get("params", {})

        # mapping resolve
        if ltype == "pretrained_backbone":
            fqcn = "transformers.BertForSequenceClassification"
        elif ltype in mapping:
            fqcn = mapping[ltype]
        else:
            fqcn = f"nn.{ltype}"
        cls_name = fqcn.split(".")[-1]


        if fqcn.startswith("transformers."):
            pretrained = ir["model"].get("init", {}).get("pretrained", None)
            if pretrained and "BertForSequenceClassification" in fqcn:
                model_code.append(
                f'        self.{ltype}{i} = {fqcn}.from_pretrained("{pretrained}", num_labels={params.get("out_features", 2)})'
            )
            else:
                model_code.append(f"        self.{ltype}{i} = {fqcn}(**{params})")
        else:
            model_code.append(f"        self.{ltype}{i} = {fqcn}(**{params})")

    model_code += [
        "",
        "    def forward(self, x, labels=None):",
        "        # Forward through layers sequentially",
    ]
    fwd = "x"
    for i, layer in enumerate(layers):
        ltype = layer["type"]
        fwd = f"self.{ltype}{i}({fwd})"
    model_code.append(f"        out = {fwd}")
    model_code.append("        return out")
    (out / "model.py").write_text("\n".join(model_code), encoding="utf-8")

    # === train.py ===
    train_code = f"""import torch
import torch.optim as optim
from model import Model

def get_dummy_data(num=100, input_dim=10, num_classes=2):
    X = torch.randn(num, input_dim)
    y = torch.randint(0, num_classes, (num,))
    return X, y

def train():
    model = Model()
    X, y = get_dummy_data()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {{epoch+1}} | Loss: {{loss.item():.4f}}")

if __name__ == "__main__":
    train()
"""
    (out / "train.py").write_text(train_code, encoding="utf-8")

    # === evaluate.py ===
    eval_code = """def evaluate():
    print("Evaluation placeholder")

if __name__ == "__main__":
    evaluate()
"""
    (out / "evaluate.py").write_text(eval_code, encoding="utf-8")

    print(f"✅ Repo generated at {out}")
