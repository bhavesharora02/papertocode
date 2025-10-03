import torch, json, os
from core.verify.dataset_loader import load_dataset_from_ir, get_dummy_loader
from core.verify.metrics import accuracy
from core.ir.schema import IR
import importlib.util, sys

def load_model_from_repo(repo_dir: str):
    model_path = os.path.join(repo_dir, "model.py")
    spec = importlib.util.spec_from_file_location("model", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    return module.Model

def verify_repo(ir: IR, repo_dir: str, report_path: str):
    print("🔍 DEBUG: Starting verify_repo...")
    print(f"IR model family: {ir.model.family}, layers: {len(ir.model.layers)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = load_model_from_repo(repo_dir)
    model = Model().to(device)

    # ⚡ Dataset loader
    # inside verify_repo(...)
    if ir.dataset.source == "hf://dummy" or ir.model.family.lower() in ["llama", "resnet", "cnn"]:
        print("⚠️ Forcing dummy dataset for CNN/LLaMA case.")
        loader = get_dummy_loader(ir.training.batch_size, is_cnn=True)
    else:
        print("🔍 DEBUG: Loading real dataset...")
        loader = load_dataset_from_ir(ir, subset_fraction=0.02)


    if loader is None or (hasattr(loader, "__len__") and len(loader) == 0):
        print("⚠️ No data loaded, falling back to dummy dataset.")
        loader = get_dummy_loader(ir.training.batch_size)

    # ⚡ Training setup
    loss_fn = getattr(torch.nn, ir.training.loss)()
    optimizer_cls = getattr(torch.optim, ir.training.optimizer.name)
    optimizer = optimizer_cls(model.parameters(), lr=ir.training.optimizer.lr)

    logs = []
    epochs = min(ir.training.epochs, 1)  # only 1 epoch for verify

    for epoch in range(epochs):
        print(f"▶️ Starting epoch {epoch+1} with {len(loader)} batches")
        for i, batch in enumerate(loader):
            if len(batch) == 3:  # huggingface style
                input_ids, attn, y = batch
                input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn, labels=y)
                logits = outputs.logits
                loss = loss_fn(logits, y)
            else:  # dummy style
                X, y = batch
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i >= 10:
                print("⚠️ Early stopping after 10 batches (verify mode).")
                break

        acc = accuracy(logits, y)
        logs.append({"epoch": epoch+1, "loss": float(loss.item()), "acc": acc})
        print(f"✅ Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    if not logs:
        logs.append({"epoch": 1, "loss": 0.0, "acc": 0.0})

    # ✅ Build results dict
    results = {"logs": logs, "pass": {}}
    for metric, target in getattr(ir.training, "target_metrics", {}).items():
        if metric == "accuracy":
            final = logs[-1]["acc"]
            within_tol = abs(final - target) <= getattr(ir.training, "tolerance", 0.05)
            results["pass"][metric] = within_tol

    # ✅ Save report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Verification report saved at {report_path}")
