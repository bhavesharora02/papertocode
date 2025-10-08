import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer


# ---------------------------------------------
# Safe Print (for Windows)
# ---------------------------------------------
def safe_print(msg: str):
    """Print safely without Unicode errors."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="ignore").decode())


# ---------------------------------------------
# Main Loader
# ---------------------------------------------
def load_dataset_from_ir(ir, subset_fraction=0.1):
    """
    Load dataset from IR with CNN/NLP auto-detection.
    Falls back safely to dummy loader.
    """
    try:
        # --- Tokenizer info (for NLP) ---
        if isinstance(ir.preprocessing, dict):
            tok_name = ir.preprocessing.get("tokenizer", "bert-base-uncased")
            max_len = ir.preprocessing.get("max_len", 128)
        else:
            tok_name = getattr(ir.preprocessing, "tokenizer", "bert-base-uncased")
            max_len = getattr(ir.preprocessing, "max_len", 128)

        # --- Detect if model is CNN ---
        model_name = getattr(ir.model, "family", "").lower()
        is_cnn = any(k in model_name for k in ["cnn", "resnet", "efficientnet", "vgg", "conv", "inception"])
        num_classes = getattr(ir.model, "num_labels", 2)

        # --- Parse dataset source ---
        dataset_src = getattr(ir.dataset, "source", None)
        if isinstance(dataset_src, str) and dataset_src.startswith("hf://"):
            name = dataset_src.replace("hf://", "").strip()
            if not name:
                safe_print("[WARNING] Empty dataset source detected, using 'cifar10' fallback.")
                name = "cifar10"

            try:
                parts = name.split("/")
                if len(parts) == 2:
                    ds = load_dataset(parts[0], parts[1])
                else:
                    ds = load_dataset(name)
            except Exception as e:
                safe_print(f"[WARNING] Dataset load failed ({name}), using fallback. Error: {e}")
                ds = load_dataset("cifar10")

            # --- Limit dataset for fast verify ---
            train_split = ds["train"]
            subset_size = max(1, int(len(train_split) * subset_fraction))
            train_split = train_split.select(range(subset_size))

            # --- Tokenize if text-based dataset ---
            text_a = ir.dataset.features.get("text_a")
            text_b = ir.dataset.features.get("text_b")
            label_key = ir.dataset.features.get("label", "label")

            if text_a:
                tokenizer = AutoTokenizer.from_pretrained(tok_name)

                def encode(batch):
                    if text_b:
                        return tokenizer(
                            batch[text_a],
                            batch[text_b],
                            padding="max_length",
                            truncation=True,
                            max_length=max_len
                        )
                    else:
                        return tokenizer(
                            batch[text_a],
                            padding="max_length",
                            truncation=True,
                            max_length=max_len
                        )

                enc = train_split.map(encode, batched=True)
                X = torch.tensor(enc["input_ids"], dtype=torch.long)
                attn = torch.tensor(enc["attention_mask"], dtype=torch.long)
                y = torch.tensor(enc[label_key], dtype=torch.long)
                dataset = TensorDataset(X, attn, y)
                return DataLoader(dataset, batch_size=ir.training.batch_size)

            else:
                # CNN fallback for image-like datasets
                safe_print("[INFO] No text feature found — using CNN dummy data.")
                return get_dummy_loader(ir.training.batch_size, num_classes=num_classes, is_cnn=True)

        else:
            safe_print("[INFO] Non-HF dataset source detected — using dummy fallback.")
            return get_dummy_loader(ir.training.batch_size, num_classes=num_classes, is_cnn=is_cnn)

    except Exception as e:
        safe_print(f"[WARNING] Dataset load failed, fallback to dummy. Error: {e}")
        return get_dummy_loader(ir.training.batch_size, num_classes=getattr(ir.model, "num_labels", 2), is_cnn=True)


# ---------------------------------------------
# Dummy Loader (CNN + NLP + Tabular)
# ---------------------------------------------
def get_dummy_loader(batch_size=32, num=100, input_dim=16, num_classes=2, with_attention=False, is_cnn=False):
    """
    Generate dummy dataset for verification.
    Automatically supports CNNs, Transformers, and simple tabular models.
    """
    if with_attention:
        input_ids = torch.randint(0, 1000, (num, input_dim), dtype=torch.long)
        attn = torch.ones(num, input_dim, dtype=torch.long)
        y = torch.randint(0, num_classes, (num,), dtype=torch.long)
        dataset = TensorDataset(input_ids, attn, y)

    elif is_cnn:
        # Fake images: [batch, channels=3, height=224, width=224]
        X = torch.randn(num, 3, 224, 224)
        y = torch.randint(0, num_classes, (num,))
        dataset = TensorDataset(X, y)

    else:
        X = torch.randn(num, input_dim)
        y = torch.randint(0, num_classes, (num,))
        dataset = TensorDataset(X, y)

    return DataLoader(dataset, batch_size=batch_size)
