import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

def load_dataset_from_ir(ir, subset_fraction=0.1):
    try:
        # --- Extract tokenizer + max_len safely ---
        if isinstance(ir.preprocessing, dict):
            tok_name = ir.preprocessing.get("tokenizer", "bert-base-uncased")
            max_len = ir.preprocessing.get("max_len", 128)
        else:
            tok_name = getattr(ir.preprocessing, "tokenizer", "bert-base-uncased")
            max_len = getattr(ir.preprocessing, "max_len", 128)

        tokenizer = AutoTokenizer.from_pretrained(tok_name)

        # --- Parse dataset source ---
        if isinstance(ir.dataset.source, str) and ir.dataset.source.startswith("hf://"):
            name = ir.dataset.source.replace("hf://", "")
            parts = name.split("/")
            if len(parts) == 2:
                ds = load_dataset(parts[0], parts[1])
            else:
                ds = load_dataset(name)

            # --- Only use a subset fraction ---
            train_split = ds["train"]
            subset_size = int(len(train_split) * subset_fraction)
            train_split = train_split.select(range(subset_size))

            # --- Feature keys ---
            text_a = ir.dataset.features.get("text_a")
            text_b = ir.dataset.features.get("text_b")
            label_key = ir.dataset.features.get("label")

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
            return get_dummy_loader(ir.training.batch_size, num_classes=getattr(ir.model, "num_labels", 2))

    except Exception as e:
        print(f"⚠️ Dataset load failed, fallback to dummy. Error: {e}")
        return get_dummy_loader(ir.training.batch_size, num_classes=getattr(ir.model, "num_labels", 2))

def get_dummy_loader(batch_size=32, num=100, input_dim=16, num_classes=2, with_attention=False, is_cnn=False):
    """
    Generate dummy dataset for verification.
    - with_attention=True → (input_ids, attn, labels)
    - is_cnn=True → (image_tensor, labels) with shape [B, C, H, W]
    - else → (X, labels)
    """
    if with_attention:
        input_ids = torch.randint(0, 1000, (num, input_dim), dtype=torch.long)
        attn = torch.ones(num, input_dim, dtype=torch.long)
        y = torch.randint(0, num_classes, (num,), dtype=torch.long)
        dataset = TensorDataset(input_ids, attn, y)

    elif is_cnn:
        # Make fake images: [batch, channels=3, height=224, width=224]
        X = torch.randn(num, 3, 224, 224)
        y = torch.randint(0, num_classes, (num,))
        dataset = TensorDataset(X, y)

    else:
        X = torch.randn(num, input_dim)
        y = torch.randint(0, num_classes, (num,))
        dataset = TensorDataset(X, y)

    return DataLoader(dataset, batch_size=batch_size)
