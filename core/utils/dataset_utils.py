# core/utils/dataset_utils.py
from datasets import load_dataset

# Common gated datasets → fallback to public datasets
PUBLIC_FALLBACKS = {
    "imagenet-1k": "cifar10",
    "imagenet": "cifar10",
    "imagenet-21k": "cifar100",
    "lsun": "mnist",
    "dummy": "cifar10"
}

def safe_load_dataset(name: str, split="train[:1]", default="cifar10"):
    """
    Try loading dataset safely. If gated or missing, use fallback mapping or default.
    Uses only ASCII logs to avoid UnicodeEncodeError on Windows.
    """
    try:
        if name in PUBLIC_FALLBACKS:
            try:
                return load_dataset(name, split=split)
            except Exception:
                fb = PUBLIC_FALLBACKS[name]
                print(f"Warning: Dataset {name} gated -> falling back to {fb}")
                return load_dataset(fb, split=split)
        else:
            return load_dataset(name, split=split)
    except Exception:
        print(f"Warning: All dataset loads failed for {name}, using default {default}")
        return load_dataset(default, split=split)
