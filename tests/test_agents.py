import json
from core.ir.schema import IR
from core.ir.build_ir import build_ir

def test_a1_a2_merge_with_mocks(monkeypatch):
    # Mock A1 result (conceptual)
    a1 = {
        "family": "Transformer",
        "variant": "BERT",
        "layers": [
            {"type": "Embedding", "dim": 768},
            {"type": "TransformerEncoder", "num_layers": 12, "heads": 12, "hidden": 768},
            {"type": "ClassifierHead", "num_classes": 2}
        ],
        "training": {
            "loss": "CrossEntropyLoss",
            "optimizer": {"name": "AdamW", "lr": 2e-5, "weight_decay": 0.01},
            "scheduler": {"name": "linear", "kwargs": {"warmup_steps": 500}},
            "batch_size": 32,
            "epochs": 3,
            "metrics": ["accuracy"]
        },
        "preprocessing": {"tokenizer": "bert-base-uncased", "max_len": 128},
        "dataset_hints": {"name": "SST-2"}
    }

    # Mock A2 result (mapped)
    a2 = {
        "dataset": {
            "name": "SST-2",
            "source": "hf://glue/sst2",
            "subset_fraction": 0.1,
            "splits": {"train":0.8,"val":0.1,"test":0.1},
            "features": {"text":"sentence","label":"label"}
        },
        "init": {"pretrained": "bert-base-uncased"},
        "training": a1["training"],
        "preprocessing": a1["preprocessing"],
        "nn_modules": {
            "TransformerEncoder": "torch.nn.TransformerEncoder",
            "CrossEntropyLoss": "torch.nn.CrossEntropyLoss"
        }
    }

    meta = {"title":"BERT","arxiv_id":"1810.04805","tasks":["text_classification"],"domain":"NLP"}
    ir = build_ir(meta, a1, a2)
    assert isinstance(ir, IR)
    assert ir.model.family == "Transformer"
    assert ir.dataset.name == "SST-2"
    # roundtrip JSON
    dumped = ir.model_dump_json()
    assert "bert-base-uncased" in dumped
