import json
from core.ir.schema import IR

def test_bert_ir_loads():
    with open("examples/bert_sst2.json", "r") as f:
        data = json.load(f)
    ir = IR(**data)
    assert ir.model.family == "Transformer"
    assert ir.dataset.name == "SST-2"

def test_resnet_ir_loads():
    with open("examples/resnet_cifar10.json", "r") as f:
        data = json.load(f)
    ir = IR(**data)
    assert ir.model.variant == "ResNet-18"
    assert ir.training.loss == "CrossEntropyLoss"
