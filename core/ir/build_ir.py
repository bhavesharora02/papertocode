from .schema import IR, PaperMeta, DatasetCfg, ModelCfg, TrainCfg, MappingCfg, Layer

def build_ir(meta: dict, a1_json: dict, a2_json: dict) -> IR:
    """Merge interpreter (A1) + mapper (A2) outputs into IR"""
    model_layers = [
        Layer(type=layer.get("type"), params={k: v for k, v in layer.items() if k != "type"})
        for layer in a1_json.get("layers", [])
    ]

    return IR(
        paper=PaperMeta(**meta),
        dataset=DatasetCfg(**a2_json["dataset"]),
        model=ModelCfg(
            family=a1_json.get("family", "Transformer"),
            variant=a1_json.get("variant"),
            layers=model_layers,
            init=a2_json.get("init", {})
        ),
        training=TrainCfg(**a2_json["training"]),
        preprocessing=a2_json.get("preprocessing", {}),
        mapping=MappingCfg(nn_modules=a2_json.get("nn_modules", {}))
    )
