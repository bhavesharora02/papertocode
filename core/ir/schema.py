from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any


class DatasetCfg(BaseModel):
    name: str
    source: str
    subset_fraction: float = 0.1
    splits: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
    features: Dict[str, str]

    @field_validator("splits")
    @classmethod
    def check_splits_sum(cls, v):
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError("Dataset splits must sum to 1.0")
        return v


class Layer(BaseModel):
    type: str
    params: Dict[str, Any] = {}


class ModelCfg(BaseModel):
    family: str
    variant: Optional[str]
    framework: str = "torch"
    layers: List[Layer]
    init: Dict[str, Any] = {}


class OptimCfg(BaseModel):
    name: str
    lr: float
    weight_decay: float = 0.0


class SchedCfg(BaseModel):
    name: Optional[str]
    kwargs: Dict[str, Any] = {}


class TrainCfg(BaseModel):
    loss: str
    optimizer: OptimCfg
    scheduler: Optional[SchedCfg]
    batch_size: int
    epochs: int
    metrics: List[str]
    target_metrics: Dict[str, float] = {}
    tolerance: float = 0.02


class PaperMeta(BaseModel):
    title: str
    arxiv_id: Optional[str]
    tasks: List[str]
    domain: str


class MappingCfg(BaseModel):
    nn_modules: Dict[str, str] = {}


class IR(BaseModel):
    paper: PaperMeta
    dataset: DatasetCfg
    model: ModelCfg
    training: TrainCfg
    preprocessing: Dict[str, Any] = {}
    mapping: MappingCfg = MappingCfg()
