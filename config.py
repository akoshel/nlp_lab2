from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class NetParams:
    ENC_EMB_DIM: int
    DEC_EMB_DIM: int
    HID_DIM: int
    N_LAYERS: int
    ENC_DROPOUT: float
    DEC_DROPOUT: float


@dataclass
class SplitRatio:
    train_size: float
    valid_size: float
    test_size: float

@dataclass
class LRSchedulerParams:
    mode: str
    factor: float
    patience: int
    threshold: float
    threshold_mode: str
    cooldown: int
    min_lr: float
    eps: float

@dataclass
class ModelParams:
    net_params: NetParams
    split_ration: SplitRatio
    dataset_path: str
    BATCH_SIZE: int
    CLIP: int
    N_EPOCHS: int
    model_out_name: str
    lr: float
    lr_scheduler: LRSchedulerParams



ModelParamsSchema = class_schema(ModelParams)


def read_training_pipeline_params(path: str) -> ModelParams:
    with open(path, "r") as input_stream:
        schema = ModelParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
