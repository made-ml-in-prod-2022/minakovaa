from typing import Optional

from dataclasses import dataclass

from .download_params import DownloadParams
from .splitting_params import SplittingParams
from .training_params import TrainingParams
from .feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    feature_params: FeatureParams
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    downloading_params: Optional[DownloadParams] = None


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(config_path: str) -> TrainingPipelineParams:
    with open(config_path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
