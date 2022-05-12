import os
from typing import List
import sys
sys.path.append(".")

from heart_cleveland.train_pipeline import run_train_pipeline
from heart_cleveland.enities.train_pipeline_params import TrainingPipelineParams
from heart_cleveland.enities.feature_params import FeatureParams
from heart_cleveland.enities.splitting_params import SplittingParams
from heart_cleveland.enities.training_params import TrainingParams


def test_run_train_pipeline(
        tmp_path,
        syntetic_dataset_path: str, test_size: float, random_state: int,
        target_col: str,
        categorical_features: List[str],
        numerical_features: List[str],
):
    feature_params = FeatureParams(categorical_features, numerical_features, target_col)
    output_model_path = tmp_path.joinpath("model.pkl")
    output_metric_path = tmp_path.joinpath("metric.txt")
    splitting_params = SplittingParams(test_size, random_state)
    train_params = TrainingParams(model_type="KNeighborsClassifier", model_params={"n_neighbors": 11},)
    training_pipeline_params = TrainingPipelineParams(
        syntetic_dataset_path,
        feature_params,
        output_model_path,
        output_metric_path,
        splitting_params,
        train_params,
        downloading_params=None,
    )
    path_to_model, metrics_test = run_train_pipeline(training_pipeline_params)

    assert path_to_model == output_model_path, "Saved path and expected path not equal"
    assert os.path.exists(path_to_model), "Model not saved"
    assert metrics_test is not None
