import os
from typing import List
import sys
sys.path.append(".")

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from ml_project.features.build_features import (
    make_features,
    extract_target,
    build_and_fit_transformer,
    drop_feature,
)
from ml_project.data.make_dataset import read_data, split_train_test_data
from ml_project.enities.splitting_params import SplittingParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.models.train_model import train_model
from ml_project.enities.training_params import TrainingParams
from ml_project.models.predict_model import (
    save_model,
    load_model,
    create_inference_pipeline,
    predict_model,
    count_metrics,
)


def test_save_load_model(tmpdir):
    expected_path = tmpdir.join("test_model.pkl")

    n_neighbors = 5
    model_to_save = KNeighborsClassifier(n_neighbors=n_neighbors)
    real_path = save_model(model_to_save, expected_path)
    assert real_path == expected_path, "Real and expected paths not equal"
    assert os.path.exists(expected_path), "Path not exists"

    model_loaded = load_model(real_path)
    assert isinstance(model_loaded, KNeighborsClassifier), f"Model should be {type(model_to_save)}"


def test_predict_after_training_model(
        syntetic_dataset_path: str, test_size: float, random_state: int,
        target_col: str,
        categorical_features: List[str],
        numerical_features: List[str],
):
    df = read_data(syntetic_dataset_path)
    split_params = SplittingParams(test_size=test_size, random_state=random_state)
    df_train, df_test = split_train_test_data(df, split_params)

    train_target = extract_target(df_train, target_col)
    df_train = drop_feature(df_train, target_col)

    feature_params = FeatureParams(categorical_features, numerical_features, target_col)
    transformer = build_and_fit_transformer(feature_params, df_train, train_target)

    df_train = make_features(transformer, df_train)
    train_params = TrainingParams(model_type="SVC", model_params={"C": 0.3},)

    model = train_model(df_train, train_target, train_params)
    assert isinstance(model, SVC)

    test_target = extract_target(df_test, target_col)
    df_test = drop_feature(df_test, target_col)

    inference_pipeline = create_inference_pipeline(model, transformer)
    predict_target = predict_model(inference_pipeline, df_test)

    metrics_test = count_metrics(predict_target, test_target)
    assert metrics_test is not None, "Can predict after training"
