import os
import logging
import json
import sys
sys.path.append(".")

import click
import mlflow

from ml_project.data.make_dataset import (
    download_from_gdrive,
    unzip_downloaded_data,
    read_data,
    split_train_test_data,
)
from ml_project.enities.train_pipeline_params import read_training_pipeline_params
from ml_project.features.build_features import (
    make_features,
    build_and_fit_transformer,
    extract_target,
    drop_feature,
)
from ml_project.models.train_model import train_model
from ml_project.models.predict_model import (
    predict_model,
    count_metrics,
    create_inference_pipeline,
    save_model,
    load_model,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)

    if training_pipeline_params.train_params.is_use_ml_flow:
        logger.info("Run with mlflow")
        with mlflow.start_run() as run:
            path_to_model, metrics_test = run_train_pipeline(training_pipeline_params)

            # Log parameters and metrics using the MLflow APIs
            for name, param in training_pipeline_params.train_params.model_params.items():
                mlflow.log_param(name, param)

            mlflow.log_metrics(metrics_test)

            model_loaded = load_model(path_to_model)
            # Log the sklearn model and register as version 1
            mlflow.sklearn.log_model(
                sk_model=model_loaded,
                artifact_path=path_to_model,
                registered_model_name=training_pipeline_params.train_params.model_type
            )
    else:
        return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    downloading_params = training_pipeline_params.downloading_params
    if downloading_params:
        os.makedirs(downloading_params.output_folder, exist_ok=True)

        path_to_zip_file = os.path.join(downloading_params.output_folder, downloading_params.zip_name)
        download_from_gdrive(
            downloading_params.gdrive_id,
            path_to_zip_file,
        )
        if os.path.exists(path_to_zip_file):
            unzip_downloaded_data(path_to_zip_file, downloading_params.output_folder)
            logger.info(f"Unzip {downloading_params.zip_name}")

    if not os.path.exists(training_pipeline_params.input_data_path):
        logger.error(f"File '{training_pipeline_params.input_data_path}' not exist")
        return None

    logger.info("Start read and split data")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"Data read. Data shape is {data.shape}")

    train_df, test_df = split_train_test_data(data, training_pipeline_params.splitting_params)
    logger.info(f"Train data shape is {train_df.shape}. Test data shape is {test_df.shape}")

    target_column = training_pipeline_params.feature_params.target_col
    train_target = extract_target(train_df, target_column)
    train_df = drop_feature(train_df, target_column)

    transformer = build_and_fit_transformer(training_pipeline_params.feature_params, train_df, train_target)
    train_df = make_features(transformer, train_df)
    logger.info("Dataset prepeared for train")

    train_params = training_pipeline_params.train_params
    logger.info(f"Start train pipeline with params {train_params}")
    model = train_model(train_df, train_target, train_params)
    predict_train = predict_model(model, train_df)
    metrics_train = count_metrics(predict_train, train_target)
    logger.info(f"Accuracy train {metrics_train['accuracy']}")

    #  Test section
    test_target = extract_target(test_df, target_column)
    test_df = drop_feature(test_df, target_column)

    inference_pipeline = create_inference_pipeline(model, transformer)
    predict_target = predict_model(inference_pipeline, test_df)

    metrics_test = count_metrics(predict_target, test_target)
    logger.info(f"Accuracy test {metrics_test['accuracy']}")

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics_test, metric_file)
    logger.info(f"Test metrics is {metrics_test}")

    path_to_model = save_model(
        inference_pipeline, training_pipeline_params.output_model_path
    )

    return path_to_model, metrics_test


@click.command(name="train_pipeline")
@click.argument("config_path", )
def train_pipeline_command(config_path: str):
    """ Run train pipeline with config from CONFIG_PATH file"""
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
