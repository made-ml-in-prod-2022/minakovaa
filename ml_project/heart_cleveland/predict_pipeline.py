import logging
import sys
sys.path.append(".")

import click

from heart_cleveland.models.predict_model import predict_model, load_model, save_prediction
from heart_cleveland.data.make_dataset import read_data


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_pipeline(model_path: str, prediction_path: str, features_to_predict_path: str):
    features = read_data(features_to_predict_path)
    logger.info(f"Data read. Data shape is {features.shape}")

    model = load_model(model_path)
    logger.info(f"Model loaded {model['model']}")

    predicted_tagret = predict_model(model, features)
    logger.info("Model predict target")

    save_prediction(predicted_tagret, prediction_path)
    logger.info(f"Prediction saved to {prediction_path}")


@click.command(name="predict_pipeline")
@click.argument("model_path")
@click.argument("prediction_out_path")
@click.argument("features_to_predict_path")
def predict_pipeline_command(model_path: str, prediction_out_path: str, features_to_predict_path: str):
    """
    Run prediction for FEATURES_TO_PREDICT_PATH *.csv file
    with model loaded from MODEL_PATH *.pkl file.
    Write predictions to PREDICTION_OUT_PATH *.txt file.
    """
    predict_pipeline(model_path, prediction_out_path, features_to_predict_path)


if __name__ == "__main__":
    predict_pipeline_command()
