import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .train_model import SklearnClfModel


def create_inference_pipeline(model: SklearnClfModel, transformer: ColumnTransformer) -> Pipeline:
    return Pipeline(steps=[("transformer_features", transformer), ("model", model)])


def count_metrics(predict_target: pd.Series, true_target: pd.Series):
    precision = precision_score(true_target, predict_target)
    recall = recall_score(true_target, predict_target)
    accuracy = accuracy_score(true_target, predict_target)
    f1 = f1_score(true_target, predict_target)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def predict_model(model_clf: Pipeline, features: pd.DataFrame):
    y_pred = model_clf.predict(features)
    return y_pred


def save_prediction(preds: pd.Series, out_path: str):
    with open(out_path, "w") as f:
        preds_str = ",\n".join(preds.astype(str))
        f.write(preds_str)


def save_model(model: Pipeline, output_path: str) -> str:
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    return output_path


def load_model(model_path) -> Pipeline:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
