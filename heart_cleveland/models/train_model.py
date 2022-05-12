from typing import Union

import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from ml_project.enities.training_params import TrainingParams

SklearnClfModel = Union[KNeighborsClassifier, SVC]


def train_model(features: pd.DataFrame, target: pd.Series, train_params: TrainingParams) -> SklearnClfModel:
    if train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(**train_params.model_params)
    elif train_params.model_type == "SVC":
        model = SVC(**train_params.model_params)
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model
