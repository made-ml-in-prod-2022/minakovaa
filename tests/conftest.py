import os
from typing import List

import pytest


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "data_for_tests.csv")


@pytest.fixture()
def test_size() -> float:
    return 0.4


@pytest.fixture()
def random_state() -> int:
    return 123


@pytest.fixture()
def target_col() -> str:
    return "condition"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]
