import os
from typing import List

import pytest

from .generate_fake_datatet import create_fake_dataset


@pytest.fixture()
def n_rows_in_syntetic_data() -> int:
    return 150


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


@pytest.fixture()
def syntetic_dataset_path(tmpdir_factory, n_rows_in_syntetic_data):
    filename = str(tmpdir_factory.mktemp("data").join("syntetic_dataset.csv"))

    fake_dataset_df = create_fake_dataset(n_rows_in_syntetic_data)
    fake_dataset_df.to_csv(filename, index=None)

    return filename
