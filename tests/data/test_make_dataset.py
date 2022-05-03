from typing import List
from numpy import isclose

from ml_project.data.make_dataset import read_data, split_train_test_data
from ml_project.enities.splitting_params import SplittingParams


def test_load_dataset(
        dataset_path: str,
        target_col: str,
        numerical_features: List[str],
        categorical_features: List[str],
):
    data = read_data(dataset_path)

    assert data.shape[0] > 0, "No data in rows in .csv"
    assert len(data.columns) == len(numerical_features) + len(categorical_features) + 1, \
        "Number of features should be equal number numerical + number categorial + target"
    assert target_col in data.columns, "Target is absent in dataframe"


def test_split_train_test_data(dataset_path: str, test_size: float, random_state: int):
    df = read_data(dataset_path)
    split_params = SplittingParams(test_size=test_size, random_state=random_state)
    df_train, df_test = split_train_test_data(df, split_params)

    assert isclose(df_test.shape[0] / df.shape[0], test_size, atol=0.01), "Not valid data splitting"
