from typing import List
import sys
sys.path.append(".")

from sklearn.compose import ColumnTransformer
from numpy import allclose

from heart_cleveland.features.build_features import (
    make_features,
    extract_target,
    build_and_fit_transformer,
    drop_feature,
)

from heart_cleveland.data.make_dataset import read_data, split_train_test_data
from heart_cleveland.enities.splitting_params import SplittingParams
from heart_cleveland.enities.feature_params import FeatureParams


def test_transform_features_pipeline(
        syntetic_dataset_path: str, test_size: float, random_state: int,
        target_col: str,
        categorical_features: List[str],
        numerical_features: List[str],
):
    df = read_data(syntetic_dataset_path)
    split_params = SplittingParams(test_size=test_size, random_state=random_state)
    df_train, df_test = split_train_test_data(df, split_params)

    train_target = extract_target(df_train, target_col)
    assert train_target.shape[0] > 0, "Not extract target from dataframe"

    df_train = drop_feature(df_train, target_col)
    assert target_col not in df_train.columns, "Target column not dropped"

    feature_params = FeatureParams(categorical_features, numerical_features, target_col)
    transformer = build_and_fit_transformer(feature_params, df_train, train_target)
    assert type(transformer) is ColumnTransformer, "transformer is not sklearn ColumnTransformer"

    trans_train_df = make_features(transformer, df_train)
    assert allclose(trans_train_df[:, :len(numerical_features)].mean(axis=0), 0, atol=1e-5), (
        "Numerical features should be Standartize")
    assert allclose(trans_train_df[:, :len(numerical_features)].std(axis=0), 1, atol=1e-5), (
        "Numerical features should be Standartize")
