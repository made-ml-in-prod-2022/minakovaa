import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from heart_cleveland.enities.feature_params import FeatureParams


def build_and_fit_transformer(params: FeatureParams, features: pd.DataFrame, target: pd.Series) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("one_hot_enc", OneHotEncoder(handle_unknown="ignore"))]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, params.numerical_features),
            ("cat", categorical_transformer, params.categorical_features),
        ]
    )

    transformer.fit(features, target)

    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> np.array:
    return transformer.transform(df)


def extract_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    return df[target_column]


def drop_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    return df.drop(feature_name, axis=1)
