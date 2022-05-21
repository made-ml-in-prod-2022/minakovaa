import os
import pickle
import requests

import gdown
import zipfile
import click
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(model_path) -> Pipeline:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, index_col=False)
    return data


def predict_model(model_clf: Pipeline, features: pd.DataFrame):
    y_pred = model_clf.predict(features)
    return y_pred


def save_prediction(preds: pd.Series, out_path: str):
    with open(out_path, "w") as f:
        preds_str = ",\n".join(preds.astype(str))
        f.write(preds_str)


def download_from_gdrive(gdrive_id, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    try:
        gdown.download(id=gdrive_id, output=output_filepath, quiet=False)
    except ConnectionError:
        print('ConnectionError. Dataset not downloaded.')
        return

    # print('Download dataset from gdrive')


def unzip_downloaded_data(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        # logger.info('Unzip downloaded archive')


def load_data(gdrive_id: str, output_folder: str, file_name: str):

    os.makedirs(output_folder, exist_ok=True)

    path_to_file = os.path.join(output_folder, file_name)
    download_from_gdrive(gdrive_id, path_to_file)


def batch_predict(
    path_to_data: str = "data.csv",
    app_address: str = "http://127.0.0.1:8000",
    output: str = "predicts.csv",
):
    data = pd.read_csv(path_to_data)
    request_features = list(data.columns)
    predicted_tagret = []

    request_data = data.values.tolist()
    response = requests.get(
        f"{app_address}/predict/",
        json={"data": request_data, "features": request_features},
    )

    if 200 == response.status_code:
        predicted_tagret = [pred["condition"] for pred in response.json()]

        save_prediction(pd.Series(predicted_tagret), output)

    # load_data(model_gdrive_id, "./", local_model_path)
    # features = read_data(local_data_path)
    # model = load_model(local_model_path)
    # predicted_tagret = predict_model(model, features)
    # save_prediction(predicted_tagret, local_output)


@click.command(name="batch_predict")
@click.argument("PATH_TO_DATA", default=os.getenv("PATH_TO_DATA"))
# @click.argument("PATH_TO_MODEL", default=os.getenv("PATH_TO_MODEL"))
@click.argument("APP_ADDRESS", default=os.getenv("APP_ADDRESS"))
@click.argument("OUTPUT", default=os.getenv("OUTPUT"))
# @click.argument("GDRIVE_ID", default="1Sgpzrv2KU01vnynifIDVCi-Ca0wddhYZ")
def batch_predict_command(
        path_to_data: str,
        # path_to_model: str,
        app_address: str,
        output: str,
        # gdrive_id: str,
):
    batch_predict(path_to_data, app_address, output)


if __name__ == "__main__":
    batch_predict_command()
