import os
import requests

import click
import pandas as pd


def save_prediction(preds: pd.Series, out_path: str):
    with open(out_path, "w") as f:
        preds_str = ",\n".join(preds.astype(str))
        f.write(preds_str)


def batch_predict(
    path_to_data: str = "data.csv",
    app_address: str = "http://127.0.0.1:8000",
    output: str = "predicts.csv",
):
    data = pd.read_csv(path_to_data)
    request_features = list(data.columns)

    request_data = data.values.tolist()
    response = requests.get(
        f"{app_address}/predict/",
        json={"data": request_data, "features": request_features},
    )

    if 200 == response.status_code:
        predicted_tagret = [pred["condition"] for pred in response.json()]

        save_prediction(pd.Series(predicted_tagret), output)


@click.command(name="batch_predict")
@click.argument("PATH_TO_DATA", default=os.getenv("PATH_TO_DATA"))
@click.argument("APP_ADDRESS", default=os.getenv("APP_ADDRESS"))
@click.argument("OUTPUT", default=os.getenv("OUTPUT"))
def batch_predict_command(
        path_to_data: str,
        app_address: str,
        output: str,
):
    batch_predict(path_to_data, app_address, output)


if __name__ == "__main__":
    batch_predict_command()
