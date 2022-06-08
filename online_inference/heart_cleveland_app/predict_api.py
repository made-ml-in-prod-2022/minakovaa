import os
import logging
import pickle
from typing import List, Union, Optional

import gdown
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()


class HeartClevelandModel(BaseModel):
    data: List[conlist(Union[float, int], min_items=13, max_items=13)]
    features: List[str]


class HeartClevelandResponse(BaseModel):
    condition: int


model: Optional[Pipeline] = None


def make_predict(model_clf, data: List, features: List[str]) -> List[HeartClevelandResponse]:
    data = pd.DataFrame(data, columns=features)

    predictions = model_clf.predict(data)
    return [HeartClevelandResponse(condition=prediction) for prediction in predictions]


@app.get("/")
async def root():
    return {"message": "This is heart cleveland prediction model api"}


@app.get("/health")
async def health():
    if model is not None:
        return JSONResponse(content="model ready", status_code=200)
    else:
        return JSONResponse(content="model not ready", status_code=500)


def download_from_gdrive(gdrive_id, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    try:
        gdown.download(id=gdrive_id, output=output_filepath, quiet=False)
        logger.info("Model downloaded from gdrive")
    except ConnectionError:
        print('ConnectionError. Dataset not downloaded.')
        logger.error('ConnectionError. Dataset not downloaded.')
        return


def load_model(model_path) -> Pipeline:
    with open(model_path, "rb") as f:
        model_loaded = pickle.load(f)
    return model_loaded


@app.on_event("startup")
def load_model_startup(model_path_arg=None, gdrive_id_arg=None):
    global model
    if model_path_arg is None:
        model_path = os.getenv("PATH_TO_MODEL")
    else:
        model_path = model_path_arg

    if model_path is None:
        err = f"PATH_TO_MODEL is None"
        logger.error(err)
        raise RuntimeError(err)

    if not os.path.exists(model_path):
        if gdrive_id_arg is None:
            gdrive_id = os.getenv("GDRIVE_ID")
        else:
            gdrive_id = gdrive_id_arg

        if gdrive_id is None:
            err = f"GDRIVE_ID is None"
            logger.error(err)
            raise RuntimeError(err)

        download_from_gdrive(gdrive_id, model_path)

    model = load_model(model_path)


@app.get("/predict/", response_model=List[HeartClevelandResponse])
def predict(request: HeartClevelandModel):
    global model
    return make_predict(model, request.data, request.features)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
