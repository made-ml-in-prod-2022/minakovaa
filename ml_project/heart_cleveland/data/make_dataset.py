# -*- coding: utf-8 -*-
import logging
import zipfile
from requests.exceptions import ConnectionError

import gdown
import pandas as pd
from sklearn.model_selection import train_test_split


from heart_cleveland.enities.splitting_params import SplittingParams

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def download_from_gdrive(gdrive_id, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    try:
        gdown.download(id=gdrive_id, output=output_filepath, quiet=False)
    except ConnectionError:
        logger.warning('ConnectionError. Dataset not downloaded.')
        return

    logger.info('Download dataset from gdrive')


def unzip_downloaded_data(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        logger.info('Unzip downloaded archive')


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, index_col=False)
    return data


def split_train_test_data(data: pd.DataFrame, split_params: SplittingParams):
    train_data, test_data = train_test_split(
        data,
        test_size=split_params.test_size,
        random_state=split_params.random_state
    )
    return train_data, test_data
