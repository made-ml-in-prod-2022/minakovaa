ml_project
==============================

Homework_01 ML in production

For train classification model using dataset from https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

**Installation**: 
~~~
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

**Usage**:

**Train model**:
```shell script
python ml_project/train_pipeline.py CONFIG_PATH
```
for example:
```shell script
python ml_project/train_pipeline.py configs/train_config_kNN.yaml
```

**Predict target**:
```shell script
python ml_project/predict_pipeline.py [OPTIONS] MODEL_PATH PREDICTION_OUT_PATH FEATURES_TO_PREDICT_PATH
```
for example:
```shell script
python ml_project/predict_pipeline.py models/model.pkl outputs/preds.txt data/raw/test.csv
```

**Test**:
```shell script
pytest tests/
```


Dataset can load with link https://drive.google.com/file/d/1PiBD7lFKmGgX8fuaaeN-Q8_kXhaeEErO/view?usp=sharing



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_project         <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    |   ├── enities       <- Define dataclasses for parameters
    │   │   ├── download_params.py 
    │   │   ├── feature_params.py
    │   │   ├── splitting_params.py
    │   │   ├── train_pipeline_params.py
    │   │   └── training_params.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
