from setuptools import find_packages, setup

setup(
    name='heart_cleveland_app',
    packages=find_packages(),
    version='0.1.0',
    description='Homework_01 ML in production',
    author='minakovaa',
    entry_points={
        "console_scripts": [
            "ml_project_train = heart_cleveland_app.train_pipeline:train_pipeline_command",
            "ml_project_predict = heart_cleveland_app.predict_pipeline:predict_pipeline_command",
        ]
    },
    license='',
)
