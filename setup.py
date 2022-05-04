from setuptools import find_packages, setup

setup(
    name='ml_project',
    packages=find_packages(),
    version='0.1.0',
    description='Homework_01 ML in production',
    author='minakovaa',
    entry_points={
        "console_scripts": [
            "ml_project_train = ml_project.train_pipeline:train_pipeline_command",
            "ml_project_predict = ml_project.predict_pipeline:predict_pipeline_command",
        ]
    },
    license='',
)
