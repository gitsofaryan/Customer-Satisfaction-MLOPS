import logging
import mlflow
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from zenml.steps import BaseStep
from zenml.client import Client


experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker)
def train_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        config: ModelNameConfig,
) -> RegressorMixin:

    try:
        model = None
        if config.modelname == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(
                X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model not found")
    except Exception as e:
        logging.error(e)
        raise e
