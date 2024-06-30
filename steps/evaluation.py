import logging
import mlflow
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2, RMSE, MAE
from typing import Tuple
from zenml.client import Client
from typing_extensions import Annotated


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker)
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
                   ) -> Tuple[Annotated[float, "MSE"],
                              Annotated[float, "R2"],
                              Annotated[float, "RMSE"],
                              Annotated[float, "MAE"]
                              ]:
    try:
        predictions = model.predict(X_test)

        mse = MSE().calculate_scores(y_test, predictions)
        mlflow.log_metric("mse", mse)

        r2 = R2().calculate_scores(y_test, predictions)
        mlflow.log_metric("r2", r2)

        rmse = RMSE().calculate_scores(y_test, predictions)
        mlflow.log_metric("rmse", rmse)

        mae = MAE().calculate_scores(y_test, predictions)
        mlflow.log_metric("mae", mae)

        return mse, r2, rmse, mae
    except Exception as e:
        logging.error(e)
        raise e
