import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataStrategy, DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data=df, strategy=process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(data=processed_data, strategy=divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning and division completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
