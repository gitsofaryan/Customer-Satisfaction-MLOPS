from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    modelname: str="LinearRegression"


