
from pydantic import BaseModel


class TrainingData(BaseModel):
    x_train: list[float]
    y_train: list[float]
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class TestingData(BaseModel):
    x_test: list[float]
    y_test: list[float]
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class ModelData(BaseModel):
    train_data: TrainingData
    test_data: TestingData
