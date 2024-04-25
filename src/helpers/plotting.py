import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.helpers.computations import denormalize_min_max
from src.models.data_models import ModelData


def plot_predection_vs_test_data(
    model_data: ModelData, predictions: np.ndarray, denormalize: bool = False
) -> Figure:
    x_test = model_data.test_data.x_test
    if denormalize:
        y_test = denormalize_min_max(
            normalized_data=np.asarray(model_data.test_data.y_test),
            max_value=model_data.test_data.y_max,
            min_value=model_data.test_data.y_min,
        )
        predictions = denormalize_min_max(
            normalized_data=np.asarray(predictions),
            max_value=model_data.test_data.y_max,
            min_value=model_data.test_data.y_min,
        )
    else:
        y_test = model_data.test_data.y_test
    figure, ax = plt.subplots()
    plt.scatter(x_test, y_test, color="blue", alpha=0.5, label="test data")
    plt.scatter(x_test, predictions, color="red", label="predictions")
    plt.legend()
    return figure


def plot_turnover(data: pd.DataFrame) -> Figure:
    figure, ax = plt.subplots()
    plt.plot(data["day_id"], data["turnover"], color="red")
    plt.bar(data["day_id"], data["turnover"], color="skyblue")
    # plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.title("Turnover Over Time")
    plt.xticks(rotation=90)

    return figure


def plot_training_data(x_train, y_train) -> Figure:
    figure, ax = plt.subplots()
    plt.scatter(x_train, y_train, color="blue", alpha=0.5, label="train data")
    plt.legend()

    return figure
