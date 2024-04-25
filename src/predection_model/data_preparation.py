from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.helpers.computations import denormalize_min_max, get_average
from src.helpers.data import get_data_for_department_business
from src.models.data_models import ModelData, TestingData, TrainingData
from src.models.enums import MeanType


def get_training_vectors(
    data_frame: pd.DataFrame,
    split_ratio: float = 0.8,
    department: int = 117,
    business: int = 53,
    meantype: MeanType = MeanType.WEEK,
    normalize_data: bool = True,
) -> ModelData:

    filtered_data = get_data_for_department_business(
        data_frame=data_frame, department=department, business=business
    )
    weekly_data = get_average(data_frame=filtered_data, meantype=meantype)
    weekly_data.reset_index(inplace=True)
    # Normalize the x and y data
    x_data = (weekly_data["day_id"].dt.isocalendar().week / 53).to_list()
    y_data = weekly_data["turnover"].to_list()
    y_min = min(y_data)
    y_max = max(y_data)
    x_min = min(x_data)
    x_max = min(x_data)
    if normalize_data:
        y_data = [(x - y_min) / (y_max - y_min) for x in y_data]
    test_split = int(split_ratio * len(x_data))

    return ModelData(
        train_data=TrainingData(
            x_train=x_data[:test_split],
            y_train=y_data[:test_split],
            x_max=x_max,
            x_min=x_min,
            y_min=y_min,
            y_max=y_max,
        ),
        test_data=TestingData(
            x_test=x_data[test_split:],
            y_test=y_data[test_split:],
            x_max=x_max,
            x_min=x_min,
            y_min=y_min,
            y_max=y_max,
        ),
    )


def forecast(
    forcat_time_inweeks: int,
    data_frame: pd.DataFrame,
    model: nn.Module,
    department: int,
    business: int,
    meantype: MeanType,
    model_data: ModelData,
):
    filtered_data = get_data_for_department_business(
        data_frame=data_frame, department=department, business=business
    )
    weekly_data = get_average(data_frame=filtered_data, meantype=meantype)
    weekly_data.reset_index(inplace=True)
    weekly_data["day_id"] = pd.to_datetime(weekly_data["day_id"])
    last_date = weekly_data["day_id"].max()
    new_dates = [last_date + timedelta(weeks=i) for i in range(1, forcat_time_inweeks)]
    x_data = torch.Tensor(
        np.asarray(
            (
                (pd.DataFrame({"day_id": new_dates})["day_id"]).dt.isocalendar().week
                / 53
            ).to_list()
        ).reshape(-1, 1)
    )
    y_data = model(x_data)
    y_data = y_data.detach().numpy()
    forcasted_data = denormalize_min_max(
        normalized_data=np.asarray(y_data.ravel()),
        max_value=model_data.test_data.y_max,
        min_value=model_data.test_data.y_min,
    )
    return pd.DataFrame({"day_id": new_dates, "turnover": forcasted_data})
