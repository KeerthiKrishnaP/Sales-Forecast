import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.enums import MeanType


class Validators():
    def __init__(self, y_true: np.ndarray, y_pred:np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
    #MAE
    def compute_mae(self)->float:

        return round(mean_absolute_error(self.y_true,self.y_pred), 4)
    #MSE
    def compute_mse(self)->float:

        return round(mean_squared_error(self.y_true, self.y_pred), 4)
    #RMSE
    def compute_rmse(self)->float:

        return round(np.sqrt(self.compute_mse()), 4)
    #R2
    def compute_r2_score(self)->float:

        return round(r2_score(self.y_true,self.y_pred), 4)


def get_average(data_frame: pd.DataFrame, meantype: MeanType) -> pd.DataFrame:
    data_frame['day_id'] = pd.to_datetime(data_frame['day_id'])
    data_frame.set_index('day_id', inplace=True)
    match meantype:
        case MeanType.YEAR:
            mean = data_frame.resample("Y").mean()
        case MeanType.QUATER:
            mean = data_frame.resample("3ME").mean()
        case MeanType.MONTH:
            mean = data_frame.resample("ME").mean()
        case MeanType.WEEK:
            mean = data_frame.resample("W").mean()
    
    return mean

def remove_mean_from_data(data_frame: pd.DataFrame, meantype: MeanType) -> pd.DataFrame:
    weekly_signal = get_average(data_frame = data_frame, meantype = MeanType.WEEK)

    return weekly_signal

def get_weekly_mean(data_frame: pd.DataFrame)->pd.DataFrame:
    weekly_mean = data_frame.resample("W").mean()
    weekly_mean = weekly_mean.set_index('day_id', inplace=True)

    return weekly_mean

def denormalize_min_max(normalized_data: np.ndarray | list[float], min_value: float, max_value: float) -> np.ndarray:
        
    return (normalized_data * (max_value - min_value)) + min_value
