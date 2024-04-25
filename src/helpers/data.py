from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.figure import Figure

from src.helpers.computations import get_average
from src.models.enums import MeanType


def load_data_to_dataframe(file_path: Path) -> pd.DataFrame:

    return pq.read_table(file_path).to_pandas()


def get_data_for_department_business(
    data_frame: pd.DataFrame, department: int, business: int
) -> pd.DataFrame:
    filtered_data = data_frame[
        (data_frame["dpt_num_department"] == department)
        & (data_frame["but_num_business_unit"] == business)
    ]
    filtered_data = filtered_data.drop(
        columns=["dpt_num_department", "but_num_business_unit"]
    )

    return filtered_data


def data_by_department(data_frame: pd.DataFrame, department_number: int) -> Figure:
    sum_by_day_for_department = (
        data_frame[(data_frame["dpt_num_department"] == department_number)]
        .groupby("day_id")["turnover"]
        .sum()
        .reset_index()
    )
    print(sum_by_day_for_department.head())
    figure, ax = plt.subplots(figsize=(10, 6))
    plt.plot(
        sum_by_day_for_department["day_id"],
        sum_by_day_for_department["turnover"],
        color="red",
    )
    plt.bar(
        sum_by_day_for_department["day_id"],
        sum_by_day_for_department["turnover"],
        color="skyblue",
    )
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.title("Turnover Over for entire department")
    plt.xticks(rotation=90)

    return figure


def plot_means(
    data_frame: pd.DataFrame, department: int, business: int, meantype: MeanType
) -> Figure:
    filtered_data = data_frame[
        (data_frame["dpt_num_department"] == department)
        & (data_frame["but_num_business_unit"] == business)
    ]
    filtered_data = filtered_data.drop(
        columns=["dpt_num_department", "but_num_business_unit"]
    )
    average = get_average(data_frame=filtered_data, meantype=meantype)
    figure, ax = plt.subplots()
    plt.plot(average, marker="o")
    plt.title(f"{meantype.name} average")
    plt.xlabel(meantype.name)
    plt.ylabel(
        f"Average Value of turnover for department {department} and business {business}"
    )

    return figure
