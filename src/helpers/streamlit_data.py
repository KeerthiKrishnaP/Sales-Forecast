import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.helpers.data import data_by_department, plot_means
from src.models.enums import MeanType


def streamlit_app(df):
    df["day_id"] = pd.to_datetime(df["day_id"])
    # side bars
    st.sidebar.title("Input Parameters")
    show_department_button = st.sidebar.checkbox("Total department data")
    show_average_button = st.sidebar.checkbox("Show the mean")
    show_shift_by_mean = st.sidebar.checkbox("Shift_by_mean")
    dpt_num_department = st.sidebar.selectbox(
        "Select Department Number", df["dpt_num_department"].unique()
    )
    but_num_business_unit = st.sidebar.selectbox(
        "Select Business Unit Number",
        df[(df["dpt_num_department"] == dpt_num_department)][
            "but_num_business_unit"
        ].unique(),
    )
    start_date = st.sidebar.date_input("Start Date", df["day_id"].min())
    end_date = st.sidebar.date_input("End Date", df["day_id"].max())
    # input for sidebars
    filtered_df = df[
        (df["but_num_business_unit"] == but_num_business_unit)
        & (df["dpt_num_department"] == dpt_num_department)
        & (df["day_id"] >= pd.to_datetime(start_date))
        & (df["day_id"] <= pd.to_datetime(end_date))
    ]
    # Plot
    st.title("Turnover Visualization")
    if filtered_df.empty:
        st.write("No data available for the selected parameters.")
    else:
        print("dates", filtered_df["day_id"])
        figure, ax = plt.subplots(figsize=(10, 6))
        plt.plot(filtered_df["day_id"], filtered_df["turnover"], color="red")
        plt.bar(filtered_df["day_id"], filtered_df["turnover"], color="skyblue")
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xlabel("Date")
        plt.ylabel("Turnover")
        plt.title("Turnover Over Time")
        plt.xticks(rotation=90)
        st.pyplot(figure)
    if show_department_button:
        st.title("Turnover Visualization for complete department")
        filtered_data = df[
            (df["dpt_num_department"] == dpt_num_department)
            & (df["day_id"] >= pd.to_datetime(start_date))
            & (df["day_id"] <= pd.to_datetime(end_date))
        ]
        figure = data_by_department(
            data_frame=filtered_data,
            department_number=dpt_num_department,
        )
        st.pyplot(figure)
    if show_average_button:
        st.title("Mean of the data")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(
                plot_means(
                    filtered_df,
                    business=but_num_business_unit,
                    department=dpt_num_department,
                    meantype=MeanType.YEAR,
                )
            )
            st.pyplot(
                plot_means(
                    filtered_df,
                    business=but_num_business_unit,
                    department=dpt_num_department,
                    meantype=MeanType.QUATER,
                )
            )
        with col2:
            st.pyplot(
                plot_means(
                    filtered_df,
                    business=but_num_business_unit,
                    department=dpt_num_department,
                    meantype=MeanType.MONTH,
                )
            )
            st.pyplot(
                plot_means(
                    filtered_df,
                    business=but_num_business_unit,
                    department=dpt_num_department,
                    meantype=MeanType.WEEK,
                )
            )
    if show_shift_by_mean:
        st.title("Detrended data")
        time_series = np.asarray(filtered_df["turnover"].values)
        mean_value = np.mean(time_series)
        detrended_data = time_series - mean_value
        figure, ax = plt.subplots(figsize=(10, 6))
        plt.plot(detrended_data, label="Detrended turnover")
        plt.xlabel("Time")
        plt.ylabel("Turnover")
        plt.title("Detrended Time Series Data")
        plt.legend()
        plt.grid(True)
        st.pyplot(figure)
