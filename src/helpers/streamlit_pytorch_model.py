import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.helpers.computations import Validators
from src.helpers.plotting import (
    plot_predection_vs_test_data,
    plot_training_data,
    plot_turnover,
)
from src.models.enums import MeanType
from src.predection_model.data_preparation import forecast, get_training_vectors
from src.predection_model.pytorch_model import NonLinearRegression, trian_model


def streamli_app_for_model(df: pd.DataFrame):
    df["day_id"] = pd.to_datetime(df["day_id"])
    st.title("PyTorch Model Visualizer")
    st.sidebar.title("Input Parameters")
    epochs = st.sidebar.number_input("Number of epochs", min_value=1, value=10)
    learning_rate = st.sidebar.number_input(
        "Enter learning rate", min_value=0.0001, value=0.001, format="%.5f"
    )
    split_ratio = st.sidebar.number_input(
        "Enter train data split", min_value=0.5, value=0.8, format="%.5f"
    )
    num_hidden_neurons = st.sidebar.number_input(
        "Enter number of hidden neurons", min_value=1, value=100
    )
    num_department = st.sidebar.selectbox(
        "Select Department Number", df["dpt_num_department"].unique()
    )
    num_business_unit = st.sidebar.selectbox(
        "Select Business Unit Number",
        df[(df["dpt_num_department"] == num_department)][
            "but_num_business_unit"
        ].unique(),
    )
    start_date = st.sidebar.date_input("Start Date", df["day_id"].min())
    end_date = st.sidebar.date_input("End Date", df["day_id"].max())
    forcast_time = st.sidebar.number_input(
        "Enter forcast time in weeks", min_value=1, value=8
    )
    filtered_df = df[
        (df["but_num_business_unit"] == num_business_unit)
        & (df["dpt_num_department"] == num_department)
        & (df["day_id"] >= pd.to_datetime(start_date))
        & (df["day_id"] <= pd.to_datetime(end_date))
    ]
    st.title("Turnover data")
    st.pyplot(plot_turnover(filtered_df))
    model_data = get_training_vectors(
        data_frame=filtered_df,
        normalize_data=True,
        department=num_department,
        business=num_business_unit,
        split_ratio=split_ratio,
    )
    run_model = st.sidebar.checkbox("Run model")
    show_prediction = st.sidebar.checkbox("Show Predictions")
    run_forcaste = st.sidebar.checkbox("Run forecast")
    if run_model:
        model_eval, predictions, loss_values = trian_model(
            model_data=model_data,
            model=NonLinearRegression,
            epochs=epochs,
            learing_rate=learning_rate,
            hidden_neurons=num_hidden_neurons,
        )

    if show_prediction:
        st.title("##Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.write("###Training data")
            st.pyplot(
                plot_training_data(
                    x_train=np.asarray(model_data.train_data.x_train),
                    y_train=np.asarray(model_data.train_data.y_train),
                )
            )
        with col2:
            st.write("###Validation")
            figure = plot_predection_vs_test_data(
                predictions=predictions, model_data=model_data, denormalize=True
            )
            st.pyplot(figure)
            validator = Validators(
                y_true=np.asarray(model_data.test_data.y_test), y_pred=predictions
            )
            st.write("## Validation Parameters")
            valiation_data = {
                "Parameter": ["MAE", "MSE", "RMSE", "R2 Score"],
                "Value": [
                    validator.compute_mae(),
                    validator.compute_mse(),
                    validator.compute_rmse(),
                    validator.compute_r2_score(),
                ],
            }
            st.table(valiation_data)
    if run_forcaste:
        predicted_data = forecast(
            forcat_time_inweeks=forcast_time,
            data_frame=filtered_df,
            model=model_eval,
            department=num_department,
            business=num_business_unit,
            meantype=MeanType.WEEK,
            model_data=model_data,
        )
        figure, ax = plt.subplots()
        plt.plot(
            predicted_data["day_id"],
            predicted_data["turnover"],
            color="red",
            label="forecast",
        )
        plt.plot(
            filtered_df["day_id"],
            filtered_df["turnover"],
            alpha=0.5,
            color="skyblue",
            label="real",
        )
        plt.xlabel("Date")
        plt.ylabel("Turnover")
        plt.title("Turnover Over Time")
        plt.legend()
        plt.xticks(rotation=90)
        st.pyplot(figure)
