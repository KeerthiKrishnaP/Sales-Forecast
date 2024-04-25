from pathlib import Path

from src.helpers.data import load_data_to_dataframe
from src.helpers.streamlit_pytorch_model import streamli_app_for_model

trainig_data_frame = load_data_to_dataframe(Path("training_data/train.parquet"))

if __name__ == "__main__":
    streamli_app_for_model(df=trainig_data_frame)
