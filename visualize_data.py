from pathlib import Path

from src.helpers.data import load_data_to_dataframe
from src.helpers.streamlit_data import streamlit_app

trainig_data_frame = load_data_to_dataframe(Path("training_data/train.parquet"))
print(trainig_data_frame.head())

if __name__ == "__main__":
    streamlit_app(df=trainig_data_frame)
