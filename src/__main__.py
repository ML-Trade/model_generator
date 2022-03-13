from datetime import datetime
from typing import List
from mltradeshared import ColumnConfig, NormFunction, TSDataPreprocessor, RNN, ModelFileInfo, DatasetMetadata, TimeMeasurement
from os import environ, path
from Data import DataUpdater
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf


def get_filepath():
    models_folder = os.path.join(os.environ["workspace"], "models")
    os.makedirs(models_folder, exist_ok=True)
    all_file_info: List[ModelFileInfo] = []
    for file_path in glob.glob(os.path.join(models_folder, "*.h5")):
        file_info = RNN.deconstruct_model_path(file_path)
        all_file_info.append(file_info)
    all_file_info.sort(key=lambda x: x.timestamp)
    return all_file_info[-1].filepath



def main():
    file_path = path.dirname(__file__)
    workspace_dir = path.join(file_path, "..")
    environ["workspace"] = path.realpath(workspace_dir)
    print(environ["workspace"])

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    data_updater = DataUpdater()
    print(f"GitHub Token: {data_updater.github_token}")
    print(f"Google Drive Token: {data_updater.google_drive_token}")
    print(f"Polygon.io Token: {data_updater.polygonio_token}")

    dataset_metadata = DatasetMetadata(
        symbol="EURUSD",
        start = datetime(2021, 9, 16),
        end = datetime(2022, 1, 1),
        candle_time = TimeMeasurement("minute", 1),
        forecast_period = TimeMeasurement("minute", 10),
        sequence_length = 80,
        train_split = 0.8
    )

    df = data_updater.get_ohlc_data(dataset_metadata)
    df.drop(columns=["vw", "n"], inplace=True)
    
    print(df)
    print("Preprocessing Data...")

    VOLUME_MA_PERIOD = 5
    pd.set_option('display.max_columns', None)
    col_config = ColumnConfig(df.columns, target_column="c", class_meanings=["BUY", "SELL"])
    col_config.set_norm_function("t", NormFunction.TIME)
    col_config.set_norm_function("v", NormFunction.MA_STD, {"period": VOLUME_MA_PERIOD})
    preprocessor = TSDataPreprocessor(col_config=col_config)
    
    dataset = preprocessor.preprocess(df,
        col_config=col_config,
        dataset_metadata=dataset_metadata
    )

    rnn = RNN(
        layers=[60, 60, 60],
        x_shape=dataset.train_x.shape,
        y_shape=dataset.train_y.shape 
    )
    rnn.train(
        dataset,
        batch_size=5000,
        max_epochs=3
    )
    # # rnn.predict(dataset)
    model_path = rnn.save_model(col_config, dataset=dataset, dataset_metadata=dataset_metadata)
    file_name = os.path.basename(model_path)
    data_updater.upload_to_drive(model_path, f"models/{file_name}")

    # filepath = get_filepath()
    # rnn.load_model(filepath)
    


if __name__ == "__main__":
    main()