from datetime import datetime
from typing import List
from mltradeshared import ColumnConfig, NormFunction, TSDataPreprocessor, RNN, ModelFileInfo, DatasetMetadata, TimeMeasurement
from mltradeshared.Trade import TradeManager
from os import environ, path
from Data import DataUpdater
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from keras.layers import GRU
from Trade import TradeSimulator


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
        forecast_period = TimeMeasurement("minute", 20),
        sequence_length = 150,
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
        is_bidirectional=False,
        dropout=0.1,
        architecture=GRU,
        x_shape=dataset.train_x.shape,
        y_shape=dataset.train_y.shape 
    )
    rnn.train(
        dataset,
        batch_size=2048,
        max_epochs=3,
        early_stop_patience=10
    )

    model_path = rnn.save_model(col_config, dataset, dataset_metadata)
    rnn, metadata = rnn.load_model(model_path, return_metadata=True)

    tm = TradeManager(
        balance=100000.0,
        max_trade_time=TimeMeasurement("minute", 35),
        trade_cooldown=TimeMeasurement("minute", 10),
        risk_per_trade=0.02,
        max_open_risk=0.06,
        dif_percentiles=metadata["dif_percentiles"]["data"],
        fraction_to_trade=0.051231,
        max_trade_history=None,
        stop_loss_ATR=1,
        take_profit_ATR=2
    )

    val_df = df.iloc[:-int(len(df) * dataset_metadata.train_split)]
    trade_sim = TradeSimulator(
        val_df, dataset.val_x, tm, rnn
    )
    
    trade_sim.start()
    # Execution time, profit, number of trades, number of wins, number of losses, w/l ratio. Average win%, average loss%
    trade_sim.summary()
    
    # # rnn.predict(dataset)

    # data_updater.upload_to_drive(model_path, f"models/{file_name}")

    # filepath = get_filepath()
    # rnn.load_model(filepath)
    


if __name__ == "__main__":
    main()