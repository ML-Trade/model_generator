from datetime import datetime
from Model import RNN
from os import environ, path
from Data import DataUpdater, TSDataPreprocessor
import numpy as np
import pandas as pd



## TODO: Move me to utils
def moving_average(series: pd.Series, period: int) -> pd.Series :
    """
    Does not remove NaNs. This must be done manually after
    """
    return series.rolling(period, center=False).mean()
    

def main():
    file_path = path.dirname(__file__)
    workspace_dir = path.join(file_path, "..")
    environ["workspace"] = path.realpath(workspace_dir)
    print(environ["workspace"])

    data_updater = DataUpdater()
    print(f"GitHub Token: {data_updater.github_token}")
    print(f"Google Drive Token: {data_updater.google_drive_token}")
    print(f"Polygon.io Token: {data_updater.polygonio_token}")

    df = data_updater.get_required_data(
        "EURUSD",
        start = datetime(2021, 9, 16),
        end = datetime.now(),
        multiplier = 1,
        measurement = "minute"
    )
    df.drop(columns=["vw", "n"], inplace=True)
    
    print(df)
    print("Preprocessing Data...")

    VOLUME_MA_PERIOD = 5
    preprocessor = TSDataPreprocessor()
    pd.set_option('display.max_columns', None)
    preprocessor.preprocess(df,
        target_col_name="c",
        sequence_length=100,
        forecast_period=10,
        time_col_name="t",
        custom_pct_change={
            "v": lambda series: moving_average(series, VOLUME_MA_PERIOD).pct_change()
    })
    


if __name__ == "__main__":
    main()