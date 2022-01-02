import pandas as pd

class TSDataPreprocessor():
    """
    Time-Series Data Preprocessor (for use with RNN)
    """
    def __init__(self, raw_data: pd.DataFrame, *,
        forecast_col_name: str,
        sequence_length = 100,
        forecast_period = 10
    ):
        """
        If preprocessed data already exists in the root data folder, it will be loaded, and preprocessing will be skipped
        We will check if it already exists by obtaining a hash of the raw_data dataframe, and comparing it to the hash saved in
        the name of the file. This will mean the same raw data was once passed before.
        """
        self.raw_data = raw_data
        self.df = raw_data.copy()
        self.forecast_col_name = forecast_col_name
        self.sequence_length = sequence_length
        self.forecast_period = forecast_period
        data_hash = hex(hash(raw_data.to_numpy().tobytes()))[2:8]

    
    def preprocess(self):
        """
        Notes:
        Somewhere here you will need to save some metadata so you can preprocess new data, not just a dataset.
        E.g. row standard deviations, means, percentiles etc.
        
        Preprocessing Volume:
        Add overall average to all volume
        Minus Long Moving Average of Volume (maybe 200)
        MinMax it between -1 and 1

        Date should be split into day of the week, day in the month, and how far through the year,
        then minmaxed between -1 and 1

        All else is standardised
        """
        pass