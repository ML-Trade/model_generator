from typing import Dict, Callable
import pandas as pd

class TSDataPreprocessor():
    """
    Time-Series Data Preprocessor (for use with RNN)
    """
    def __init__(self, raw_data: pd.DataFrame, *,
        target_col_name: str,
        sequence_length = 100,
        forecast_period = 10,
        custom_norm_functions: Dict[str, Callable[[pd.Series], pd.Series]] = {},
    ):
        """
        If preprocessed data already exists in the root data folder, it will be loaded, and preprocessing will be skipped
        We will check if it already exists by obtaining a hash of the raw_data dataframe, and comparing it to the hash saved in
        the name of the file. This will mean the same raw data was once passed before.

        @param: custom_norm_functions : a dictionary of column names and their associated normalisation function. Any column
        not specified uses the default normalisation function, which is percentage change, then standardisation.
        
        """
        self.raw_data = raw_data
        self.df = raw_data.copy()
        self.target_col_name = target_col_name
        self.sequence_length = sequence_length
        self.forecast_period = forecast_period
        self.custom_norm_functions = custom_norm_functions
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


        # Convert to pct change
        keys = self.custom_norm_functions.keys()
        for col_name in self.df:
            new_col
            if col_name not in keys:
                new_col = self.default_norm(self.df[col_name])
            else:
                new_col = self.custom_norm_functions[col_name](self.df[col_name])
            self.df[col_name] = new_col
    
        # Add Target

        self.df[self.target_col_name] = target_col

        # Standardise / Normalise (maybe pass these functions in?)

        # Balance

        # Convert into numpy sequences

        # Split into training and validation

        # Save data

        # Shuffle training set 
        
        pass