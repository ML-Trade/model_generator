from datetime import datetime
from os import times
import re
from turtle import Turtle
from typing import Dict, Callable, Union
import pandas as pd
import numpy as np

##  TODO: move this to utils
def minmax_norm(array: Union[list, np.ndarray]) -> pd.Series:
    np_array = np.array(array)
    return pd.Series((np_array - np.min(np_array)) / (np.max(array) - np.min(array)))

def standardise(arr: np.ndarray):
    return (arr - np.mean(arr)) / np.std(arr)

class TSDataPreprocessor():
    """
    Time-Series Data Preprocessor (for use with RNN)
    """
    def __init__(self, raw_data: pd.DataFrame, *,
        target_col_name: str,
        sequence_length = 100,
        forecast_period = 10,
        time_col_name = None,
        custom_pct_change: Dict[str, Callable[[pd.Series], pd.Series]] = {},
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
        self.custom_pct_change = custom_pct_change
        self.time_col_name = time_col_name
        data_hash = hex(hash(raw_data.to_numpy().tobytes()))[2:8]

    
    def preprocess(self):
        """
        Notes:
        Somewhere here you will need to save some metadata so you can preprocess new data, not just a dataset.
        E.g. row standard deviations, means, percentiles etc.
        
        Preprocessing Volume:
        Make it a moving average (between 3 and 200 picked by GA)
        Then have it as percent change and standardised
        We best percieve volume as how it changes / slopes. This will best capture this

        All else is standardised
        """


        # Remove time column for later handling
        time_col = None
        if self.time_col_name is not None:
            time_col = self.df[self.time_col_name].to_numpy()
            self.df.drop(columns=[self.time_col_name], inplace=True)

        # Convert to pct change
        
        keys = self.custom_pct_change.keys()
        for col_name in self.df:
            new_col = None
            if col_name not in keys:
                new_col = self.df[col_name].pct_change()
            else:
                new_col = self.custom_pct_change[col_name](self.df[col_name])
            self.df[col_name] = new_col
        self.df.dropna(inplace=True)

        # Handle time data
        if time_col is not None:   
            time_of_day_col = []
            day_of_week_col = []
            week_of_year_col = []
            for val in time_col:
                timestamp = datetime.fromtimestamp(val / 1000)
                time_of_day_col.append(timestamp.second + (timestamp.minute * 60) + (timestamp.hour * 60 * 60))
                day_of_week_col.append(timestamp.weekday())
                week_of_year_col.append(timestamp.isocalendar().week)
            
            self.df["time_of_day"] = minmax_norm(time_of_day_col)
            self.df["day_of_week"] = minmax_norm(day_of_week_col)
            self.df["week_of_year"] = minmax_norm(week_of_year_col)

        # Add Target (target can be added after since its classification)
        target = []
        raw_target_col = self.df[self.target_col_name]
        for index, value in raw_target_col.items():
            try:
                if value < raw_target_col[index + self.forecast_period]:
                    ## TODO: Enum or globalise this?
                    target.append(1) # Buy
                else:
                    # NOTE: This may have a slight bias to selling; if theyre equal target is sell
                    target.append(0) # Sell
            except:
                target.append(np.nan)
        self.df["target"] = target
        self.df.dropna(inplace=True)

        # Standardise / Normalise (maybe pass these functions in?)
        std_exceptions = ["target", "time_of_day", "day_of_week", "week_of_year"] # Don't std these
        for col in self.df.columns:
            if col not in std_exceptions:
                self.df[col] = standardise(self.df[col].to_numpy())

        # Balance

        # Cleanup (remove NA etc)

        self.df.dropna(inplace=True)

        # Convert into numpy sequences

        # Split into training and validation

        # Save data

        # Shuffle training set 
        
        print("Values after preprocessing:")
        print(self.df)