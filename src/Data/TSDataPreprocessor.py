from collections import deque
from datetime import datetime
import os
from typing import Deque, Dict, Callable, List, Union
from isort import file
import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object
import hashlib
import glob
from dataclasses import dataclass


##  TODO: move this to utils
def minmax_norm(array: Union[list, np.ndarray]) -> pd.Series:
    np_array = np.array(array)
    return pd.Series((np_array - np.min(np_array)) / (np.max(array) - np.min(array)))

def standardise(arr: np.ndarray):
    return (arr - np.mean(arr)) / np.std(arr)

@dataclass
class PreprocessedFileInfo:
    dataset_type: str
    data_hash: str
    title: str
    sequence_length: int
    forecast_period: int

@dataclass
class Datasets:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray

class TSDataPreprocessor():
    """
    Time-Series Data Preprocessor (for use with RNN)
    """
    def __init__(self):
        """       
        If preprocessed data already exists in the root data folder, it will be loaded, and preprocessing will be skipped
        We will check if it already exists by obtaining a hash of the raw_data dataframe, and comparing it to the hash saved in
        the name of the file. This will mean the same raw data was once passed before.

        @param: custom_norm_functions : a dictionary of column names and their associated normalisation function. Any column
        not specified uses the default normalisation function, which is percentage change, then standardisation.
        
        Shuffling must be done manually or passed as a param to preprocess. Saved datasets are NOT shuffled

        The dataframe caption will be used in the file name when saving the dataset file. Set the dataframe caption with df.style.set_caption("my caption")
        """
    @staticmethod
    def deconstruct_filename(filename: str):
        filename_no_path = os.path.basename(filename)
        filename_no_ext = os.path.splitext(filename_no_path)[0]
        split_filename = filename_no_ext.split("__")
        dataset_type = split_filename[0]
        data_hash = split_filename[1]
        title = split_filename[2]
        sequence_length = int(split_filename[3].split('-')[-1])
        forecast_period = int(split_filename[4].split('-')[-1])
        return PreprocessedFileInfo(dataset_type, data_hash, title, sequence_length, forecast_period)

    @staticmethod
    def get_preprocessed_filename(data_hash: str, df_title: str, sequence_length: int, forecast_period: int):
        return f"{data_hash}__{df_title}__SeqLen-{sequence_length}__Forecast-{forecast_period}.npy"


    
    def preprocess(self, raw_data: pd.DataFrame, *,
        target_col_name: str,
        sequence_length = 100,
        forecast_period = 10,
        train_split = 0.8,
        time_col_name = None,
        custom_pct_change: Dict[str, Callable[[pd.Series], pd.Series]] = {}
    ) -> Datasets:
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
        df = raw_data.copy()
        df_title = df.style.caption or "~"
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(raw_data, index=True).values).hexdigest()[2:8] # First 6 hex chars
        train_folder = os.path.join(os.environ["workspace"], "data", "preprocessed", "train")
        val_folder = os.path.join(os.environ["workspace"], "data", "preprocessed", "validation")
        for file_path in glob.glob(os.path.join(train_folder, "*.npy")):
            file_info = self.deconstruct_filename(file_path)
            if file_info.data_hash == data_hash:
                print("Same hash; this dataset has been preprocessed before. Using old version")
                filename_template = self.get_preprocessed_filename(file_info.data_hash, file_info.title, file_info.sequence_length, file_info.forecast_period)
                train_x = np.load(os.path.join(train_folder, f"train-x__{filename_template}"))
                train_y = np.load(os.path.join(train_folder, f"train-y__{filename_template}"))
                val_x = np.load(os.path.join(val_folder, f"val-x__{filename_template}"))
                val_y = np.load(os.path.join(val_folder, f"val-y__{filename_template}"))
                return Datasets(train_x, train_y, val_x, val_y)



        # Remove time column for later handling
        time_col = None
        if time_col_name is not None:
            time_col = df[time_col_name].to_numpy()
            df.drop(columns=[time_col_name], inplace=True)

        # Convert to pct change
        
        keys = custom_pct_change.keys()
        for col_name in df:
            new_col = None
            if col_name not in keys:
                new_col = df[col_name].pct_change()
            else:
                new_col = custom_pct_change[col_name](df[col_name])
            df[col_name] = new_col
        df.dropna(inplace=True)

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
            
            df["time_of_day"] = minmax_norm(time_of_day_col)
            df["day_of_week"] = minmax_norm(day_of_week_col)
            df["week_of_year"] = minmax_norm(week_of_year_col)

        # Add Target (target can be added after since its classification)
        target = []
        raw_target_col = df[target_col_name]
        for index, value in raw_target_col.items():
            try:
                if value < raw_target_col[index + forecast_period]:
                    ## TODO: Enum or globalise this?
                    target.append(1.0) # Buy
                else:
                    # NOTE: This may have a slight bias to selling; if theyre equal target is sell
                    target.append(0.0) # Sell
            except:
                target.append(np.nan)
        df["target"] = target
        df.dropna(inplace=True)

        # Standardise / Normalise (maybe pass these functions in?)
        std_exceptions = ["target", "time_of_day", "day_of_week", "week_of_year"] # Don't std these
        for col in df.columns:
            if col not in std_exceptions:
                df[col] = standardise(df[col].to_numpy())


        # Cleanup (remove NA etc)

        df.dropna(inplace=True)

        # Convert into numpy sequences
        # [
        #    [[sequence1], target1]
        #    [[sequence2], target2]
        # ]  
        sequences = [] 
        cur_sequence: Deque = deque(maxlen=sequence_length)
        target_index = df.columns.get_loc("target")
        for index, value in enumerate(df.to_numpy()):
            # Since value is only considered a single value in the sequence (even though itself is an array), to make it a sequence, we encapsulate it in an array so:
            # sequence1 = [[values1], [values2], [values3]]
            val_without_target = np.concatenate((value[:target_index], value[target_index + 1:]))
            cur_sequence.append(val_without_target) # Append all but target to cur_sequence
            if len(cur_sequence) == sequence_length:
                seq = list(cur_sequence)
                sequences.append([np.array(seq), value[target_index]]) # value[-1] is the target        
        

        data_x_list = []
        data_y_list = []
        for seq, target in sequences:
            data_x_list.append(seq)
            data_y_list.append(target)
        
        data_x = np.array(data_x_list)
        data_y = np.array(data_y_list)

        # Balance
        target_index = list(df.columns).index("target")
        group_indices: Dict[str, List[int]] = {}
        groups, counts = np.unique(data_y, return_counts=True)
        for index, target in enumerate(data_y):
            for group in groups:
                if group not in group_indices: group_indices[group] = []
                if target == group: group_indices[group].append(index)

        for group, indices in group_indices.items():
            np.random.shuffle(group_indices[group]) # Shuffle removal order
            dif = len(indices) - np.min(counts)
            for i in range(dif):
                index = group_indices[group].pop()
                data_x[index] = np.full(data_x[index].shape, np.nan)
                data_y[index] = np.nan
        
        data_x = data_x[~np.isnan(data_x)].reshape(-1, *data_x.shape[1:])
        data_y = data_y[~np.isnan(data_y)]
                

        # Split into training and validation

        train_x, val_x = np.split(data_x, [int(train_split * len(data_x))])
        train_y, val_y = np.split(data_y, [int(train_split * len(data_y))])

        # Save data

        folder = os.path.join(os.environ["workspace"], "data", "preprocessed")
        filename = self.get_preprocessed_filename(data_hash, df_title, sequence_length, forecast_period)
        np.save(os.path.join(folder, "train", f"train-x__{filename}"), train_x)
        np.save(os.path.join(folder, "train", f"train-y__{filename}"), train_y)
        np.save(os.path.join(folder, "validation", f"val-x__{filename}"), val_x)
        np.save(os.path.join(folder, "validation", f"val-y__{filename}"), val_y)

        # Shuffle training set 

        # TODO: COME BACK TO SHUFFLE LATER
        # random.shuffle(sequences) # Shuffle sequences to avoid order effects on learning
        print("Values after preprocessing:")
        print(df)

        return Datasets(train_x, train_y, val_x, val_y)