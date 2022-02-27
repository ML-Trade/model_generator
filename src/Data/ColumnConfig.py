import json
from typing import Any, Dict, List
from collections import OrderedDict
from enum import Enum

class NormFunction(Enum):
    STD = "std"
    MA_STD = "ma_std"
    MINMAX = "minmax"
    TIME = "time"


class ColumnConfig:
    """
    The configuration associated with preprocessing a dataframe for each column.
    This is to be passed into a Data Preprocessor such as TSDataPreprocessor.

    JSON Format:
    ```
    [
        {
            name: "SOME_NAME",
            norm_function: "std | ma_std | minmax | time",
            period: 69,
            class_meanings: ["SELL", "BUY"]
        },
        ...
    ]
    ```
    The order of the array is important; this is the order of the columns in the df / numpy sequence array
    
    "period" is optional and is used for ma_std (moving average then pct_change then standardise)
    "class_meanings" is optional. It is only relevant when the name is "target".
    
    """
    def __init__(self, columns: List[str], class_meanings: List[str]) -> None:
        self.config = OrderedDict()
        for col in columns:
            self.config[col] = {}
        self.config["target"] = {}
        self.config["target"]["class_meanings"] = class_meanings
        self.default_function = NormFunction.STD
        self.default_function_args: Dict[str, Any] = {}

    def set_default_norm_function(self, norm_function: NormFunction, args: dict = {}):
        self.default_function = norm_function
        self.default_function_args = args

    def to_dict(self) -> OrderedDict:
        config: OrderedDict[str, Any] = OrderedDict()
        for col_name, value in self.config.items():
            config[col_name] = {}
            if "norm_function" not in value and col_name != "target":
                config[col_name]["norm_function"] = self.default_function
                config[col_name] = {**self.default_function_args, **config[col_name]}
            else:
                config[col_name] = value
        return config
                
    def set_norm_function(self, name: str, norm_function: NormFunction, args: dict = {}):
        self.config[name]["norm_function"] = norm_function
        self.config[name] = {**args, **self.config[name]}

    def to_json(self) -> str:
        config = self.to_dict()
        # Saved as an array to maintain ordering when serialising and deserialising JSON
        config_arr = []
        for col_name, value in config.items():
            config_value = value
            config_value["name"] = col_name
            if col_name != "target":
                config_value["norm_function"] = config_value["norm_function"].value
            config_arr.append(config_value)
        return json.dumps(config_arr)

    @staticmethod
    def from_json(json_string: str):
        config_arr: List[dict] = json.loads(json_string)
        colname_arr = []
        class_meanings = []
        for val in config_arr:
            colname_arr.append(val["name"])
            if val["name"] == "target":
                class_meanings = val["class_meanings"]
        col_config = ColumnConfig(columns=colname_arr, class_meanings=class_meanings)
        for val in config_arr:
            if val["name"] == "target":
                continue
            val_args = val.copy()
            del val_args["name"]
            col_config.set_norm_function(val["name"], NormFunction(val["norm_function"]), val_args)
        return col_config