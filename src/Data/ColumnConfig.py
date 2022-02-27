from typing import Any, Dict
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
            name: "SOME_NAME"
            norm_function: "std | ma_std | minmax | time"
            period: 69
        },
        ...
    ]
    ```
    The order of the array is important; this is the order of the columns in the df / numpy sequence array
    
    "period" is optional and is used for ma_std (moving average then pct_change then standardise)
    
    """
    def __init__(self) -> None:
        self.default_function = NormFunction.STD
        self.default_function_args: Dict[str, Any] = {}

    def set_default_norm_function(self, norm_function: NormFunction, **args):
        self.default_function = norm_function
        self.default_function_args = args

    def to_json(self) -> str:
        pass

    @staticmethod
    def from_json(json_string: str):
        return ColumnConfig()