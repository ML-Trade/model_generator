from typing import List
from .Model import Model
from enum import Enum
from tensorflow.keras.layers import LSTM, GRU
from Data.TSDataPreprocessor import Dataset


class Architecture (Enum):
    LSTM = LSTM
    GRU = GRU

class RNN(Model):
    def __init__(self, layers: List[int],
        *,
        architecture = Architecture.LSTM.value,
        dropout = 0.1,
        is_bidirectional = False,
    ) -> None:
        """
        :param layers: A list of integers. Each integer represents the number of neurons in a layer
        """
        super().__init__()

    def train(self, dataset: Dataset,
        *,
        max_epochs = 100,
        early_stop_patience = 6,
        batch_size = 1
    ):
        pass
    
    def predict(self):
        return super().predict()
    
    def load_data(self):
        pass

    def save_model(self):
        """
        Save model locally
        
        Saved model's filenames will include their model type (e.g. RNN) their fitness rating,
        and date/time. Associated files will include metadata such as number of layers,
        each later shape, other stats etc. 
        """
        pass