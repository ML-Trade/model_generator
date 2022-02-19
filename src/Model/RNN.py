from typing import List, Union
from .Model import Model
from enum import Enum
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM
from keras.models import Sequential
from Data.TSDataPreprocessor import Dataset
from tensorflow.python.client import device_lib
import keras


class Architecture (Enum):
    LSTM = LSTM
    GRU = GRU

ArchitectureType = Union[LSTM, GRU, CuDNNGRU, CuDNNLSTM]
class RNN(Model):
    def __init__(self, *,
        layers: List[int],
        x_shape: List[int],
        y_shape: List[int],
        architecture = Architecture.LSTM.value,
        dropout = 0.1,
        is_bidirectional = False,
    ) -> None:
        """
        :param layers: A list of integers. Each integer represents the number of neurons in a layer
        """
        super().__init__()
        # TODO: Test if this is actually needed in newer versions of Tensorflow
        self.Architecture: ArchitectureType = self._use_gpu_if_available(architecture)
        self.is_bidirectional = is_bidirectional
        self.dropout = dropout
        self.layers = layers
        self.model = self._create_model(x_shape, y_shape)

    def load_model(filename: str):
        """
        The file passed is a tarball. It contains the metadata as well as the model / weights files.
        This function extracts the tarball into the temp directory defined by tempfile.gettempdir()
        (failing this we could just make a temp directory, but we then have to worry about cleanup).
        We use these files to init the model.
        """
        pass

    def _use_gpu_if_available(self, architecture: Architecture):
        local_devices = device_lib.list_local_devices()
        gpus = [x.name for x in local_devices if x.device_type == 'GPU']
        if len(gpus) != 0:
            if architecture == GRU:
                architecture = CuDNNGRU
            elif architecture == LSTM:
                architecture = CuDNNLSTM
        return architecture
    
    def _create_model(self, x_shape: List[int], y_shape: List[int]):
        model = Sequential()
        for num_neurons in self.layers:
            layer_template = self.Architecture()
            # Finish with Dense(1, activation = "sigmoid") <- or softmax or tanh etc.

            # TODO: BIG THOUGHT
            # I should create two outputs, one per class (buy | sell). The output is then the probablity
            # of it being in that class
            # This means I need to redo the data preprocessing e.g. data_x or data_y
            # Maybe add a ticket for this in the roadmap



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