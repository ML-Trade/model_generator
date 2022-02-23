from typing import List, Tuple, Union
from .Model import Model
from enum import Enum
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Dense, Input, Bidirectional, Dropout, BatchNormalization, Flatten
from keras.callbacks import TensorBoard, EarlyStopping
from Data.TSDataPreprocessor import Dataset
from tensorflow.python.client import device_lib
import keras
import numpy as np


class Architecture (Enum):
    LSTM = LSTM
    GRU = GRU

ArchitectureType = Union[LSTM, GRU, CuDNNGRU, CuDNNLSTM]
class RNN(Model):
    def __init__(self, *,
        layers: List[int],
        x_shape: Tuple[int, ...],
        y_shape: Tuple[int, ...],
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
    
    def _create_model(self, x_shape: Tuple[int, ...], y_shape: Tuple[int, ...]):
        def get_layer_template(num_neurons: int, return_sequences: bool):
            # Don't return sequences on last one since otherwise dense layer returns multi dimensional tensor, not single output
            if not self.is_bidirectional: return self.Architecture(num_neurons, return_sequences=return_sequences) 
            else: return Bidirectional(self.Architecture(num_neurons, return_sequences=return_sequences))
        
        model_layers = []
        model_layers.append(Input(shape=(x_shape[1:]))) # input_shape[0] is just len(x_shape)
        for index, num_neurons in enumerate(self.layers):
            return_sequences = index < len(self.layers) - 1
            LayerTemplate = get_layer_template(num_neurons, return_sequences)
            model_layers.append(LayerTemplate(model_layers[-1]))
            model_layers.append(Dropout(self.dropout)(model_layers[-1]))
            model_layers.append(BatchNormalization()(model_layers[-1]))

        num_classes = y_shape[1]
        model_layers.append(Dense(num_classes, activation="sigmoid")(model_layers[-1]))

        model = keras.models.Model(inputs = model_layers[0], outputs = model_layers[-1])

        model.compile(
            loss=["categorical_crossentropy"], 
            optimizer="adam",
            metrics=["categorical_crossentropy", "accuracy"]
        )
        
        return model
        
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
        early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        training_history = self.model.fit(
            x=dataset.train_x,
            y=dataset.train_y,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_data=(dataset.val_x, dataset.val_y),
            # callbacks=[tensorboard, early_stop],
            callbacks=[early_stop],
            shuffle=True
        )

        score = self.model.evaluate(dataset.val_x, dataset.val_y, verbose=0)
        score = {out: score[i] for i, out in enumerate(self.model.metrics_names)}
        print('Scores:', score)
    
    def predict(self, dataset: Dataset):
        predictions = self.model.predict(dataset.val_x)
        correct = 0
        for index, pred in enumerate(predictions):
            if np.argmax([pred[0], pred[1]]) == np.argmax([dataset.val_y[index][0], dataset.val_y[index][1]]): correct += 1
        print(f"Actual accuracy: {correct / len(dataset.val_y)}")

    
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