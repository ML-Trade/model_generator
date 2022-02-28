from dataclasses import dataclass
from datetime import date, datetime
import json
from msilib.schema import Error
from platform import architecture
from typing import Dict, List, Tuple, Union

from Data.ColumnConfig import ColumnConfig
from .Model import Model
from enum import Enum
from keras.layers import LSTM, GRU, Dense, Input, Bidirectional, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping
from Data.TSDataPreprocessor import Dataset
import keras
import numpy as np
import os
import tarfile
import tempfile

@dataclass
class ModelFileInfo:
    filepath: str
    architecture: str
    layers: str
    loss: float
    timestamp: datetime

ArchitectureType = Union[LSTM, GRU]
def get_architecture_name(architecture_type: ArchitectureType):
    if architecture_type == LSTM: return "LSTM"
    if architecture_type == GRU: return "GRU"
class RNN(Model):

    @staticmethod
    def deconstruct_model_path(filepath: str):
        filename_no_path = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(filename_no_path)[0]
        split_filename = filename_no_ext.split("__")
        architecture = split_filename[0]
        layers = split_filename[1]
        loss = split_filename[2].split('-')[1]
        timestamp = datetime.fromisoformat(split_filename[3].replace(";", ":"))
        return ModelFileInfo(filepath, architecture, layers, float(loss), timestamp)

    def __init__(self, *,
        layers: List[int],
        x_shape: Tuple[int, ...],
        y_shape: Tuple[int, ...],
        architecture: ArchitectureType = LSTM,
        dropout = 0.1,
        is_bidirectional = False,
    ) -> None:
        """
        :param layers: A list of integers. Each integer represents the number of neurons in a layer
        """
        super().__init__()
        # TODO: Test if this is actually needed in newer versions of Tensorflow
        self.Architecture = architecture
        self.is_bidirectional = is_bidirectional
        self.dropout = dropout
        self.layers = layers
        self.model = self._create_model(x_shape, y_shape)
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.score = {
            "loss": np.inf
        }


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
        

    def train(self, dataset: Dataset,
        *,
        max_epochs = 100,
        early_stop_patience = 6,
        batch_size = 1
    ):
        early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        # tensorboard = TensorBoard(log_dir=f"{os.environ['WORKSPACE']}/logs/{self.seq_info}__{self.get_model_info_str()}__{datetime.now().timestamp()}")

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
        self.score = {out: score[i] for i, out in enumerate(self.model.metrics_names)}
        print('Scores:', self.score)
    
    def predict(self, dataset: Dataset):
        predictions = self.model.predict(dataset.val_x)
        correct = 0
        for index, pred in enumerate(predictions):
            if np.argmax([pred[0], pred[1]]) == np.argmax([dataset.val_y[index][0], dataset.val_y[index][1]]): correct += 1
        print(f"Actual accuracy: {correct / len(dataset.val_y)}")


    def load_model(self, filepath: str):
        self.model = keras.models.load_model(filepath)
        print(f"Successfully loaded from {filepath}")
        """
        The file passed is a tarball. It contains the metadata as well as the model / weights files.
        This function extracts the tarball into the temp directory defined by tempfile.gettempdir()
        (failing this we could just make a temp directory, but we then have to worry about cleanup).
        We use these files to init the model.
        """

    def save_model(self, col_config: ColumnConfig):
        """
        Save model locally
        
        Saved model's filenames will include their model type (e.g. RNN) their fitness rating,
        and date/time. Associated files will include metadata such as number of layers,
        each later shape, other stats etc. 
        """

        metadata: dict = {}
        metadata["col_config"] = json.loads(col_config.to_json()) # I need a dict but in the json format
        metadata["layers"] = self.layers
        metadata["architecture"] = get_architecture_name(self.Architecture)
        metadata["dropout"] = self.dropout
        metadata["is_bidirectional"] = self.is_bidirectional
        metadata["x_shape"] = self.x_shape
        metadata["y_shape"] = self.y_shape
        metadata_json = json.dumps(metadata, indent=2)
        
        with tempfile.TemporaryDirectory() as temp_dirname:
            models_folder = os.path.join(os.environ["workspace"], "models")
            os.makedirs(models_folder, exist_ok=True)
            layer_text = "-".join([str(x) for x in self.layers])
            timestamp = datetime.now().isoformat(timespec="seconds").replace(":", ";")
            tar_filename = f"RNN__{layer_text}__Loss-{self.score['loss']:.4f}__{timestamp}.tar"

            model_path = os.path.join(temp_dirname, "model.h5")
            metadata_path = os.path.join(temp_dirname, "metadata.json")
            self.model.save(model_path, save_format="h5")
            with open(metadata_path, "w") as json_file:
                json_file.write(metadata_json)

            with tarfile.open(os.path.join(models_folder, tar_filename), "w") as tar:
                tar.add(model_path, arcname="model.h5")
                tar.add(metadata_path, arcname="metadata.json")
        

        

        