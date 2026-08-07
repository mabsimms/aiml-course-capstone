import keras_tuner as kt
import numpy as np
import keras

from capstone.dnn import build_dnn_model

_HP_DISPATCH = { 
    "int": "Int",
    "float": "Float",
    "choice": "Choice"
}

class SpamHyperModel(kt.HyperModel):
    def __init__(
            self,
            train_text: np.ndarray,
            train_features: np.ndarray,
            search_space: dict,
            fixed_hyperparameters: dict | None = None
    ):
        self.train_text = train_text
        self.train_features = train_features
        self.search_space = search_space
        self.fixed_hyperparameters = fixed_hyperparameters

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        sampled = {}
        for name, spec in self.search_space.items():
            spec = dict(spec)
            method_name = _HP_DISPATCH[spec.pop("type")]
            sampled[name] = getattr(hp, method_name)(name, **spec)
        hyperparams = { **sampled, **self.fixed_hyperparameters }
        return build_dnn_model(self.train_text, self.train_features, **hyperparams)