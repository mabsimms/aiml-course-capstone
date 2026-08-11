 # Reference - https://keras.io/keras_tuner/getting_started/

import keras_tuner as kt
import numpy as np
import pandas as pd
import keras

import logging
logger = logging.getLogger("capstone")

from capstone.dnn import build_dnn_model, _encode_text
from pathlib import Path

from sklearn.utils.class_weight import compute_class_weight
from capstone.tokenization.tokenization import train_tokenizer, PAD_TOKEN

_HP_DISPATCH = { 
    "int": "Int",
    "float": "Float",
    "choice": "Choice"
}

# Ignore these tokenizer specific keys during tuning runs
_TOKENIZER_KEYS = { 
    "max_tokens",    
    "min_frequency"
}

class SpamHyperModel(kt.HyperModel):
    # HyperModel subclassing (build(self, hp) contract): https://keras.io/keras_tuner/api/hypermodels/
    # Config-driven search space: https://keras.io/keras_tuner/guides/tailor_the_search_space/
    def __init__(
            self,
            encoded_text: np.ndarray,
            numeric_features: np.ndarray,
            vocab_size: int,
            search_space: dict,
            fixed_hyperparameters: dict | None = None
    ):
        self.encoded_text = encoded_text
        self.numeric_features = numeric_features
        self.vocab_size = vocab_size
       
        self.search_space = search_space

        # Filter out tokenizer keys
        self.fixed_hyperparameters = {
            k: v for k, v in (fixed_hyperparameters or {}).items() if k not in _TOKENIZER_KEYS
        } 

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        sampled = {}
        for name, spec in self.search_space.items():
            spec = dict(spec)
            method_name = _HP_DISPATCH[spec.pop("type")]
            sampled[name] = getattr(hp, method_name)(name, **spec)
        hyperparams = { **sampled, **self.fixed_hyperparameters }
        return build_dnn_model(self.numeric_features, self.vocab_size, **hyperparams)


def tune_dnn(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    search_space: dict,
    fixed_hyperparameters: dict,
    max_trials: int = 15,
    epochs: int = 15,
    tuner_dir : Path = Path("artifacts/tuner"),
    project_name : str = "dnn_search",
    overwrite : bool = False,
    verbose : int = 1,
) -> dict:
    if verbose > 0:
        callback_verbose = 1
    else:
        callback_verbose = 0
        
    class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(df_train["Label"]),
            y=df_train["Label"]
    )      
    class_weight_dict = dict(zip(np.unique(df_train["Label"]), class_weights))
    train_features = df_train[feature_cols].to_numpy(dtype=np.float64)

    max_tokens = fixed_hyperparameters.get("max_tokens", 20_000)
    output_sequence_length = fixed_hyperparameters.get("output_sequence_length", 4_500)
    min_frequency = fixed_hyperparameters.get("min_frequency", 0)

    tokenizer = train_tokenizer(
        df_train["text"],
        vocab_size=max_tokens,
        output_sequence_length=output_sequence_length,
        min_frequency=min_frequency
    )
    # Ensure that the padding token is the first in the vocabulary set
    assert tokenizer.token_to_id(PAD_TOKEN) == 0, "Expected PAD token id 0"
    encoded_text = _encode_text(tokenizer, df_train["text"])

    logger.info("Starting keras tuner search; max_trials=%s, epochs=%s, search_space=%s", 
                max_trials, epochs, list(search_space.keys()))

    hypermodel = SpamHyperModel(
        encoded_text, train_features, tokenizer.get_vocab_size(), search_space, fixed_hyperparameters
    )

    tuner = kt.RandomSearch(
        hypermodel,
        objective="val_loss",
        max_trials=max_trials,
        directory=str(tuner_dir),
        project_name=project_name,
        overwrite=overwrite
    )

    early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                    verbose = callback_verbose
    )

    tuner.search(
         {
            "text": encoded_text,
            "engineered_features": train_features
        },
        df_train["Label"],
        validation_split=0.1,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=verbose
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    return { 
        "hyperparameters": { **best_hp.values, **fixed_hyperparameters },
        "best_val_loss": best_trial.score,
        "trials_completed": len(tuner.oracle.trials)
    }