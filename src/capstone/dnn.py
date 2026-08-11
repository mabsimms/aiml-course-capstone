import pandas as pd
import numpy as np
import tensorflow as tf
import keras

from tokenizers import Tokenizer

from capstone.tokenization.tokenization import train_tokenizer, save_tokenizer, load_tokenizer, PAD_TOKEN, UNK_TOKEN

from pathlib import Path
import tempfile
import zipfile
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import logging
logger = logging.getLogger("capstone")

def _lowercase_only(input_text):
    return tf.strings.lower(input_text)

def build_dnn_model(    
    train_features: np.ndarray,
    vocab_size : int,    
    output_sequence_length : int = 4_500,
    embedding_dim : int = 64,
    lstm_units : int = 64,
    dense_units : int = 32,
    dropout_rate : float = 0.3,
    learning_rate : float = 1e-3,
    use_cudnn : str | bool = "auto"    
) -> keras.Model:
    logger.info(
        "Building DNN model: vocab_size=%s, output_sequence_length=%s, embedding_dim=%s, "
        "lstm_units=%s, dense_units=%s, dropout_rate=%s, learning_rate=%s, use_cudnn=%s",
        vocab_size, output_sequence_length, embedding_dim, lstm_units, dense_units, dropout_rate,
        learning_rate, use_cudnn
    )
     
    # Normalize features (per https://keras.io/api/layers/preprocessing_layers/numerical/normalization/)
    normalizer = keras.layers.Normalization()
    normalizer.adapt(train_features)

    # Set up input shapes (https://keras.io/api/layers/core_layers/input/)
    text_input = keras.Input(shape=(output_sequence_length,), dtype=tf.int32, name="text")
    numeric_input = keras.Input(shape=(train_features.shape[1],), name="engineered_features")

    # Text branch with embedding (https://keras.io/api/layers/core_layers/embedding/)
    # and memory layer (https://keras.io/api/layers/recurrent_layers/lstm/)    
    text_branch = keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
    ltsm_layer = keras.layers.LSTM(lstm_units, use_cudnn=use_cudnn)
    text_branch = ltsm_layer(text_branch)

    # adding logging on use of cuDNN per https://keras.io/api/layers/recurrent_layers/lstm/
    logger.info("LSTM cuDNN-eligibility params: dropout=%s, recurrent_dropout=%s, "
                "activation=%s, recurrent_activation=%s, unroll=%s, use_bias=%s, use_cudnn=%s",
                ltsm_layer.dropout,
                ltsm_layer.recurrent_dropout,
                ltsm_layer.activation.__name__,
                ltsm_layer.recurrent_activation.__name__,
                ltsm_layer.unroll,
                ltsm_layer.use_bias,
                use_cudnn
    )
    logger.info("LSTM.supports_masking = %s", ltsm_layer.supports_masking)
    
    # Numeric branch
    numeric_branch = normalizer(numeric_input)
    numeric_branch = keras.layers.Dense(16, activation="relu")(numeric_branch)

    # Fuse branches (via https://keras.io/api/layers/merging_layers/concatenate/)
    # with dropout (https://keras.io/api/layers/regularization_layers/dropout/)
    merged = keras.layers.Concatenate()([text_branch, numeric_branch])
    x = keras.layers.Dense(dense_units, activation="relu")(merged)
    x = keras.layers.Dropout(dropout_rate)(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    # Bind into the model
    model = keras.Model(inputs=[text_input, numeric_input], outputs=output)

    # Compile the model using the Adam optimizer
    model.compile(
        # Use the Adam optimizer (https://keras.io/api/optimizers/adam/) as a better fit 
        # for handling parameters with different gradient levels
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        # Match to sigmoid function (binary output - spam or not spam), penalizing "confident wrong" answers
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model


def _encode_text(tokenizer: Tokenizer, text: pd.Series) -> np.ndarray:
    encodings = tokenizer.encode_batch(text.tolist())
    return np.array([encoding.ids for encoding in encodings], dtype=np.int32)

def train_dnn(
        df_train : pd.DataFrame, 
        feature_cols : list[str],
        hyperparams : dict | None = None,
        epochs : int = 15,
        verbose : int = 1
) -> tuple[keras.Model, keras.callbacks.History, dict, Tokenizer]:
    if hyperparams is None:
        hyperparams = {}

    if verbose > 0:
        callback_verbose = 1
    else:
        callback_verbose = 0

    # Create a class weight to handle class imbalance in the targets (https://keras.io/examples/structured_data/imbalanced_classification/)
    # Penalize misclassification in the minority (spam) class, using the same weighting function as the 
    # classic model
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_train["Label"]),
        y=df_train["Label"]
    )
    class_weight_dict = dict(zip(np.unique(df_train["Label"]), class_weights))
    
    train_features = df_train[feature_cols].to_numpy(dtype=np.float64)

    max_tokens = hyperparams.get("max_tokens", 20_000)
    output_sequence_length = hyperparams.get("output_sequence_length", 4_500)
    min_frequency = hyperparams.get("min_frequency", 0)

    tokenizer = train_tokenizer(
        df_train["text"],
        vocab_size=max_tokens,
        output_sequence_length=output_sequence_length,
        min_frequency=min_frequency
    )
    assert tokenizer.token_to_id(PAD_TOKEN) == 0, "Expected PAD token id 0"    
    encoded_text = _encode_text(tokenizer, df_train["text"])

    # Do not pass max_tokens through
    model_hyperparameters = { 
        k: v for k, v in hyperparams.items() if k not in ("max_tokens", "min_frequency")
    }
    model = build_dnn_model(train_features, tokenizer.get_vocab_size(), 
                            **model_hyperparameters)

    early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                    verbose = callback_verbose
                )

    history = model.fit(
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

    return model, history, {
        'stopped_epoch': early_stopping.stopped_epoch,
        'stopped_best_weights' : early_stopping.best_weights
    }, tokenizer


def dnn_predict_proba(model, tokenizer, feature_cols, verbose : int = 1):
    def predict(df):
        encoded_text = _encode_text(tokenizer, df["text"])        
        features = df[feature_cols].to_numpy(dtype=np.float64)
        predictions = model.predict({
            "text": encoded_text,
            "engineered_features": features
        }, verbose=verbose)
        return predictions.ravel()
    return predict

def load_dnn_model(
        weights_path: Path,
        tokenizer_path: Path,
        feature_cols: list[str], 
        hyperparameters: dict
) -> tuple[keras.Model, Tokenizer]:
    tokenizer = load_tokenizer(str(tokenizer_path))
    dummy_features = np.zeros((1, len(feature_cols)), dtype=np.float64)

    model = build_dnn_model(dummy_features, tokenizer.get_vocab_size(), **hyperparameters)
    model.load_weights(weights_path)
     
    return model, tokenizer
