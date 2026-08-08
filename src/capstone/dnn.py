import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import keras_tuner as kt
from pathlib import Path
import tempfile
import zipfile

from sklearn.utils.class_weight import compute_class_weight

import logging
logger = logging.getLogger("capstone")

def build_dnn_model(
    train_text: pd.Series,
    train_features: np.ndarray,
    max_tokens: int = 20_000,
    output_sequence_length : int = 3000,
    embedding_dim : int = 64,
    lstm_units : int = 64,
    dense_units : int = 32,
    dropout_rate : float = 0.3,
    learning_rate : float = 1e-3,
    use_cudnn : str | bool = "auto",
    vocabulary : list[str] | None = None
) -> keras.Model:
    logger.info(
        "Building DNN model: max_tokens=%s, output_sequence_length=%s, embedding_dim=%s, "
        "lstm_units=%s, dense_units=%s, dropout_rate=%s, learning_rate=%s, use_cudnn=%s",
        max_tokens, output_sequence_length, embedding_dim, lstm_units, dense_units, dropout_rate,
        learning_rate, use_cudnn
    )
    
    # Vectorize text (per https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/)
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        # Set output_sequence_length to avoid padding issues with cuDNN on unbounded input
        output_sequence_length=output_sequence_length
    )
    logger.info("TextVectorization output_sequence_length=%s",
        vectorize_layer.get_config().get("output_sequence_length"))

    if vocabulary is not None:
        # Reconstructing a previously trained model from a saved vocabulary (working around a Keras 
        # bug where a saved vectorization set fails to restore via load_model())
        print(len(vocabulary), list(vocabulary).count(""))
        indices = [i for i, term in enumerate(vocabulary) if term == ""]
        print(indices)
        
        vectorize_layer.set_vocabulary(vocabulary)
    else:
        vectorize_layer.adapt(train_text)

    # Normalize features (per https://keras.io/api/layers/preprocessing_layers/numerical/normalization/)
    normalizer = keras.layers.Normalization()
    normalizer.adapt(train_features)

    # Set up input shapes (https://keras.io/api/layers/core_layers/input/)
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")
    numeric_input = keras.Input(shape=(train_features.shape[1],), name="engineered_features")

    # Text branch with embedding (https://keras.io/api/layers/core_layers/embedding/)
    # and memory layer (https://keras.io/api/layers/recurrent_layers/lstm/)
    text_branch = vectorize_layer(text_input)
    text_branch = keras.layers.Embedding(max_tokens, embedding_dim, mask_zero=True)(text_branch)
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

def train_dnn(
        df_train : pd.DataFrame, 
        feature_cols : list[str],
        hyperparams : dict | None = None,
        epochs : int = 15,
        verbose : int = 1
) -> tuple[keras.Model, keras.callbacks.History, dict]:
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

    train_text = df_train["text"].to_numpy(dtype=object)
    train_features = df_train[feature_cols].to_numpy(dtype=np.float64)

    model = build_dnn_model(train_text, train_features, **hyperparams)

    early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                    verbose = callback_verbose
                )

    history = model.fit(
        {
            "text": train_text,
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
    }

def train_dnn_cached(df_train : pd.DataFrame, feature_cols : list[str], cache_path : Path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached model from {cache_path}")
        return keras.models.load_model(cache_path)

    model = train_dnn(df_train, feature_cols)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(cache_path)
    print(f"Saved trained model tos {cache_path}")

    return model

def dnn_predict_proba(model, feature_cols, verbose : int = 1):
    def predict(df):
        text = df["text"].to_numpy(dtype=object)
        features = df[feature_cols].to_numpy(dtype=np.float64)
        predictions = model.predict({
            "text": text,
            "engineered_features": features
        }, verbose=verbose)
        return predictions.ravel()
    return predict

def load_dnn_model(artifact_path: Path, feature_cols: list[str], hyperparameters: dict) -> keras.Model:
    # Keras round-tripping is fragile; this is the workaround
    with zipfile.ZipFile(artifact_path) as archive:
        vocabulary = archive.read("assets/layers/text_vectorization/vocabulary.txt").decode("utf-8").splitlines()

        with tempfile.NamedTemporaryFile(suffix=".weights.h5") as weights_file:
            weights_file.write(archive.read("model.weights.h5"))
            weights_file.flush()

            dummy_text = pd.Series([""])
            dummy_features = np.zeros((1, len(feature_cols)), dtype=np.float64)

            model = build_dnn_model(
                dummy_text, dummy_features, vocabulary=vocabulary, **hyperparameters
            )
            model.load_weights(weights_file.name)

    return model