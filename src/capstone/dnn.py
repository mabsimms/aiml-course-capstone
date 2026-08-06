import pandas as pd
import numpy as np
import tensorflow as tf
import keras

from sklearn.utils.class_weight import compute_class_weight

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
    use_cudnn : bool = False    
) -> keras.Model:
    # Vectorize text (per https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/)
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        # Set output_sequence_length to avoid padding issues with cuDNN on unbounded input
        output_sequence_length=output_sequence_length
    )
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
    text_branch = keras.layers.LSTM(lstm_units, use_cudnn=use_cudnn)(text_branch)

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

def train_dnn(df_train, feature_cols, verbose=1):
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

    model = build_dnn_model(train_text, train_features)
    model.fit(
        {
            "text": train_text,
            "engineered_features": train_features
        },
        df_train["Label"],
        validation_split=0.1,
        epochs=15,
        class_weight=class_weight_dict,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )
        ],
        verbose=verbose
    )

    return model

