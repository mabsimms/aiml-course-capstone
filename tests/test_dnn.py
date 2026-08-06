import numpy as np
import pandas as pd
from pathlib import Path

from capstone.dnn import build_dnn_model, train_dnn

def test_build_dnn_model():
    train_text = pd.Series([
        "free moneys the click now!!",
        "meeting notes attached",
        "win the greatest prize today",
        "quarterly report review"
    ])
    train_features = np.array([
        [10.0, 2.0],
        [1.0, 2.0],
        [12.0, 3.0],
        [2.0, 0.0]
    ])

    model = build_dnn_model(
        train_text, 
        train_features, 
        max_tokens=100, 
        embedding_dim=8,
        lstm_units=4,
        dense_units=4
    )

    assert model.optimizer is not None

    predictions = model.predict(
        {"text": train_text.to_numpy(dtype=object), "engineered_features": train_features},
        verbose=0
    )

    assert predictions.shape == (4, 1)
    assert (predictions > 0).all() and (predictions <= 1).all()

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dnn_train_sample.csv"

def test_train_dnn():
    df_train = pd.read_csv(FIXTURE_PATH)
    model = train_dnn(df_train, feature_cols=["feature_a"])

    predictions = model.predict(
        {
            "text": df_train["text"].to_numpy(),
            "engineered_features": df_train[["feature_a"]].to_numpy()
        },
        verbose=0
    )

    assert predictions.shape == (len(df_train), 1)
    assert (predictions > 0).all() and (predictions <= 1).all()