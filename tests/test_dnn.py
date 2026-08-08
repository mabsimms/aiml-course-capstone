import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import keras
import kagglehub
import zipfile

from capstone.dataset import prepare_experiment
from capstone.dnn import build_dnn_model, train_dnn, load_dnn_model
from capstone.predictor import load_predictor

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
    model, _, _ = train_dnn(df_train, feature_cols=["feature_a"], verbose=0)

    predictions = model.predict(
        {
            "text": df_train["text"].to_numpy(),
            "engineered_features": df_train[["feature_a"]].to_numpy()
        },
        verbose=0
    )

    assert predictions.shape == (len(df_train), 1)
    assert (predictions > 0).all() and (predictions <= 1).all()

def get_vocabulary(model: keras.Model) -> list[str]:
    vectorize_layer = next(
        layer for layer in model.layers if isinstance(layer, keras.layers.TextVectorization)
    )
    return vectorize_layer.get_vocabulary()

def test_dnn_vocabulary_export(tmp_path):
    raw_dir = Path(kagglehub.dataset_download("nitishabharathi/email-spam-dataset"))
    raw = { 
        csv_file.stem: pd.read_csv(csv_file) for csv_file in raw_dir.glob("*.csv")
    }
    print("Starting full training run")
    df_train, _, feature_cols = prepare_experiment(raw, stratify_by_source=True)
    model = build_dnn_model(
        df_train["text"],
        df_train[feature_cols].to_numpy(dtype=np.float64),        
    )
    print("Full training complete")

    live_vocabulary = get_vocabulary(model)

    save_path = tmp_path / "vocab_check.keras"
    model.save(save_path)

    with zipfile.ZipFile(save_path) as archive:
        exported_vocabulary = archive.read(
            "assets/layers/text_vectorization/vocabulary.txt"
        ).decode("utf-8").splitlines()

    assert exported_vocabulary == live_vocabulary
    
def test_dnn_save_load_roundtrip(tmp_path):
    raw_dir = Path(kagglehub.dataset_download("nitishabharathi/email-spam-dataset"))
    raw = { 
        csv_file.stem: pd.read_csv(csv_file) for csv_file in raw_dir.glob("*.csv")
    }
    df_train, _, feature_cols = prepare_experiment(raw, stratify_by_source=True)

    hyperparameters = { 
        "embedding_dim": 8,
        "lstm_units": 4,
        "dense_units": 4
    }

    model = build_dnn_model(
        df_train["text"],
        df_train[feature_cols].to_numpy(dtype=np.float64),
        **hyperparameters
    )

    save_path = tmp_path / "roundtrip.keras"
    model.save(save_path)

    reloaded = keras.models.load_model(save_path)
    
    predictions = reloaded.predict(
        {
            "text": df_train["text"].to_numpy(dtype=object), 
            "engineered_features": df_train[feature_cols].to_numpy(dtype=np.float64)
        },
        verbose=0
    )
    assert predictions.shape == (len(df_train), 1)


# DNN_MANIFEST_PATH = Path("artifacts/dnn_baseline.manifest.json")

# def test_load_dnn_predictor():
#     if not DNN_MANIFEST_PATH.exists():
#         pytest.skip("No trained DNN artifact present")

#     predictor = load_predictor(DNN_MANIFEST_PATH)

#     assert predictor.model_type == "dnn"    
#     assert len(predictor.feature_cols) > 0