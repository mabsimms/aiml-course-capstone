import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import keras
import kagglehub
import zipfile

from capstone.dataset import prepare_experiment
from capstone.dnn import build_dnn_model, load_dnn_model, train_dnn
from capstone.predictor import load_predictor
from capstone.tokenization.tokenization import train_tokenizer, save_tokenizer, load_tokenizer, PAD_TOKEN, UNK_TOKEN

pytestmark = pytest.mark.train

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

    # Using our externalized tokenizer
    tokenizer = train_tokenizer(train_text, vocab_size=100, output_sequence_length=16)
    encodings = tokenizer.encode_batch(train_text.tolist())
    encoded_text = np.array([e.ids for e in encodings], dtype=np.int32)

    model = build_dnn_model(
        train_features,
        vocab_size=tokenizer.get_vocab_size(), 
        output_sequence_length=16,
        embedding_dim=8,
        lstm_units=4,
        dense_units=4
    )
    assert model.optimizer is not None

    predictions = model.predict(
        {
            "text": encoded_text,
            "engineered_features": train_features
        },
        verbose=0
    )

    assert predictions.shape == (4, 1)
    assert (predictions > 0).all() and (predictions <= 1).all()

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dnn_train_sample.csv"

def test_train_dnn():
    df_train = pd.read_csv(FIXTURE_PATH)
    model, _, _, tokenizer = train_dnn(df_train, feature_cols=["feature_a"], verbose=0)

    encodings = tokenizer.encode_batch(df_train["text"].tolist())
    encoded_text = np.array([e.ids for e in encodings], dtype=np.int32)

    predictions = model.predict(
        {
            "text": encoded_text,
            "engineered_features": df_train[["feature_a"]].to_numpy()
        },
        verbose=0
    )

    assert predictions.shape == (len(df_train), 1)
    assert (predictions > 0).all() and (predictions <= 1).all()
 
def test_dnn_save_load_roundtrip(tmp_path):
    #raw_dir = Path(kagglehub.dataset_download("nitishabharathi/email-spam-dataset"))
    #raw = { 
    #    csv_file.stem: pd.read_csv(csv_file) for csv_file in raw_dir.glob("*.csv")
    #}
    #df_train, _, feature_cols = prepare_experiment(raw, stratify_by_source=True)
    df_train = pd.read_csv(FIXTURE_PATH)

    hyperparameters = { 
        "max_tokens": 20_000,
        "output_sequence_length": 32,
        "embedding_dim": 8,
        "lstm_units": 4,
        "dense_units": 4
    }

    model, _, _, tokenizer = train_dnn(df_train, feature_cols=["feature_a"], verbose=0)

    weights_path = tmp_path / "roundtrip.weights.h5"
    tokenizer_path = tmp_path / "roundtrip.tokenizer.json"

    model.save_weights(weights_path)
    save_tokenizer(tokenizer, tokenizer_path)

    # TODO - need to include the hyperparameters to rebuild from saved model

# DNN_MANIFEST_PATH = Path("artifacts/dnn_baseline.manifest.json")

# def test_load_dnn_predictor():
#     if not DNN_MANIFEST_PATH.exists():
#         pytest.skip("No trained DNN artifact present")

#     predictor = load_predictor(DNN_MANIFEST_PATH)

#     assert predictor.model_type == "dnn"    
#     assert len(predictor.feature_cols) > 0
