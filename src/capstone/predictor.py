import json
import joblib
import keras

import pandas as pd
import numpy as np

from typing import Callable
from pathlib import Path
from dataclasses import dataclass

from capstone.classic import classical_predict_proba
from capstone.dnn import dnn_predict_proba, load_dnn_model
from capstone.manifest import SCHEMA_VERSION

@dataclass
class Predictor:
    model_type: str
    feature_cols: list[str]
    predict: Callable[[pd.DataFrame], np.ndarray]

def load_predictor(manifest_path : Path) -> Predictor:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version {schema_version} in {manifest_path}"
            f"(expected {SCHEMA_VERSION})"
        )

    model_type = manifest["model_type"]
    feature_cols = manifest["feature_cols"]
    artifact_path = manifest_path.parent / manifest["artifact_path"]

    if model_type == "classic":
        pipeline = joblib.load(artifact_path)
        predict = classical_predict_proba(pipeline)
    elif model_type == "dnn":        
        weights_path = artifact_path.with_suffix(".weights.h5")
        tokenizer_path = artifact_path.with_suffix(".tokenizer.json")
        model, tokenizer = load_dnn_model(
            weights_path, tokenizer_path, feature_cols, manifest["hyperparameters"]
        )        
        predict = dnn_predict_proba(model, tokenizer, feature_cols, verbose=0)
    else:
        raise ValueError(f"Unknown model type {model_type} in {manifest_path}")

    return Predictor(model_type=model_type, feature_cols=feature_cols, predict=predict)