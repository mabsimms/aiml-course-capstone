import numpy as np
import pandas as pd

from typing import Callable

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

def evaluate_model(
        predict_proba_fn: Callable[[pd.DataFrame], np.ndarray],
        df: pd.DataFrame,
        label_col: str,
        threshold : float = 0.5
) -> dict:
    probability = predict_proba_fn(df)
    predictions = (probability >= threshold).astype(int)

    return {
        "precision": precision_score(df[label_col], predictions),
        "recall": recall_score(df[label_col], predictions),
        "f1": f1_score(df[label_col], predictions),
        "roc_auc": roc_auc_score(df[label_col], probability),
        "confusion_matrix": confusion_matrix(df[label_col], predictions),
        "classification_report": classification_report(df[label_col], predictions, output_dict=True),
    }
