import numpy as np
import pandas as pd

from capstone.evaluate import evaluate_model

def test_evaluate_model():
    df = pd.DataFrame({"Label": [0, 1]})
    probabilities = np.array([0.3, 0.6])

    def stub_predict_proba(df):
        return probabilities

    low_threshold_result = evaluate_model(stub_predict_proba, df, "Label", threshold=0.5)
    assert low_threshold_result["recall"] == 1.0
    assert low_threshold_result["precision"] == 1.0

    