import pandas as pd
import pytest
from capstone.features import engineer_features, find_outliers

def test_engineer_features():
    df = pd.DataFrame({
        "Subject": ["AB!!"],
        "Body": ["hi http://www.x.com www.x.com"]
    })

    result = engineer_features(df)

    assert result.loc[0, "subject_length"] == 4
    assert result.loc[0, "subject_word_count"] == 1
    assert result.loc[0, "subject_upper_ratio"] == 1.0
    assert result.loc[0, "subject_digit_ratio"] == 0.0
    assert result.loc[0, "subject_exclaim_count"] == 2
    assert result.loc[0, "subject_question_count"] == 0
    assert result.loc[0, "has_subject"] == True
    
def test_outliers():
    df = pd.DataFrame({
        "normal_feature": [1, 2, 2, 3, 2, 1, 100],
        "is_flag": [True, False, True, False, True, False, True],
        "Label": [0, 0, 0, 0, 0, 0, 1],
    })

    result = find_outliers(df, feature_cols=["normal_feature", "is_flag"], label_col="Label")

    # Boolean column excluded by select_dtypes — only "normal_feature" should be analyzed
    assert result["feature"].tolist() == ["normal_feature"]

    row = result.iloc[0]
    assert row["lower_bound"] == 0.0
    assert row["upper_bound"] == 4.0
    assert row["outlier_count"] == 1
    assert row["outlier_rate"] == pytest.approx(1 / 7)
    assert row["label_mix"] == {1: 1.0}