import pandas as pd

from capstone.dataset import build_dataset, dedup_dataset, select_feature_columns

def test_build_dataset_combines_sources():
    raw = { 
        "SourceA": pd.DataFrame({
            "Body": ["Subject: Hi\nsome body text one"],
            "Label": [0]
        }),
        "SourceB": pd.DataFrame({
            "Body": ["Subject: Hi\nsome body text two"],
            "Label": [1]
        })
    }

    result = build_dataset(raw)
    assert len(result) == 2
    assert result.index.to_list() == [0, 1]
    assert "subject_length" in result.columns


def test_dedup_dataset():
    df = pd.DataFrame({
        "Subject": ["Hi", "Hi", "Heya"],
        "Body": ["there", "there", "not there"],
        "Label": [0, 0, 1]
    })

    result = dedup_dataset(df)

    assert len(result) == 2
    assert result["text"].to_list() == ["Hi there", "Heya not there"]
    assert result.index.tolist() == [0, 1]

def test_feature_columns_drops_constants():
    df = pd.DataFrame({
        "varying": [1, 2, 3],
        "constant": [8, 8, 8]
    })

    result = select_feature_columns(df, candidate_cols=["varying", "constant"])
    assert result == ["varying"]