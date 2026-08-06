import pandas as pd

from capstone.prep import prep_data

def test_prep_data_splits_subject_prefixed_body():
    df = pd.DataFrame({
        "Body": [
            "Subject: All teh free monies!\nClick here now",
            "No subject just body text, get teh free monies",
            None
        ],
        "Label": [1, 0, 1]
    })

    result = prep_data(df, name="TestSource", body_col="Body", subject_col="Subject", label_col="Label")

    assert len(result) == 2

    prefixed_row = result.iloc[0]
    assert prefixed_row["Subject"] == "All teh free monies!"
    assert prefixed_row["Body"] == "Click here now"
    assert prefixed_row["Label"] == 1
    assert prefixed_row["Source"] == "TestSource"

