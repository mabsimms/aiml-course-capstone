import pandas as pd

from capstone.prep import prep_data
from capstone.features import engineer_features

def build_dataset(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prepped = [ 
        engineer_features(prep_data(df, name, "Body", "Subject", "Label"))
        for name, df in raw.items()
    ]
    return pd.concat(prepped, ignore_index=True)

def dedup_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["Subject"] + " " + df["Body"]
    df = df.drop_duplicates(subset=["text"], keep="first")
    return df.reset_index(drop=True)

def select_feature_columns(df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    return [col for col in candidate_cols if df[col].nunique() > 1]