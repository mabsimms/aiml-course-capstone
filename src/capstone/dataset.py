import pandas as pd

from capstone.prep import prep_data
from capstone.features import engineer_features
from sklearn.model_selection import train_test_split

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

def filter_low_quality_text(df, column_name, max_tokens=100_000):
    is_empty = df[column_name].str.strip() == ""
    token_counts = df[column_name].str.split().str.len()
    is_corrupted = token_counts > max_tokens

    return df[~is_empty & ~is_corrupted]

def prepare_experiment(
    raw: dict[str, pd.DataFrame], 
    stratify_by_source : bool, 
    test_size : float = 0.2, 
    random_state : int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = build_dataset(raw)
    df = dedup_dataset(df)
    df = filter_low_quality_text(df, "text")
    
    if stratify_by_source:
        strat_key = df["Label"].astype(str) + "_" + df["Source"]
    else:
        strat_key = df["Label"]
    
    df_train, df_test = train_test_split(df, test_size=test_size, stratify=strat_key, random_state=random_state)

    exclude_cols = ["Source", "Subject", "Body", "Label", "text"]
    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = select_feature_columns(df_train, candidate_cols)
    
    return df_train, df_test, feature_cols

