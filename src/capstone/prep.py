"""Data loading and cleaning utilities for the spam classifier"""

import pandas as pd

def prep_data(
        df : pd.DataFrame, 
        name: str, 
        body_col: str, 
        subject_col : str = "Subject", 
        label_col : str = "Label"
) -> pd.DataFrame:   
    # Drop any rows with an empty body column 
    df = df[df[body_col].notna()]

    # Look for the Body: column to be prefixed with Subject:, if so extract
    is_subject_prefixed = df[body_col].str.match(r"Subject:", case=False)
    parts = df[body_col].str.partition("\n")
    parts.columns = ['header', 'separator', 'body']

    # If subject prefixed, grab the Subject from the header, otherwise ""
    subject = parts["header"].str[len("Subject:"):].str.strip().where(is_subject_prefixed, "")
    body = parts["body"].where(is_subject_prefixed, df[body_col])
     
    df_prepped = pd.DataFrame()    
    df_prepped[subject_col] = subject
    df_prepped[body_col] = body
    df_prepped[label_col] = df[label_col]
    df_prepped.insert(0, "Source", name)
    
    return df_prepped
