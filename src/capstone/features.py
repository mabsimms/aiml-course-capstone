""" Extract features outside of the structure of which words appear """

import pandas as pd
import re

# Dictionary of special characters; we'll prune out after extraction (let evidence determine
# the meaningful features)
PUNCTUATION_NAMES = {
    "!": "exclaim", 
    '"': "dquote", 
    "#": "hash", 
    "$": "dollar", 
    "%": "percent",
    "&": "amp", 
    "'": "squote", 
    "(": "lparen", 
    ")": "rparen", 
    "*": "asterisk",
    "+": "plus", 
    ",": "comma", 
    "-": "hyphen", 
    ".": "period", 
    "/": "slash",
    ":": "colon", 
    ";": "semicolon", 
    "<": "lt", 
    "=": "eq", 
    ">": "gt",
    "?": "question", 
    "@": "at", 
    "[": "lbracket", 
    "\\": "backslash", 
    "]": "rbracket",
    "^": "caret", 
    "_": "underscore", 
    "`": "backtick", 
    "{": "lbrace", 
    "|": "pipe",
    "}": "rbrace", 
    "~": "tilde",
}

def engineer_features(
        df : pd.DataFrame, 
        subject_col : str = "Subject", 
        body_col : str = "Body"
) -> pd.DataFrame:
    df = df.copy()

    for col, prefix in [(subject_col, "subject"), (body_col, "body")]:
        text = df[col].str

        # Text Length
        length = text.len()
        word_count = text.split().str.len()

        # Letters and captialization ratios
        alpha_count = text.count(r"[A-Za-z]")
        upper_count = text.count(r"[A-Z]")
        upper_ratio = (upper_count / alpha_count).fillna(0)

        # Digit Ratios
        digit_count = text.count(r"\d")
        digit_ratio = (digit_count / length).fillna(0)
       
        # Flag for embedded urls
        has_http_url = text.contains(r"https?://", regex=True, case=False, na=False)
        has_web_url = text.contains(r"\bwww\.[a-zA-Z0-9]+\.[a-zA-Z]{2,}", regex=True, case=False, na=False)

        df[f"{prefix}_length"]        = length
        df[f"{prefix}_word_count"]    = word_count
        df[f"{prefix}_upper_ratio"]   = upper_ratio
        df[f"{prefix}_digit_ratio"]   = digit_ratio
 
        df[f"{prefix}_has_http_url"]  = has_http_url
        df[f"{prefix}_has_web_url"]   = has_web_url

        # Punctuation counts
        for char, name in PUNCTUATION_NAMES.items():
            df[f"{prefix}_{name}_count"] = text.count(re.escape(char))

    df["has_subject"] = df[subject_col].str.len() > 0


    return df

# Look for outliers in numeric feature columns
def find_outliers(
        df: pd.DataFrame, 
        feature_cols: list[str], 
        label_col : str
) -> pd.DataFrame:
    feature_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    rows = [ ] 
    
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)

        rows.append({
            "feature": col,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": int(is_outlier.sum()),
            "outlier_rate": is_outlier.mean(),
            "label_mix": df.loc[is_outlier, label_col].value_counts(normalize=True).to_dict()
        })

    return pd.DataFrame(rows)