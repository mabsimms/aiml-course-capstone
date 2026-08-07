from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import joblib
from typing import Callable
from pathlib import Path

def classical_predict_proba(pipeline: Pipeline) -> Callable[[pd.DataFrame], np.ndarray]:
    def predict(df: pd.DataFrame) -> np.ndarray:
          return pipeline.predict_proba(df)[:,1]
    return predict

def build_classical_pipeline(feature_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("tfidf", TfidfVectorizer(stop_words="english"), "text"),
        ("engineered", StandardScaler(), feature_cols)
    ])

    return Pipeline([
        ("features", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

def train_classical_model(
            df_train: pd.DataFrame,
            feature_cols: list[str],
            label_col: str = "Label",
            param_grid : dict | None = None,
            n_jobs : int = -1,
            cv : int = 5,
            verbose : int = 0
) -> GridSearchCV:
        if param_grid is None:            
            param_grid = { 
                # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
                # Inverse regularization, lower values apply stronger regularization
                # (simpler boundary, less overfitting risk).  Span a broad enough range
                # to find the inflection point
                "clf__C": [0.01, 0.1, 1, 10, 100, 1000],

                # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
                # Cap the TF-IDF vocabulary to the N most frequent terms.  Smaller vocab
                # reduces dimensionality (and overfitting risk), but risks droopping 
                # rare but meaningful spam specific terms
                "features__tfidf__max_features": [5000, 20000, None],

                # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
                # Unigrams (single words) and bigrams (adjacent word pairs) to capture
                # short phrases (e.g. "click here")
                "features__tfidf__ngram_range": [(1, 1), (1, 2)]
            }

        pipeline = build_classical_pipeline(feature_cols)

        # Sweep across hyperparameters
        grid = GridSearchCV(
            pipeline, 
            param_grid,
            # Optimize for precision/recall tradeoff
            scoring = "f1",
            # N-fold cross-validation within the training data (default to 5)
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose
        )        
        grid.fit(df_train, df_train[label_col])
        return grid
    
def train_classical_model_cached(cache_path : Path, 
        df_train: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "Label",
        param_grid : dict | None = None,
        n_jobs : int = -1,
        cv : int = 5,
        verbose : int = 0
) -> GridSearchCV:
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached model from {cache_path}")
        return joblib.load(cache_path)

    grid = train_classical_model(
        df_train = df_train, 
        feature_cols = feature_cols, 
        label_col = label_col, 
        param_grid = param_grid, 
        n_jobs = n_jobs, 
        cv = cv, 
        verbose = verbose
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid, cache_path)
    print(f"Saved trained model to {cache_path}")
    return grid