"""Machine-learning model: predict whether a student will pass a course.

The target variable is ``slaagde`` (1 = pass, 0 = fail).
Features: leeftijd, credits, semester, studierichting (one-hot encoded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.database.db import get_results_enriched


NUMERIC_FEATURES = ["leeftijd", "credits", "semester"]
CATEGORICAL_FEATURES = ["studierichting"]
TARGET = "slaagde"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_pipeline() -> Pipeline:
    """Return an sklearn Pipeline with preprocessing and a Random Forest classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix *X* and target vector *y* from enriched results.

    Parameters
    ----------
    df:
        Output of :func:`src.database.db.get_results_enriched`.

    Returns
    -------
    X, y
    """
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features].copy()
    y = df[TARGET].astype(int)
    return X, y


def train(conn) -> tuple[Pipeline, dict]:
    """Train and evaluate the classification model.

    Parameters
    ----------
    conn:
        Active SQLite connection.

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    metrics  : dict with accuracy and classification_report string
    """
    df = get_results_enriched(conn)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Faalt", "Slaagt"])

    metrics = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipeline, metrics


def predict(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return predicted labels (0/1) for feature matrix *X*.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline returned by :func:`train`.
    X:
        DataFrame with the same columns as the training features.
    """
    return pipeline.predict(X)


def predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return class probabilities for feature matrix *X*.

    Returns
    -------
    Array of shape (n_samples, 2) where column 1 is P(slaagde=1).
    """
    return pipeline.predict_proba(X)
