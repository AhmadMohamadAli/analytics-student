"""Transform raw DataFrames: clean, impute, validate."""

from __future__ import annotations

import pandas as pd
import numpy as np


def clean_students(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the students DataFrame.

    - Drops rows with a missing student_id or naam.
    - Fills missing leeftijd with the median.
    - Fills missing studierichting with 'Onbekend'.
    """
    df = df.copy()
    df.dropna(subset=["student_id", "naam"], inplace=True)
    if df["leeftijd"].isnull().any():
        df["leeftijd"] = df["leeftijd"].fillna(df["leeftijd"].median())
    df["leeftijd"] = df["leeftijd"].astype(int)
    df["studierichting"] = df["studierichting"].fillna("Onbekend")
    return df.reset_index(drop=True)


def clean_courses(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the courses DataFrame.

    - Drops rows with a missing course_id or naam.
    - Ensures credits and semester are positive integers.
    """
    df = df.copy()
    df.dropna(subset=["course_id", "naam"], inplace=True)
    df["credits"] = pd.to_numeric(df["credits"], errors="coerce").fillna(0).astype(int)
    df["semester"] = pd.to_numeric(df["semester"], errors="coerce").fillna(1).astype(int)
    return df.reset_index(drop=True)


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the results DataFrame.

    - Drops rows where student_id or course_id is missing.
    - Imputes missing scores with the column mean.
    - Recalculates 'slaagde' (1 if score >= 10, else 0).
    - Clips scores to the valid range [0, 20].
    """
    df = df.copy()
    df.dropna(subset=["student_id", "course_id"], inplace=True)
    mean_score = df["score"].mean(skipna=True)
    df["score"] = df["score"].fillna(round(mean_score, 2))
    df["score"] = df["score"].clip(0, 20)
    df["slaagde"] = (df["score"] >= 10).astype(int)
    return df.reset_index(drop=True)


def transform(
    raw: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply all cleaning steps and return a dict of clean DataFrames.

    Parameters
    ----------
    raw:
        Output of :func:`src.pipeline.ingest.load_raw_data`.

    Returns
    -------
    dict with keys 'students', 'courses', 'results'.
    """
    return {
        "students": clean_students(raw["students"]),
        "courses": clean_courses(raw["courses"]),
        "results": clean_results(raw["results"]),
    }
