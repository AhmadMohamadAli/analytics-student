"""
pipeline.py
-----------
Loads and cleans the raw student dataset.
Run this script first before anything else.

Usage:
    python src/pipeline.py
"""

import os
import pandas as pd


RAW_PATH = "data/raw/students.csv"
CLEAN_PATH = "data/clean/students_clean.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV file. Tries semicolon separator first, then comma."""
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] < 5:
            df = pd.read_csv(path, sep=",")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download it from Kaggle:\n"
            "https://www.kaggle.com/datasets/uciml/student-alcohol-consumption\n"
            "and place the file as 'data/raw/students.csv'"
        )
    print(f"[load]  Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, impute missing values, engineer features."""

    original_len = len(df)

    # 1 — Drop exact duplicates
    df = df.drop_duplicates()
    print(f"[clean] Dropped {original_len - len(df)} duplicate rows.")

    # 2 — Impute numeric columns with median
    numeric_cols = df.select_dtypes(include="number").columns
    missing_before = df[numeric_cols].isnull().sum().sum()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print(f"[clean] Imputed {missing_before} missing numeric values with median.")

    # 3 — Impute categorical columns with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 4 — Feature engineering
    if "G3" in df.columns:
        df["passed"] = (df["G3"] >= 10).astype(int)
        print(f"[clean] Pass rate: {df['passed'].mean():.1%}")

    if "G1" in df.columns and "G2" in df.columns and "G3" in df.columns:
        df["avg_grade"] = df[["G1", "G2", "G3"]].mean(axis=1).round(2)

    print(f"[clean] Final shape: {df.shape}")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save cleaned dataframe to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save]  Saved clean data to '{path}'.")


def run_pipeline() -> pd.DataFrame:
    print("=" * 50)
    print("  Student Analytics — Data Pipeline")
    print("=" * 50)
    df_raw = load_data(RAW_PATH)
    df_clean = clean_data(df_raw)
    save_data(df_clean, CLEAN_PATH)
    print("=" * 50)
    print("  Pipeline completed successfully!")
    print("=" * 50)
    return df_clean


if __name__ == "__main__":
    run_pipeline()
