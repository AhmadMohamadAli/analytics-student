"""Ingest raw CSV files from a directory into Pandas DataFrames."""

from __future__ import annotations

import os

import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a single CSV file and return a DataFrame.

    Parameters
    ----------
    filepath:
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    return pd.read_csv(filepath)


def load_raw_data(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Load all raw CSV files (students, courses, results) from *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory that contains students.csv, courses.csv, results.csv.

    Returns
    -------
    dict with keys 'students', 'courses', 'results'.
    """
    expected = ["students", "courses", "results"]
    frames: dict[str, pd.DataFrame] = {}
    for name in expected:
        path = os.path.join(data_dir, f"{name}.csv")
        frames[name] = load_csv(path)
    return frames
