"""Load clean DataFrames into a SQLite database."""

from __future__ import annotations

import sqlite3

import pandas as pd


def get_connection(db_path: str = "data/analytics.db") -> sqlite3.Connection:
    """Return a SQLite connection with foreign-key enforcement enabled.

    Parameters
    ----------
    db_path:
        Path to the SQLite file (created if it does not exist).
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the database tables if they do not already exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            student_id  INTEGER PRIMARY KEY,
            naam        TEXT    NOT NULL,
            leeftijd    INTEGER,
            studierichting TEXT
        );

        CREATE TABLE IF NOT EXISTS courses (
            course_id   INTEGER PRIMARY KEY,
            naam        TEXT    NOT NULL,
            credits     INTEGER,
            semester    INTEGER
        );

        CREATE TABLE IF NOT EXISTS results (
            result_id   INTEGER PRIMARY KEY,
            student_id  INTEGER NOT NULL REFERENCES students(student_id),
            course_id   INTEGER NOT NULL REFERENCES courses(course_id),
            score       REAL,
            slaagde     INTEGER
        );
        """
    )
    conn.commit()


def load_dataframes(
    clean: dict[str, pd.DataFrame],
    conn: sqlite3.Connection,
) -> None:
    """Insert clean DataFrames into the database, replacing existing data.

    Parameters
    ----------
    clean:
        Output of :func:`src.pipeline.transform.transform`.
    conn:
        Active SQLite connection.
    """
    # Load in dependency order (students and courses before results)
    for table in ("students", "courses", "results"):
        clean[table].to_sql(table, conn, if_exists="replace", index=False)
    conn.commit()


def run_pipeline(
    data_dir: str = "data/raw",
    db_path: str = "data/analytics.db",
) -> None:
    """Run the full ETL pipeline: ingest → transform → load.

    Parameters
    ----------
    data_dir:
        Directory containing the raw CSV files.
    db_path:
        Target SQLite database path.
    """
    from src.pipeline.ingest import load_raw_data
    from src.pipeline.transform import transform

    raw = load_raw_data(data_dir)
    clean = transform(raw)
    conn = get_connection(db_path)
    try:
        create_schema(conn)
        load_dataframes(clean, conn)
        print(f"Pipeline complete. Database written to '{db_path}'.")
    finally:
        conn.close()


if __name__ == "__main__":
    run_pipeline()
