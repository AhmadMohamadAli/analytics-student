"""Database access helpers for the analytics application."""

from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd


def get_connection(db_path: str = "data/analytics.db") -> sqlite3.Connection:
    """Return a SQLite connection with foreign-key checks enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def query(sql: str, conn: sqlite3.Connection, params: Optional[tuple] = None) -> pd.DataFrame:
    """Execute *sql* and return the result as a DataFrame.

    Parameters
    ----------
    sql:
        SQL query string.
    conn:
        Active database connection.
    params:
        Optional tuple of query parameters.
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_students(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all rows from the students table."""
    return query("SELECT * FROM students ORDER BY student_id", conn)


def get_courses(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all rows from the courses table."""
    return query("SELECT * FROM courses ORDER BY course_id", conn)


def get_results(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all rows from the results table."""
    return query("SELECT * FROM results ORDER BY result_id", conn)


def get_results_enriched(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return results joined with student and course information."""
    sql = """
        SELECT
            r.result_id,
            s.student_id,
            s.naam        AS student_naam,
            s.leeftijd,
            s.studierichting,
            c.course_id,
            c.naam        AS course_naam,
            c.credits,
            c.semester,
            r.score,
            r.slaagde
        FROM results r
        JOIN students s ON s.student_id = r.student_id
        JOIN courses  c ON c.course_id  = r.course_id
        ORDER BY r.result_id
    """
    return query(sql, conn)


def validate_foreign_keys(conn: sqlite3.Connection) -> bool:
    """Return True if the PRAGMA foreign_key_check passes (no violations)."""
    result = pd.read_sql_query("PRAGMA foreign_key_check;", conn)
    return result.empty
