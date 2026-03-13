"""Unit tests for the database layer."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.pipeline.load import create_schema, load_dataframes, get_connection as load_get_conn
from src.database.db import (
    get_students,
    get_courses,
    get_results,
    get_results_enriched,
    validate_foreign_keys,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_data() -> dict[str, pd.DataFrame]:
    students = pd.DataFrame(
        {
            "student_id": [1, 2],
            "naam": ["Alice", "Bob"],
            "leeftijd": [20, 21],
            "studierichting": ["Informatica", "Bouw"],
        }
    )
    courses = pd.DataFrame(
        {
            "course_id": [1, 2],
            "naam": ["Wiskunde", "Programmeren"],
            "credits": [6, 6],
            "semester": [1, 2],
        }
    )
    results = pd.DataFrame(
        {
            "result_id": [1, 2, 3, 4],
            "student_id": [1, 1, 2, 2],
            "course_id": [1, 2, 1, 2],
            "score": [15.0, 9.0, 12.0, 7.0],
            "slaagde": [1, 0, 1, 0],
        }
    )
    return {"students": students, "courses": courses, "results": results}


@pytest.fixture
def in_memory_db(clean_data) -> sqlite3.Connection:
    """Return an in-memory SQLite connection with schema + data loaded."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    create_schema(conn)
    load_dataframes(clean_data, conn)
    return conn


# ---------------------------------------------------------------------------
# Schema & loading
# ---------------------------------------------------------------------------

class TestDatabaseLoad:
    def test_tables_created(self, in_memory_db):
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", in_memory_db
        )["name"].tolist()
        assert "students" in tables
        assert "courses" in tables
        assert "results" in tables

    def test_student_count(self, in_memory_db):
        df = get_students(in_memory_db)
        assert len(df) == 2

    def test_course_count(self, in_memory_db):
        df = get_courses(in_memory_db)
        assert len(df) == 2

    def test_result_count(self, in_memory_db):
        df = get_results(in_memory_db)
        assert len(df) == 4


# ---------------------------------------------------------------------------
# Enriched results
# ---------------------------------------------------------------------------

class TestEnrichedResults:
    def test_enriched_has_expected_columns(self, in_memory_db):
        df = get_results_enriched(in_memory_db)
        expected_cols = {
            "result_id",
            "student_id",
            "student_naam",
            "leeftijd",
            "studierichting",
            "course_id",
            "course_naam",
            "credits",
            "semester",
            "score",
            "slaagde",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_enriched_row_count(self, in_memory_db):
        df = get_results_enriched(in_memory_db)
        assert len(df) == 4


# ---------------------------------------------------------------------------
# Foreign-key validation
# ---------------------------------------------------------------------------

class TestForeignKeyCheck:
    def test_valid_data_passes(self, in_memory_db):
        assert validate_foreign_keys(in_memory_db) is True
