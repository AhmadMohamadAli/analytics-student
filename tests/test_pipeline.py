"""Unit tests for the data pipeline (ingest + transform)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.pipeline.transform import clean_students, clean_courses, clean_results, transform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_students() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": [1, 2, 3, None],
            "naam": ["Alice", "Bob", "Charlie", None],
            "leeftijd": [20, None, 22, 19],
            "studierichting": ["Informatica", None, "Bouw", "Chemie"],
        }
    )


@pytest.fixture
def raw_courses() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "course_id": [1, 2],
            "naam": ["Wiskunde", "Programmeren"],
            "credits": [6, None],
            "semester": [1, 2],
        }
    )


@pytest.fixture
def raw_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "result_id": [1, 2, 3, 4],
            "student_id": [1, 1, 2, None],
            "course_id": [1, 2, 1, 2],
            "score": [15.0, None, 8.0, 12.0],
            "slaagde": [1, None, 0, 1],
        }
    )


# ---------------------------------------------------------------------------
# clean_students
# ---------------------------------------------------------------------------

class TestCleanStudents:
    def test_drops_rows_with_null_student_id(self, raw_students):
        clean = clean_students(raw_students)
        assert clean["student_id"].isnull().sum() == 0

    def test_drops_rows_with_null_naam(self, raw_students):
        clean = clean_students(raw_students)
        assert clean["naam"].isnull().sum() == 0

    def test_imputes_leeftijd(self, raw_students):
        clean = clean_students(raw_students)
        assert clean["leeftijd"].isnull().sum() == 0

    def test_leeftijd_is_integer(self, raw_students):
        clean = clean_students(raw_students)
        assert clean["leeftijd"].dtype == int

    def test_fills_studierichting(self, raw_students):
        clean = clean_students(raw_students)
        assert clean["studierichting"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# clean_courses
# ---------------------------------------------------------------------------

class TestCleanCourses:
    def test_fills_missing_credits(self, raw_courses):
        clean = clean_courses(raw_courses)
        assert clean["credits"].isnull().sum() == 0

    def test_credits_is_integer(self, raw_courses):
        clean = clean_courses(raw_courses)
        assert clean["credits"].dtype == int


# ---------------------------------------------------------------------------
# clean_results
# ---------------------------------------------------------------------------

class TestCleanResults:
    def test_drops_rows_with_null_student_id(self, raw_results):
        clean = clean_results(raw_results)
        assert clean["student_id"].isnull().sum() == 0

    def test_imputes_missing_score(self, raw_results):
        clean = clean_results(raw_results)
        assert clean["score"].isnull().sum() == 0

    def test_recalculates_slaagde(self, raw_results):
        clean = clean_results(raw_results)
        expected = (clean["score"] >= 10).astype(int)
        pd.testing.assert_series_equal(
            clean["slaagde"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_score_clipped_to_valid_range(self):
        df = pd.DataFrame(
            {
                "result_id": [1, 2],
                "student_id": [1, 2],
                "course_id": [1, 1],
                "score": [-5.0, 25.0],
                "slaagde": [0, 1],
            }
        )
        clean = clean_results(df)
        assert clean["score"].min() >= 0
        assert clean["score"].max() <= 20


# ---------------------------------------------------------------------------
# transform (integration)
# ---------------------------------------------------------------------------

class TestTransform:
    def test_returns_all_keys(self, raw_students, raw_courses, raw_results):
        raw = {
            "students": raw_students,
            "courses": raw_courses,
            "results": raw_results,
        }
        result = transform(raw)
        assert set(result.keys()) == {"students", "courses", "results"}

    def test_no_missing_values_after_transform(
        self, raw_students, raw_courses, raw_results
    ):
        raw = {
            "students": raw_students,
            "courses": raw_courses,
            "results": raw_results,
        }
        result = transform(raw)
        for table, df in result.items():
            nulls = df.isnull().sum().sum()
            assert nulls == 0, f"Table '{table}' still has {nulls} missing values"
