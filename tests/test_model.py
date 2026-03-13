"""Unit tests for the ML model."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.pipeline.load import create_schema, load_dataframes
from src.ml.model import build_pipeline, prepare_features, train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_large_dataset(n: int = 200) -> dict[str, pd.DataFrame]:
    """Create a synthetic dataset large enough for train/test split."""
    import numpy as np

    rng = np.random.default_rng(0)
    students = pd.DataFrame(
        {
            "student_id": range(1, n + 1),
            "naam": [f"S{i}" for i in range(1, n + 1)],
            "leeftijd": rng.integers(17, 26, size=n),
            "studierichting": rng.choice(
                ["Informatica", "Elektronica", "Mechanica", "Bouw", "Chemie"], size=n
            ),
        }
    )
    courses = pd.DataFrame(
        {
            "course_id": [1, 2, 3],
            "naam": ["Wiskunde", "Programmeren", "Netwerken"],
            "credits": [6, 6, 4],
            "semester": [1, 1, 2],
        }
    )
    # Ability strongly tied to studierichting so features are predictive
    program_ability = {
        "Informatica": 15.0,
        "Elektronica": 13.0,
        "Mechanica": 11.0,
        "Bouw": 8.0,
        "Chemie": 5.0,
    }
    records = []
    rid = 1
    for _, row in students.iterrows():
        base = program_ability[row["studierichting"]]
        for cid in courses["course_id"]:
            score = float(np.clip(rng.normal(base, 1.5), 0, 20))
            records.append(
                {
                    "result_id": rid,
                    "student_id": int(row["student_id"]),
                    "course_id": int(cid),
                    "score": round(score, 2),
                    "slaagde": int(score >= 10),
                }
            )
            rid += 1
    results = pd.DataFrame(records)
    return {"students": students, "courses": courses, "results": results}


@pytest.fixture
def db_conn():
    data = _make_large_dataset(200)
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    create_schema(conn)
    load_dataframes(data, conn)
    return conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLModel:
    def test_pipeline_builds(self):
        p = build_pipeline()
        assert p is not None

    def test_train_returns_pipeline_and_metrics(self, db_conn):
        pipeline, metrics = train(db_conn)
        assert pipeline is not None
        assert "accuracy" in metrics
        assert "classification_report" in metrics

    def test_accuracy_meets_requirement(self, db_conn):
        """Accuracy on test set must be >= 75%."""
        _, metrics = train(db_conn)
        assert metrics["accuracy"] >= 0.75, (
            f"Accuracy {metrics['accuracy']:.2%} < 75% requirement"
        )

    def test_predict_returns_binary_labels(self, db_conn):
        pipeline, _ = train(db_conn)
        X_new = pd.DataFrame(
            [{"leeftijd": 20, "credits": 5, "semester": 1, "studierichting": "Informatica"}]
        )
        preds = pipeline.predict(X_new)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, db_conn):
        import numpy as np

        pipeline, _ = train(db_conn)
        X_new = pd.DataFrame(
            [
                {"leeftijd": 20, "credits": 5, "semester": 1, "studierichting": "Informatica"},
                {"leeftijd": 18, "credits": 6, "semester": 2, "studierichting": "Bouw"},
            ]
        )
        proba = pipeline.predict_proba(X_new)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
