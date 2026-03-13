"""
database.py
-----------
Handles all SQLite database operations:
creating tables, inserting data, and querying.

Usage:
    python src/database.py
"""

import os
import sqlite3
import pandas as pd


DB_PATH = "database/students.db"
CLEAN_PATH = "data/clean/students_clean.csv"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def create_tables(conn: sqlite3.Connection) -> None:
    """Create database schema."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            school      TEXT,
            sex         TEXT,
            age         INTEGER,
            address     TEXT,
            famsize     TEXT,
            Pstatus     TEXT,
            Medu        INTEGER,
            Fedu        INTEGER,
            Mjob        TEXT,
            Fjob        TEXT,
            reason      TEXT,
            guardian    TEXT
        );

        CREATE TABLE IF NOT EXISTS performance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  INTEGER REFERENCES students(id),
            studytime   INTEGER,
            failures    INTEGER,
            absences    INTEGER,
            G1          REAL,
            G2          REAL,
            G3          REAL,
            avg_grade   REAL,
            passed      INTEGER
        );
    """)
    conn.commit()
    print("[db] Tables created.")


def load_from_csv(path: str = CLEAN_PATH) -> pd.DataFrame:
    """Load the cleaned CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Clean data not found at '{path}'.\n"
            "Run 'python src/pipeline.py' first."
        )
    return pd.read_csv(path)


def insert_data(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Insert DataFrame rows into the database tables."""

    student_cols = [c for c in ["school", "sex", "age", "address", "famsize",
                                 "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
                                 "reason", "guardian"] if c in df.columns]
    perf_cols = [c for c in ["studytime", "failures", "absences",
                              "G1", "G2", "G3", "avg_grade", "passed"] if c in df.columns]

    # Clear existing data
    conn.execute("DELETE FROM performance")
    conn.execute("DELETE FROM students")
    conn.commit()

    for _, row in df.iterrows():
        cur = conn.execute(
            f"INSERT INTO students ({', '.join(student_cols)}) VALUES ({', '.join(['?']*len(student_cols))})",
            [row.get(c) for c in student_cols]
        )
        student_id = cur.lastrowid
        conn.execute(
            f"INSERT INTO performance (student_id, {', '.join(perf_cols)}) VALUES (?, {', '.join(['?']*len(perf_cols))})",
            [student_id] + [row.get(c) for c in perf_cols]
        )

    conn.commit()
    print(f"[db] Inserted {len(df)} student records.")


def query(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Run any SQL query and return result as DataFrame."""
    conn = get_connection(db_path)
    result = pd.read_sql(sql, conn)
    conn.close()
    return result


def setup_database() -> None:
    """Full setup: create tables and load data from clean CSV."""
    print("=" * 50)
    print("  Student Analytics — Database Setup")
    print("=" * 50)
    df = load_from_csv()
    conn = get_connection()
    create_tables(conn)
    insert_data(df, conn)
    conn.close()

    # Quick verification
    result = query("SELECT COUNT(*) as total FROM students")
    print(f"[db] Verification — total students in DB: {result['total'][0]}")
    print("=" * 50)
    print("  Database setup complete!")
    print("=" * 50)


if __name__ == "__main__":
    setup_database()
