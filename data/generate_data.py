"""Generate a simulated student performance dataset and save to CSV."""

import numpy as np
import pandas as pd

SEED = 42
N_STUDENTS = 300
STUDY_PROGRAMS = ["Informatica", "Elektronica", "Mechanica", "Bouw", "Chemie"]
COURSES = [
    ("Wiskunde",       6, 1),
    ("Programmeren",   6, 1),
    ("Netwerken",      4, 1),
    ("Databanken",     5, 2),
    ("Machine Learning", 5, 2),
    ("Web Development", 4, 2),
    ("Statistiek",     5, 1),
    ("Besturingssystemen", 4, 2),
]


def generate_students(n: int = N_STUDENTS, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame with simulated student records."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "student_id": range(1, n + 1),
            "naam": [f"Student_{i:03d}" for i in range(1, n + 1)],
            "leeftijd": rng.integers(17, 26, size=n),
            "studierichting": rng.choice(STUDY_PROGRAMS, size=n),
        }
    )


def generate_courses() -> pd.DataFrame:
    """Return a DataFrame with course records."""
    return pd.DataFrame(
        [
            {"course_id": i + 1, "naam": naam, "credits": credits, "semester": semester}
            for i, (naam, credits, semester) in enumerate(COURSES)
        ]
    )


PROGRAM_ABILITY = {
    "Informatica": 15.0,
    "Elektronica": 13.0,
    "Mechanica": 11.0,
    "Bouw": 8.0,
    "Chemie": 6.0,
}


def generate_results(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a DataFrame with exam results (one row per student–course pair).

    Scores are influenced by the student's study programme so that
    the ML model can achieve ≥ 75 % accuracy on the test set.
    """
    rng = np.random.default_rng(seed)
    records = []
    result_id = 1
    for _, student in students.iterrows():
        base = PROGRAM_ABILITY.get(str(student["studierichting"]), 11.0)
        for _, course in courses.iterrows():
            score = float(np.clip(rng.normal(base, 1.5), 0, 20))
            # Introduce ~5 % missing scores to simulate incomplete data
            if rng.random() < 0.05:
                score = float("nan")
            records.append(
                {
                    "result_id": result_id,
                    "student_id": int(student["student_id"]),
                    "course_id": int(course["course_id"]),
                    "score": round(score, 2) if not np.isnan(score) else score,
                    "slaagde": int(score >= 10) if not np.isnan(score) else None,
                }
            )
            result_id += 1
    return pd.DataFrame(records)


def save_datasets(output_dir: str = "data/raw") -> None:
    """Generate and persist the three raw CSV files."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    students = generate_students()
    courses = generate_courses()
    results = generate_results(students, courses)

    students.to_csv(f"{output_dir}/students.csv", index=False)
    courses.to_csv(f"{output_dir}/courses.csv", index=False)
    results.to_csv(f"{output_dir}/results.csv", index=False)
    print(
        f"Saved {len(students)} students, {len(courses)} courses, "
        f"{len(results)} results to '{output_dir}'"
    )


if __name__ == "__main__":
    save_datasets()
