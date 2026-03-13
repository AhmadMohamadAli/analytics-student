# Student Performance Analytics Dashboard

An end-to-end data analytics project for the course **Toegepaste Informatica** (2025–2026).  
It covers the full data lifecycle: from raw CSV ingestion through an ETL pipeline, SQLite storage, exploratory data analysis, a machine-learning classifier, and finally an interactive Streamlit dashboard.

---

## Project structure

```
analytics-student/
├── data/
│   ├── generate_data.py      # Simulated dataset generator
│   └── raw/                  # Generated CSV files (students, courses, results)
├── src/
│   ├── pipeline/
│   │   ├── ingest.py         # Load CSV files into DataFrames
│   │   ├── transform.py      # Clean & validate data
│   │   └── load.py           # Write DataFrames to SQLite + run full ETL
│   ├── database/
│   │   └── db.py             # Query helpers & FK validation
│   ├── analysis/
│   │   └── eda.py            # Summary statistics & Matplotlib/Seaborn figures
│   └── ml/
│       └── model.py          # Random Forest classifier (≥ 75 % accuracy)
├── dashboard/
│   └── app.py                # Streamlit web application
├── tests/
│   ├── test_pipeline.py      # Unit tests for ingest + transform
│   ├── test_database.py      # Unit tests for database layer
│   └── test_model.py         # Unit tests for ML model
└── requirements.txt
```

---

## Technology stack

| Layer            | Technology              | Purpose                    |
|------------------|-------------------------|----------------------------|
| Data Processing  | Python 3 / Pandas / NumPy | Ingest & transform         |
| Database         | SQLite                  | Persistent storage         |
| Visualisation    | Seaborn / Matplotlib    | Charts & figures           |
| Machine Learning | Scikit-learn            | Pass/fail classifier       |
| Dashboard        | Streamlit               | Interactive web UI         |
| Version control  | Git / GitHub            | Collaboration & code       |

---

## Database schema

```
students  (student_id PK, naam, leeftijd, studierichting)
courses   (course_id PK, naam, credits, semester)
results   (result_id PK, student_id FK, course_id FK, score, slaagde)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python data/generate_data.py
```

### 3. Run the ETL pipeline

```bash
python -m src.pipeline.load
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will automatically generate data and run the ETL pipeline if the database does not exist.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

All 25 tests should pass, including a test that verifies the ML model achieves ≥ 75 % accuracy on the hold-out test set.

---

## Dashboard pages

| Page             | Description                                           |
|------------------|-------------------------------------------------------|
| 📊 Overzicht     | KPI metrics, score distribution, pass rate per course |
| 📈 Analyse       | EDA: per programme, per course, correlation heatmap   |
| 🤖 ML Model      | Train classifier, view metrics, predict for new input |
| 🗄️ Data          | Browse raw tables, filter results, FK integrity check |

---

## Quality criteria (from project plan)

| Component      | Criterion                          | Status  |
|----------------|------------------------------------|---------|
| Data Pipeline  | No missing values after cleaning   | ✅      |
| Database       | All foreign keys valid             | ✅      |
| ML Model       | Accuracy ≥ 75 % on test set        | ✅ ~89 % |
| Dashboard      | All pages functional               | ✅      |
