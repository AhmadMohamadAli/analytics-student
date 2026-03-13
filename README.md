

# 🎓 Student Performance Analytics

A complete data analytics project built for the **Toegepaste Informatica** programme.  
It covers the full data lifecycle: ingestion → cleaning → database → analysis → ML → dashboard.

---

## 📸 Project Overview

| Layer | Technology |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Database | SQLite |
| Analysis & Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Random Forest) |
| Dashboard | Streamlit |
| Version Control | Git / GitHub |

---

## 🗂️ Project Structure

```
student-analytics/
├── data/
│   ├── raw/                  ← Place Kaggle CSV here
│   └── clean/                ← Auto-generated after pipeline
├── database/
│   └── students.db           ← Auto-generated SQLite database
├── src/
│   ├── pipeline.py           ← Step 1: Load & clean data
│   ├── database.py           ← Step 2: Save to SQLite
│   └── model.py              ← Step 3: Train ML model
├── dashboard/
│   └── app.py                ← Step 4: Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/student-analytics.git
cd student-analytics
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Go to [Kaggle — Student Alcohol Consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)  
Download and rename the file to `students.csv`, then place it in `data/raw/`.

---

## ▶️ Run the Project (in order)

```bash
# Step 1 — Clean the data
python src/pipeline.py

# Step 2 — Build the database
python src/database.py

# Step 3 — Train the ML model
python src/model.py

# Step 4 — Launch the dashboard
streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Features

- **Data Pipeline** — automated cleaning, imputation, feature engineering
- **SQLite Database** — normalised schema with students and performance tables
- **Exploratory Data Analysis** — grade distributions, pass rates, correlations
- **ML Predictor** — Random Forest classifier predicting pass/fail (target ≥ 75% accuracy)
- **Interactive Dashboard** — filters, KPI cards, charts, and live predictor

---

## 📋 Project Management Plan

The full PMP document is available in the repository root:  
`PMP_Student_Performance_Analytics.docx`

---

## 👤 Author

- **Programme:** Toegepaste Informatica  
- **Academic year:** 2025 – 2026  

---

## 📄 License

This project is created for educational purposes only.
