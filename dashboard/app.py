"""
app.py
------
Streamlit dashboard for Student Performance Analytics.

Usage:
    streamlit run dashboard/app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #EBF3FB;
        border-left: 4px solid #1F4E79;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1F4E79; }
    .metric-label { font-size: 0.85rem; color: #555; margin-top: 0.2rem; }
    h1 { color: #1F4E79 !important; }
    h2 { color: #2E75B6 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/clean/students_clean.csv"
    if not os.path.exists(path):
        st.error("❌ Clean data not found. Run `python src/pipeline.py` first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    path = "database/model.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


df_full = load_data()
model   = load_model()


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
st.sidebar.title("Filters")

schools = ["All"] + sorted(df_full["school"].unique().tolist()) if "school" in df_full.columns else ["All"]
sex_opts = ["All"] + sorted(df_full["sex"].unique().tolist()) if "sex" in df_full.columns else ["All"]

sel_school = st.sidebar.selectbox("School", schools)
sel_sex    = st.sidebar.selectbox("Gender", sex_opts)

if "age" in df_full.columns:
    age_range = st.sidebar.slider(
        "Age range",
        int(df_full["age"].min()),
        int(df_full["age"].max()),
        (int(df_full["age"].min()), int(df_full["age"].max()))
    )
else:
    age_range = None

# Apply filters
df = df_full.copy()
if sel_school != "All":
    df = df[df["school"] == sel_school]
if sel_sex != "All":
    df = df[df["sex"] == sel_sex]
if age_range and "age" in df.columns:
    df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(df)} / {len(df_full)} students")


# ── Main content ───────────────────────────────────────────────────────────────
st.title("🎓 Student Performance Analytics")
st.markdown("Interactive dashboard for analysing and predicting student academic results.")
st.markdown("---")


# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(df)}</div>
        <div class="metric-label">Total Students</div>
    </div>""", unsafe_allow_html=True)

with k2:
    pass_rate = df["passed"].mean() if "passed" in df.columns else 0
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{pass_rate:.0%}</div>
        <div class="metric-label">Pass Rate</div>
    </div>""", unsafe_allow_html=True)

with k3:
    avg_g3 = df["G3"].mean() if "G3" in df.columns else 0
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_g3:.1f}</div>
        <div class="metric-label">Average Final Score (G3)</div>
    </div>""", unsafe_allow_html=True)

with k4:
    avg_abs = df["absences"].mean() if "absences" in df.columns else 0
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_abs:.1f}</div>
        <div class="metric-label">Average Absences</div>
    </div>""", unsafe_allow_html=True)


st.markdown("---")


# ── Charts Row 1 ───────────────────────────────────────────────────────────────
st.subheader("📊 Grade Distribution & Pass/Fail")

c1, c2 = st.columns(2)

with c1:
    if "G3" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df["G3"], bins=20, color="#2E75B6", edgecolor="white", ax=ax)
        ax.axvline(10, color="#C0392B", linestyle="--", linewidth=1.5, label="Pass line (10)")
        ax.set_title("Final Score Distribution (G3)", fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend()
        sns.despine()
        st.pyplot(fig)
        plt.close()

with c2:
    if "passed" in df.columns:
        counts = df["passed"].value_counts().rename({0: "Fail", 1: "Pass"})
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ["#C0392B", "#27AE60"]
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Pass / Fail Ratio", fontweight="bold")
        st.pyplot(fig)
        plt.close()


# ── Charts Row 2 ───────────────────────────────────────────────────────────────
st.subheader("📈 Study Behaviour & Performance")

c3, c4 = st.columns(2)

with c3:
    if "studytime" in df.columns and "G3" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        study_map = {1: "<2h", 2: "2–5h", 3: "5–10h", 4: ">10h"}
        df_plot = df.copy()
        df_plot["studytime_label"] = df_plot["studytime"].map(study_map)
        order = ["<2h", "2–5h", "5–10h", ">10h"]
        sns.boxplot(data=df_plot, x="studytime_label", y="G3",
                    order=order, palette="Blues", ax=ax)
        ax.set_title("Final Score by Study Time", fontweight="bold")
        ax.set_xlabel("Weekly Study Time")
        ax.set_ylabel("Final Score (G3)")
        sns.despine()
        st.pyplot(fig)
        plt.close()

with c4:
    if "failures" in df.columns and "passed" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        pass_rate_by_fail = df.groupby("failures")["passed"].mean().reset_index()
        sns.barplot(data=pass_rate_by_fail, x="failures", y="passed",
                    palette="Reds_r", ax=ax)
        ax.set_title("Pass Rate by Past Failures", fontweight="bold")
        ax.set_xlabel("Number of Past Failures")
        ax.set_ylabel("Pass Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        sns.despine()
        st.pyplot(fig)
        plt.close()


# ── Correlation heatmap ────────────────────────────────────────────────────────
st.subheader("🔥 Correlation Heatmap")

num_cols = [c for c in ["age", "studytime", "failures", "absences",
                         "G1", "G2", "G3", "Medu", "Fedu", "passed"]
            if c in df.columns]
if len(num_cols) >= 4:
    fig, ax = plt.subplots(figsize=(10, 4))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, square=False)
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    st.pyplot(fig)
    plt.close()


# ── ML Predictor ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Pass/Fail Predictor")

if model is None:
    st.warning("⚠️ Model not found. Run `python src/model.py` to train it first.")
else:
    st.markdown("Enter a student's details to predict whether they will pass.")
    p1, p2, p3, p4 = st.columns(4)

    age       = p1.number_input("Age",        min_value=15, max_value=22, value=17)
    studytime = p2.selectbox("Study time/week", [1, 2, 3, 4],
                              format_func=lambda x: {1:"<2h",2:"2–5h",3:"5–10h",4:">10h"}[x])
    failures  = p3.number_input("Past failures", min_value=0, max_value=4, value=0)
    absences  = p4.number_input("Absences",    min_value=0, max_value=93, value=4)

    p5, p6, p7, p8 = st.columns(4)
    g1   = p5.number_input("Grade 1 (G1)", min_value=0, max_value=20, value=12)
    g2   = p6.number_input("Grade 2 (G2)", min_value=0, max_value=20, value=12)
    medu = p7.number_input("Mother's education (0–4)", min_value=0, max_value=4, value=2)
    fedu = p8.number_input("Father's education (0–4)", min_value=0, max_value=4, value=2)

    if st.button("🔮 Predict", use_container_width=True):
        features   = ["age", "studytime", "failures", "absences", "G1", "G2", "Medu", "Fedu"]
        input_data = pd.DataFrame([[age, studytime, failures, absences, g1, g2, medu, fedu]],
                                   columns=features)
        pred  = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        if pred == 1:
            st.success(f"✅ Prediction: **PASS** — Confidence: {proba[1]:.0%}")
        else:
            st.error(f"❌ Prediction: **FAIL** — Confidence: {proba[0]:.0%}")

        st.progress(float(proba[1]), text=f"Pass probability: {proba[1]:.0%}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
    "Student Performance Analytics Dashboard · Toegepaste Informatica · 2025–2026"
    "</div>",
    unsafe_allow_html=True
)
