"""Student Performance Analytics Dashboard – Streamlit application.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

# Allow imports from the project root when running with `streamlit run`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.database.db import (
    get_connection,
    get_students,
    get_courses,
    get_results_enriched,
    validate_foreign_keys,
)
from src.analysis.eda import (
    score_summary,
    pass_rate_by_course,
    pass_rate_by_program,
    avg_score_per_student,
    fig_score_distribution,
    fig_pass_rate_by_course,
    fig_avg_score_by_program,
    fig_correlation_heatmap,
)
from src.ml.model import train, prepare_features

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Database path helper
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "analytics.db")


@st.cache_resource(show_spinner="Verbinding maken met database…")
def _get_conn():
    """Return a cached SQLite connection, running the pipeline first if needed."""
    if not os.path.isfile(DB_PATH):
        _run_pipeline()
    return get_connection(DB_PATH)


def _run_pipeline():
    """Generate data and run the ETL pipeline if the database does not exist."""
    import subprocess

    root = os.path.join(os.path.dirname(__file__), "..")
    # Generate raw CSVs
    subprocess.run(
        [sys.executable, "data/generate_data.py"],
        cwd=root,
        check=True,
    )
    # Run ETL
    from src.pipeline.load import run_pipeline

    run_pipeline(
        data_dir=os.path.join(root, "data", "raw"),
        db_path=DB_PATH,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🎓 Student Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigatie",
    [
        "📊 Overzicht",
        "📈 Analyse",
        "🤖 ML Model",
        "🗄️ Data",
    ],
)

conn = _get_conn()

# ---------------------------------------------------------------------------
# Page: Overzicht
# ---------------------------------------------------------------------------

if page == "📊 Overzicht":
    st.title("Student Performance Analytics Dashboard")
    st.markdown(
        "Welkom! Dit dashboard geeft inzicht in de academische prestaties van studenten."
    )

    df_results = get_results_enriched(conn)
    df_students = get_students(conn)
    df_courses = get_courses(conn)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Studenten", len(df_students))
    col2.metric("Vakken", len(df_courses))
    col3.metric("Resultaten", len(df_results))
    col4.metric(
        "Globaal slaagpercentage",
        f"{df_results['slaagde'].mean() * 100:.1f} %",
    )

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Verdeling van examencijfers")
        st.pyplot(fig_score_distribution(conn))
    with col_r:
        st.subheader("Slaagpercentage per vak")
        st.pyplot(fig_pass_rate_by_course(conn))

    st.subheader("Beschrijvende statistieken – scores")
    st.dataframe(score_summary(conn), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Analyse
# ---------------------------------------------------------------------------

elif page == "📈 Analyse":
    st.title("Verkennende Data Analyse (EDA)")

    tab1, tab2, tab3 = st.tabs(
        ["Per studierichting", "Per vak", "Correlaties"]
    )

    with tab1:
        st.subheader("Cijferverdeling per studierichting")
        st.pyplot(fig_avg_score_by_program(conn))

        st.subheader("Slaagpercentage per studierichting")
        df_prog = pass_rate_by_program(conn)
        st.dataframe(df_prog, use_container_width=True)

    with tab2:
        st.subheader("Slaagpercentage per vak")
        df_course = pass_rate_by_course(conn)
        st.dataframe(df_course, use_container_width=True)
        st.pyplot(fig_pass_rate_by_course(conn))

    with tab3:
        st.subheader("Correlatiematrix")
        st.pyplot(fig_correlation_heatmap(conn))

    st.markdown("---")
    st.subheader("Gemiddeld cijfer per student")
    df_avg = avg_score_per_student(conn)
    # Filter by programme
    programs = ["Alle"] + sorted(df_avg["studierichting"].unique().tolist())
    selected = st.selectbox("Filter op studierichting", programs)
    if selected != "Alle":
        df_avg = df_avg[df_avg["studierichting"] == selected]
    st.dataframe(df_avg.sort_values("avg_score", ascending=False), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: ML Model
# ---------------------------------------------------------------------------

elif page == "🤖 ML Model":
    st.title("Voorspellend Model – Slaagkans")
    st.markdown(
        "Een **Random Forest** classifier voorspelt of een student zal slagen, "
        "op basis van: leeftijd, credits, semester en studierichting."
    )

    if st.button("🚀 Train model"):
        with st.spinner("Model trainen…"):
            pipeline, metrics = train(conn)
        st.session_state["pipeline"] = pipeline
        st.session_state["metrics"] = metrics

    if "metrics" in st.session_state:
        m = st.session_state["metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{m['accuracy'] * 100:.2f} %")
        col2.metric("Trainingsset", m["n_train"])
        col3.metric("Testset", m["n_test"])

        passes = m["accuracy"] >= 0.75
        if passes:
            st.success(f"✅ Accuracy ({m['accuracy']*100:.2f}%) ≥ 75% – model voldoet aan de eis!")
        else:
            st.warning(f"⚠️ Accuracy ({m['accuracy']*100:.2f}%) < 75% – model voldoet niet aan de eis.")

        st.subheader("Classification Report")
        st.code(m["classification_report"])

    st.markdown("---")
    st.subheader("Voorspelling voor een nieuwe student")
    with st.form("predict_form"):
        col_a, col_b, col_c, col_d = st.columns(4)
        leeftijd = col_a.number_input("Leeftijd", min_value=16, max_value=35, value=20)
        credits = col_b.number_input("Credits (vak)", min_value=1, max_value=12, value=5)
        semester = col_c.number_input("Semester", min_value=1, max_value=2, value=1)
        studierichting = col_d.selectbox(
            "Studierichting",
            ["Informatica", "Elektronica", "Mechanica", "Bouw", "Chemie"],
        )
        submitted = st.form_submit_button("Voorspel")

    if submitted:
        if "pipeline" not in st.session_state:
            st.info("Train het model eerst via de knop hierboven.")
        else:
            import numpy as np

            X_new = pd.DataFrame(
                [
                    {
                        "leeftijd": leeftijd,
                        "credits": credits,
                        "semester": semester,
                        "studierichting": studierichting,
                    }
                ]
            )
            pipeline = st.session_state["pipeline"]
            proba = pipeline.predict_proba(X_new)[0][1]
            pred = pipeline.predict(X_new)[0]
            label = "✅ Slaagt" if pred == 1 else "❌ Faalt"
            st.metric("Voorspelling", label, f"P(slaagt) = {proba:.1%}")

# ---------------------------------------------------------------------------
# Page: Data
# ---------------------------------------------------------------------------

elif page == "🗄️ Data":
    st.title("Ruwe Datatabellen")

    tab_s, tab_c, tab_r = st.tabs(["Studenten", "Vakken", "Resultaten"])

    with tab_s:
        df = get_students(conn)
        st.write(f"**{len(df)} studenten**")
        st.dataframe(df, use_container_width=True)

    with tab_c:
        df = get_courses(conn)
        st.write(f"**{len(df)} vakken**")
        st.dataframe(df, use_container_width=True)

    with tab_r:
        df = get_results_enriched(conn)
        st.write(f"**{len(df)} resultaten**")
        # Allow filtering by pass/fail
        filter_col, _ = st.columns([1, 3])
        show = filter_col.selectbox("Filter resultaten", ["Alle", "Geslaagd", "Gezakt"])
        if show == "Geslaagd":
            df = df[df["slaagde"] == 1]
        elif show == "Gezakt":
            df = df[df["slaagde"] == 0]
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Database integriteitscontrole")
    ok = validate_foreign_keys(conn)
    if ok:
        st.success("✅ Alle foreign keys zijn geldig.")
    else:
        st.error("❌ Foreign key violations gedetecteerd!")
