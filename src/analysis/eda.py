"""Exploratory Data Analysis helpers.

All functions accept a *conn* (SQLite connection) and return either a
:class:`pandas.DataFrame` of statistics or a :class:`matplotlib.figure.Figure`.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.database.db import get_results_enriched, get_students, get_courses


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def score_summary(conn) -> pd.DataFrame:
    """Return descriptive statistics for the score column."""
    df = get_results_enriched(conn)
    return df["score"].describe().rename("score_stats").to_frame()


def pass_rate_by_course(conn) -> pd.DataFrame:
    """Return the pass rate (%) per course, sorted descending."""
    df = get_results_enriched(conn)
    grouped = (
        df.groupby("course_naam")["slaagde"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pass_rate", "count": "n_results"})
        .reset_index()
    )
    grouped["pass_rate"] = (grouped["pass_rate"] * 100).round(1)
    return grouped.sort_values("pass_rate", ascending=False)


def pass_rate_by_program(conn) -> pd.DataFrame:
    """Return the pass rate (%) per study programme."""
    df = get_results_enriched(conn)
    grouped = (
        df.groupby("studierichting")["slaagde"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pass_rate", "count": "n_results"})
        .reset_index()
    )
    grouped["pass_rate"] = (grouped["pass_rate"] * 100).round(1)
    return grouped.sort_values("pass_rate", ascending=False)


def avg_score_per_student(conn) -> pd.DataFrame:
    """Return each student's average score and pass rate."""
    df = get_results_enriched(conn)
    return (
        df.groupby(["student_id", "student_naam", "studierichting"])
        .agg(avg_score=("score", "mean"), pass_rate=("slaagde", "mean"))
        .reset_index()
        .assign(
            avg_score=lambda x: x["avg_score"].round(2),
            pass_rate=lambda x: (x["pass_rate"] * 100).round(1),
        )
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_score_distribution(conn) -> plt.Figure:
    """Histogram of all exam scores."""
    df = get_results_enriched(conn)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["score"], bins=20, kde=True, ax=ax, color="steelblue")
    ax.axvline(10, color="red", linestyle="--", label="Slaaggrens (10)")
    ax.set_title("Verdeling van examencijfers")
    ax.set_xlabel("Score")
    ax.set_ylabel("Aantal")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_pass_rate_by_course(conn) -> plt.Figure:
    """Horizontal bar chart: pass rate per course."""
    df = pass_rate_by_course(conn)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="pass_rate", y="course_naam", palette="Blues_d", ax=ax)
    ax.set_title("Slaagpercentage per vak")
    ax.set_xlabel("Slaagpercentage (%)")
    ax.set_ylabel("Vak")
    ax.set_xlim(0, 105)
    fig.tight_layout()
    return fig


def fig_avg_score_by_program(conn) -> plt.Figure:
    """Box plot of scores per study programme."""
    df = get_results_enriched(conn)
    fig, ax = plt.subplots(figsize=(9, 5))
    order = (
        df.groupby("studierichting")["score"].median().sort_values(ascending=False).index
    )
    sns.boxplot(
        data=df,
        x="studierichting",
        y="score",
        order=order,
        palette="Set2",
        ax=ax,
    )
    ax.axhline(10, color="red", linestyle="--", label="Slaaggrens (10)")
    ax.set_title("Cijferverdeling per studierichting")
    ax.set_xlabel("Studierichting")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_correlation_heatmap(conn) -> plt.Figure:
    """Correlation heatmap for numeric columns in the enriched results."""
    df = get_results_enriched(conn)
    numeric_cols = ["leeftijd", "credits", "semester", "score", "slaagde"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlatiematrix")
    fig.tight_layout()
    return fig
