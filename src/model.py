"""
model.py
--------
Trains a Random Forest classifier to predict whether
a student will pass (G3 >= 10) based on study features.

Usage:
    python src/model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder


CLEAN_PATH  = "data/clean/students_clean.csv"
MODEL_PATH  = "database/model.joblib"
FEATURES    = ["age", "studytime", "failures", "absences", "G1", "G2",
               "Medu", "Fedu"]


def load_data(path: str = CLEAN_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Clean data not found at '{path}'.\n"
            "Run 'python src/pipeline.py' first."
        )
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    """Select and encode features, return X and y."""
    available = [f for f in FEATURES if f in df.columns]
    df_model = df[available].copy()

    # Encode any remaining categorical columns
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

    X = df_model
    y = df["passed"]
    return X, y, available


def train(X, y) -> RandomForestClassifier:
    """Train the Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test, feature_names) -> None:
    """Print evaluation metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy  : {acc:.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    print("  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Fail  Pass")
    print(f"  Actual Fail  {cm[0][0]:>4}  {cm[0][1]:>4}")
    print(f"  Actual Pass  {cm[1][0]:>4}  {cm[1][1]:>4}")

    print("\n  Feature Importances:")
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<12} {bar} {imp:.3f}")


def save_model(model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n[model] Saved to '{path}'.")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Run 'python src/model.py' first."
        )
    return joblib.load(path)


def run_training() -> None:
    print("=" * 50)
    print("  Student Analytics — Model Training")
    print("=" * 50)

    df = load_data()
    X, y, feature_names = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[model] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    model = train(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"[model] 5-Fold CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    evaluate(model, X_test, y_test, feature_names)
    save_model(model)

    print("=" * 50)
    print("  Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    run_training()
