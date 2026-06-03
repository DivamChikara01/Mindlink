"""
MindLink model comparison.

Run:
    python compare_models.py

Compares simple models using subject-independent GroupKFold.
Do not choose a model because it sounds fancy. Choose it because it performs well and is explainable.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_mindlink_model import process_subject


def load_df() -> pd.DataFrame:
    files = sorted(glob.glob("data/WESAD/S*/S*.pkl"))
    rows = []
    for file in files:
        print("Loading", file)
        rows.extend(process_subject(file))
    return pd.DataFrame(rows).dropna()


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    df = load_df()
    X = df.drop(columns=["label", "subject", "start_sec"])
    y = df["label"]
    groups = df["subject"]

    cv = GroupKFold(n_splits=min(5, df["subject"].nunique()))

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "SVM RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    scoring = {
        "f1": make_scorer(f1_score, zero_division=0),
        "stress_recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "accuracy": "accuracy",
    }

    rows = []
    for name, model in models.items():
        print(f"\nTesting {name}")
        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "mean_f1": scores["test_f1"].mean(),
                "mean_stress_recall": scores["test_stress_recall"].mean(),
                "mean_accuracy": scores["test_accuracy"].mean(),
            }
        )

    results = pd.DataFrame(rows).sort_values("mean_f1", ascending=False)
    print("\n===== MODEL COMPARISON =====")
    print(results)

    results.to_csv("results/model_comparison.csv", index=False)
    print("\nSaved: results/model_comparison.csv")


if __name__ == "__main__":
    main()
