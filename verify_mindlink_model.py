"""
MindLink Model Verification Suite

Run:
    python verify_mindlink_model.py

This checks whether the model is probably learning real signal:
1. Subject-independent validation
2. Dumb baseline comparison
3. Shuffled-label test
4. Feature importance

Outputs saved to:
    results/
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

from train_mindlink_model import process_subject


def load_all_rows() -> pd.DataFrame:
    files = sorted(glob.glob("data/WESAD/S*/S*.pkl"))
    if not files:
        raise FileNotFoundError("No WESAD files found. Expected data/WESAD/S2/S2.pkl")

    rows = []
    for file in files:
        print("Loading", file)
        rows.extend(process_subject(file))

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        raise ValueError("No rows created.")
    return df


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    df = load_all_rows()
    X = df.drop(columns=["label", "subject", "start_sec"])
    y = df["label"]
    groups = df["subject"]

    cv = GroupKFold(n_splits=min(5, df["subject"].nunique()))

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\n===== REAL MODEL: SUBJECT-INDEPENDENT TEST =====")
    real_f1 = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1", n_jobs=-1)
    real_acc = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="accuracy", n_jobs=-1)
    pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)

    real_report = classification_report(
        y,
        pred,
        target_names=["baseline", "stress"],
        output_dict=True,
        zero_division=0,
    )
    real_cm = confusion_matrix(y, pred)

    print("F1 scores:", real_f1)
    print("Average F1:", real_f1.mean())
    print("Accuracy scores:", real_acc)
    print("Average accuracy:", real_acc.mean())
    print("Confusion matrix:")
    print(real_cm)
    print(classification_report(y, pred, target_names=["baseline", "stress"], zero_division=0))

    print("\n===== DUMB BASELINE MODEL =====")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_f1 = cross_val_score(dummy, X, y, groups=groups, cv=cv, scoring="f1", n_jobs=-1)
    dummy_acc = cross_val_score(dummy, X, y, groups=groups, cv=cv, scoring="accuracy", n_jobs=-1)
    print("Dumb F1 scores:", dummy_f1)
    print("Dumb average F1:", dummy_f1.mean())
    print("Dumb accuracy scores:", dummy_acc)
    print("Dumb average accuracy:", dummy_acc.mean())

    print("\n===== SHUFFLED LABEL TEST =====")
    rng = np.random.default_rng(42)
    y_shuffled = pd.Series(rng.permutation(y), index=y.index)
    shuffled_f1 = cross_val_score(model, X, y_shuffled, groups=groups, cv=cv, scoring="f1", n_jobs=-1)
    shuffled_acc = cross_val_score(model, X, y_shuffled, groups=groups, cv=cv, scoring="accuracy", n_jobs=-1)
    print("Shuffled F1 scores:", shuffled_f1)
    print("Shuffled average F1:", shuffled_f1.mean())
    print("Shuffled accuracy scores:", shuffled_acc)
    print("Shuffled average accuracy:", shuffled_acc.mean())

    print("\n===== FEATURE IMPORTANCE =====")
    model.fit(X, y)
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print(importance.head(25))

    pd.DataFrame(
        {
            "fold": list(range(1, len(real_f1) + 1)),
            "real_f1": real_f1,
            "real_accuracy": real_acc,
            "dummy_f1": dummy_f1,
            "dummy_accuracy": dummy_acc,
            "shuffled_f1": shuffled_f1,
            "shuffled_accuracy": shuffled_acc,
        }
    ).to_csv("results/validation_scores.csv", index=False)

    pd.DataFrame(real_cm, index=["actual_baseline", "actual_stress"], columns=["pred_baseline", "pred_stress"]).to_csv(
        "results/confusion_matrix.csv"
    )

    pd.DataFrame(real_report).transpose().to_csv("results/classification_report.csv")
    importance.to_csv("results/feature_importance.csv", index=False)

    summary = f"""# MindLink Validation Summary

## Task
Baseline vs stress-pattern change detection using WESAD wrist-sensor features.

## Validation method
Subject-independent GroupKFold cross-validation. This tests on people the model did not train on.

## Results
- Average F1: {real_f1.mean():.4f}
- Average accuracy: {real_acc.mean():.4f}
- Stress recall: {real_report['stress']['recall']:.4f}
- Baseline recall: {real_report['baseline']['recall']:.4f}

## Baseline comparisons
- Dumb model average F1: {dummy_f1.mean():.4f}
- Shuffled-label average F1: {shuffled_f1.mean():.4f}

## Interpretation
If the real model is much better than the dumb and shuffled-label models, it is probably learning real sensor patterns instead of random noise.

## Limitation
This is still not a medical device and not emotion detection. It needs original MindLink device data later.
"""
    Path("results/validation_summary.md").write_text(summary, encoding="utf-8")

    print("\nSaved validation files in results/")
    print("results/validation_scores.csv")
    print("results/confusion_matrix.csv")
    print("results/classification_report.csv")
    print("results/feature_importance.csv")
    print("results/validation_summary.md")


if __name__ == "__main__":
    main()
