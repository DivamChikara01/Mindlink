"""
Fair holdout test on one subject.

Example:
    python holdout_subject_test.py S2

This trains on every subject except S2, then tests on S2.
That is more honest than testing the final model on a subject it has already seen.
"""

from __future__ import annotations

import glob
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from train_mindlink_model import process_subject


def main() -> None:
    holdout_subject = sys.argv[1] if len(sys.argv) > 1 else "S2"

    files = sorted(glob.glob("data/WESAD/S*/S*.pkl"))
    rows = []

    for file in files:
        print("Loading", file)
        rows.extend(process_subject(file))

    df = pd.DataFrame(rows).dropna()

    train_df = df[df["subject"] != holdout_subject]
    test_df = df[df["subject"] == holdout_subject]

    if train_df.empty or test_df.empty:
        raise ValueError(f"Could not create train/test split for {holdout_subject}")

    X_train = train_df.drop(columns=["label", "subject", "start_sec"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label", "subject", "start_sec"])
    y_test = test_df["label"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"\n===== HOLDOUT SUBJECT TEST: {holdout_subject} =====")
    print("Train windows:", len(train_df))
    print("Test windows:", len(test_df))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["baseline", "stress"], zero_division=0))


if __name__ == "__main__":
    main()
