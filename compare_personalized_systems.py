"""
MindLink Personalized Systems Comparison

Run inside mindlink_ai_starter:
    python compare_personalized_systems.py

Compares:
A) SVM only
B) Personalized baseline deviation only
C) Combined SVM + personalized baseline

Outputs:
    results/personalized_systems_per_subject.csv
    results/personalized_systems_summary.csv
    results/personalized_systems_window_predictions.csv
    results/personalized_systems_report.md
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_mindlink_model import process_subject


SENSOR_PREFIXES = ["EDA", "BVP", "TEMP", "ACC"]


def load_all_data() -> pd.DataFrame:
    files = sorted(glob.glob("data/WESAD/S*/S*.pkl"))
    if not files:
        raise FileNotFoundError("No WESAD files found. Expected data/WESAD/S2/S2.pkl")

    rows = []
    for file in files:
        print("Loading", file)
        rows.extend(process_subject(file))

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        raise ValueError("No usable rows were created.")
    return df


def make_svm_model():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    )


def robust_subject_baseline(subject_df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.Series, pd.Series]:
    """
    Uses WESAD baseline windows as a calibration period.
    In a real product, this would be a user's initial baseline-learning period.
    """
    baseline_df = subject_df[subject_df["label"] == 0].copy()
    if baseline_df.empty:
        baseline_df = subject_df.copy()

    med = baseline_df[feature_columns].median()
    mad = (baseline_df[feature_columns] - med).abs().median()
    robust_scale = 1.4826 * mad
    std = baseline_df[feature_columns].std().fillna(0)

    scale = robust_scale.copy()
    scale[scale < 1e-8] = std[scale < 1e-8]
    scale[scale < 1e-8] = 1.0
    return med, scale


def baseline_deviation_scores(df: pd.DataFrame, feature_columns: List[str]) -> pd.Series:
    """
    Returns 0-100 personalized deviation score.
    Higher score = farther from that subject's own baseline.
    """
    scores = []

    for _, subject_df in df.groupby("subject", sort=False):
        med, scale = robust_subject_baseline(subject_df, feature_columns)
        z = ((subject_df[feature_columns] - med).abs() / scale).clip(upper=12)

        sensor_z = []
        for sensor in SENSOR_PREFIXES:
            cols = [c for c in feature_columns if c.startswith(sensor)]
            if cols:
                sensor_z.append(z[cols].median(axis=1))
            else:
                sensor_z.append(pd.Series(0.0, index=subject_df.index))

        max_z = pd.concat(sensor_z, axis=1).max(axis=1)
        score = max_z.apply(lambda v: min(100.0, 100.0 * (1.0 - math.exp(-float(v) / 3.0))))
        scores.append(score)

    return pd.concat(scores).sort_index()


def tune_threshold(scores: np.ndarray, y_true: np.ndarray, thresholds: Iterable[float]) -> Tuple[float, float]:
    """
    Pick threshold using training subjects only.
    Primary objective: stress F1.
    Tie-breaker: stress recall.
    """
    best_threshold = 50.0
    best_f1 = -1.0
    best_recall = -1.0

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        report = classification_report(
            y_true,
            pred,
            target_names=["baseline", "stress"],
            output_dict=True,
            zero_division=0,
        )
        f1 = report["stress"]["f1-score"]
        recall = report["stress"]["recall"]

        if (f1 > best_f1) or (np.isclose(f1, best_f1) and recall > best_recall):
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_recall = float(recall)

    return best_threshold, best_f1


def tune_combined(
    svm_scores: np.ndarray,
    dev_scores: np.ndarray,
    y_true: np.ndarray,
    weights: Iterable[float],
    thresholds: Iterable[float],
) -> Tuple[float, float, float]:
    """
    combined_score = weight * svm_score + (1 - weight) * baseline_deviation_score
    """
    best_weight = 0.5
    best_threshold = 50.0
    best_f1 = -1.0
    best_recall = -1.0

    for weight in weights:
        combined = weight * svm_scores + (1.0 - weight) * dev_scores

        for threshold in thresholds:
            pred = (combined >= threshold).astype(int)
            report = classification_report(
                y_true,
                pred,
                target_names=["baseline", "stress"],
                output_dict=True,
                zero_division=0,
            )
            f1 = report["stress"]["f1-score"]
            recall = report["stress"]["recall"]

            if (f1 > best_f1) or (np.isclose(f1, best_f1) and recall > best_recall):
                best_weight = float(weight)
                best_threshold = float(threshold)
                best_f1 = float(f1)
                best_recall = float(recall)

    return best_weight, best_threshold, best_f1


def metrics_row(system: str, subject: str, y_true: np.ndarray, pred: np.ndarray, threshold, weight):
    report = classification_report(
        y_true,
        pred,
        target_names=["baseline", "stress"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "system": system,
        "subject": subject,
        "test_windows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "baseline_precision": float(report["baseline"]["precision"]),
        "baseline_recall": float(report["baseline"]["recall"]),
        "baseline_f1": float(report["baseline"]["f1-score"]),
        "stress_precision": float(report["stress"]["precision"]),
        "stress_recall": float(report["stress"]["recall"]),
        "stress_f1": float(report["stress"]["f1-score"]),
        "false_stress_alerts": int(cm[0][1]),
        "missed_stress_windows": int(cm[1][0]),
        "tuned_threshold": None if threshold is None else float(threshold),
        "tuned_weight": None if weight is None else float(weight),
    }


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    df = load_all_data()
    feature_columns = [c for c in df.columns if c not in ["label", "subject", "start_sec"]]
    subjects = sorted(df["subject"].unique())

    thresholds = np.linspace(0, 100, 201)
    weights = np.linspace(0, 1, 11)

    all_results = []
    prediction_rows = []

    print("\nSubjects:", subjects)
    print("\nRunning personalized systems experiment...")

    for holdout_subject in subjects:
        print(f"\n===== HOLDOUT SUBJECT: {holdout_subject} =====")

        train_df = df[df["subject"] != holdout_subject].copy()
        test_df = df[df["subject"] == holdout_subject].copy()

        X_train = train_df[feature_columns]
        y_train = train_df["label"].to_numpy()
        train_groups = train_df["subject"]

        X_test = test_df[feature_columns]
        y_test = test_df["label"].to_numpy()

        # Baseline deviation scores.
        train_dev_scores = baseline_deviation_scores(train_df, feature_columns).loc[train_df.index].to_numpy()
        test_dev_scores = baseline_deviation_scores(test_df, feature_columns).loc[test_df.index].to_numpy()

        # Out-of-fold SVM train scores for threshold tuning.
        inner_cv = GroupKFold(n_splits=min(5, train_df["subject"].nunique()))
        train_svm_scores = cross_val_predict(
            make_svm_model(),
            X_train,
            y_train,
            groups=train_groups,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1] * 100

        # Final SVM trained on non-holdout subjects.
        svm_final = make_svm_model()
        svm_final.fit(X_train, y_train)
        test_svm_scores = svm_final.predict_proba(X_test)[:, 1] * 100

        svm_threshold, _ = tune_threshold(train_svm_scores, y_train, thresholds)
        dev_threshold, _ = tune_threshold(train_dev_scores, y_train, thresholds)
        combined_weight, combined_threshold, _ = tune_combined(
            train_svm_scores,
            train_dev_scores,
            y_train,
            weights,
            thresholds,
        )

        svm_pred = (test_svm_scores >= svm_threshold).astype(int)
        dev_pred = (test_dev_scores >= dev_threshold).astype(int)
        combined_test_scores = combined_weight * test_svm_scores + (1.0 - combined_weight) * test_dev_scores
        combined_pred = (combined_test_scores >= combined_threshold).astype(int)

        all_results.append(metrics_row("SVM only", holdout_subject, y_test, svm_pred, svm_threshold, None))
        all_results.append(metrics_row("Personalized baseline only", holdout_subject, y_test, dev_pred, dev_threshold, 0.0))
        all_results.append(metrics_row("Combined SVM + baseline", holdout_subject, y_test, combined_pred, combined_threshold, combined_weight))

        print("SVM threshold:", round(svm_threshold, 2))
        print("Baseline threshold:", round(dev_threshold, 2))
        print("Combined weight:", round(combined_weight, 2), "threshold:", round(combined_threshold, 2))

        for i, idx in enumerate(test_df.index):
            prediction_rows.append({
                "subject": holdout_subject,
                "start_sec": int(test_df.loc[idx, "start_sec"]),
                "true_label": int(y_test[i]),
                "svm_score": float(test_svm_scores[i]),
                "baseline_deviation_score": float(test_dev_scores[i]),
                "combined_score": float(combined_test_scores[i]),
                "svm_pred": int(svm_pred[i]),
                "baseline_pred": int(dev_pred[i]),
                "combined_pred": int(combined_pred[i]),
            })

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(prediction_rows)

    summary_df = (
        results_df
        .groupby("system")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_stress_precision=("stress_precision", "mean"),
            mean_stress_recall=("stress_recall", "mean"),
            mean_stress_f1=("stress_f1", "mean"),
            total_false_stress_alerts=("false_stress_alerts", "sum"),
            total_missed_stress_windows=("missed_stress_windows", "sum"),
            weak_subjects_under_0_70_f1=("stress_f1", lambda x: int((x < 0.70).sum())),
        )
        .reset_index()
        .sort_values("mean_stress_f1", ascending=False)
    )

    print("\n===== SYSTEM SUMMARY =====")
    print(summary_df)

    print("\n===== PER-SUBJECT RESULTS =====")
    print(results_df.sort_values(["subject", "system"]))

    results_df.to_csv("results/personalized_systems_per_subject.csv", index=False)
    summary_df.to_csv("results/personalized_systems_summary.csv", index=False)
    predictions_df.to_csv("results/personalized_systems_window_predictions.csv", index=False)

    report = "# MindLink Personalized Systems Comparison\n\n"
    report += "## Research question\n"
    report += "Can personalized baseline deviation improve wearable stress-pattern alerting compared to a general machine-learning stress classifier?\n\n"
    report += "## Systems compared\n"
    report += "1. SVM only\n"
    report += "2. Personalized baseline only\n"
    report += "3. Combined SVM + baseline\n\n"
    report += "## Validation method\n"
    report += "Outer subject-holdout testing. Each subject is tested after training/tuning on the other subjects.\n\n"
    report += "Thresholds and combined weights are tuned on training subjects only.\n\n"
    report += "## Summary results\n\n"
    report += summary_df.to_markdown(index=False)
    report += "\n\n## Honest interpretation\n"
    report += "The best system is the one with high stress F1 and stress recall, but false alerts and missed stress windows must also be considered.\n\n"
    report += "For MindLink, stress recall is important because missing stress-pattern changes is a major weakness. Too many false alerts can also make caregivers ignore the device.\n\n"
    report += "## Limitations\n"
    report += "- WESAD is lab data, not original MindLink wristband data.\n"
    report += "- Baseline calibration uses WESAD baseline labels.\n"
    report += "- This is not emotion detection.\n"
    report += "- This is not medical diagnosis.\n"
    report += "- Real-world testing would require ethical approval and consent.\n"

    Path("results/personalized_systems_report.md").write_text(report, encoding="utf-8")

    print("\nSaved:")
    print("results/personalized_systems_per_subject.csv")
    print("results/personalized_systems_summary.csv")
    print("results/personalized_systems_window_predictions.csv")
    print("results/personalized_systems_report.md")


if __name__ == "__main__":
    main()
