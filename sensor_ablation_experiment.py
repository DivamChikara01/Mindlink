"""
MindLink Sensor Ablation Experiment

Research question:
Which wearable sensor groups contribute most to MindLink's stress-pattern detection?

Run inside your existing mindlink_ai_starter folder:

    python sensor_ablation_experiment.py

This tests sensor sets:
- ALL
- EDA only
- BVP only
- TEMP only
- ACC only
- EDA + BVP
- EDA + BVP + TEMP

For each sensor set, it compares:
1. SVM only
2. Personalized baseline only
3. Combined SVM + baseline

Validation:
- Outer subject-holdout testing
- Thresholds and combined weights are tuned on training subjects only

Outputs:
    results/sensor_ablation_per_subject.csv
    results/sensor_ablation_summary.csv
    results/sensor_ablation_best_by_system.csv
    results/sensor_ablation_report.txt

Important:
This is not emotion detection and not medical diagnosis.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_mindlink_model import process_subject


SENSOR_SETS: Dict[str, List[str]] = {
    "ALL": ["EDA", "BVP", "TEMP", "ACC"],
    "EDA_only": ["EDA"],
    "BVP_only": ["BVP"],
    "TEMP_only": ["TEMP"],
    "ACC_only": ["ACC"],
    "EDA_BVP": ["EDA", "BVP"],
    "EDA_BVP_TEMP": ["EDA", "BVP", "TEMP"],
}


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


def columns_for_sensors(all_feature_columns: List[str], sensors: List[str]) -> List[str]:
    cols = []
    for col in all_feature_columns:
        if any(col.startswith(sensor) for sensor in sensors):
            cols.append(col)

    if not cols:
        raise ValueError(f"No feature columns found for sensors: {sensors}")

    return cols


def make_svm_model():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    )


def robust_subject_baseline(subject_df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.Series, pd.Series]:
    """
    Uses WESAD baseline windows as a simulated calibration period.
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


def baseline_deviation_scores(df: pd.DataFrame, feature_columns: List[str], sensors: List[str]) -> pd.Series:
    """
    Returns 0-100 personalized deviation score.

    Higher score = farther from that subject's baseline.
    """
    all_scores = []

    for _, subject_df in df.groupby("subject", sort=False):
        med, scale = robust_subject_baseline(subject_df, feature_columns)
        z = ((subject_df[feature_columns] - med).abs() / scale).clip(upper=12)

        sensor_z_values = []

        for sensor in sensors:
            cols = [c for c in feature_columns if c.startswith(sensor)]
            if cols:
                sensor_z_values.append(z[cols].median(axis=1))

        if sensor_z_values:
            max_z = pd.concat(sensor_z_values, axis=1).max(axis=1)
        else:
            max_z = pd.Series(0.0, index=subject_df.index)

        score = max_z.apply(lambda v: min(100.0, 100.0 * (1.0 - math.exp(-float(v) / 3.0))))
        all_scores.append(score)

    return pd.concat(all_scores).sort_index()


def tune_threshold(scores: np.ndarray, y_true: np.ndarray, thresholds: Iterable[float]) -> Tuple[float, float]:
    """
    Select threshold on training subjects only.
    Objective: maximize stress F1.
    Tie-breaker: maximize stress recall.
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


def metrics_row(
    sensor_set: str,
    system: str,
    subject: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    tuned_threshold,
    tuned_weight,
):
    report = classification_report(
        y_true,
        pred,
        target_names=["baseline", "stress"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "sensor_set": sensor_set,
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
        "tuned_threshold": None if tuned_threshold is None else float(tuned_threshold),
        "tuned_weight": None if tuned_weight is None else float(tuned_weight),
    }


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    df = load_all_data()
    all_feature_columns = [c for c in df.columns if c not in ["label", "subject", "start_sec"]]
    subjects = sorted(df["subject"].unique())

    thresholds = np.linspace(0, 100, 201)
    weights = np.linspace(0, 1, 11)

    all_results = []

    print("\nSubjects:", subjects)
    print("\nRunning sensor ablation experiment...")

    for sensor_set_name, sensors in SENSOR_SETS.items():
        print(f"\n================ SENSOR SET: {sensor_set_name} ================")

        feature_columns = columns_for_sensors(all_feature_columns, sensors)
        print(f"Using {len(feature_columns)} features from {sensors}")

        for holdout_subject in subjects:
            print(f"Holdout: {holdout_subject}")

            train_df = df[df["subject"] != holdout_subject].copy()
            test_df = df[df["subject"] == holdout_subject].copy()

            X_train = train_df[feature_columns]
            y_train = train_df["label"].to_numpy()
            train_groups = train_df["subject"]

            X_test = test_df[feature_columns]
            y_test = test_df["label"].to_numpy()

            train_dev_scores = baseline_deviation_scores(train_df, feature_columns, sensors).loc[train_df.index].to_numpy()
            test_dev_scores = baseline_deviation_scores(test_df, feature_columns, sensors).loc[test_df.index].to_numpy()

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
            combined_scores = combined_weight * test_svm_scores + (1.0 - combined_weight) * test_dev_scores
            combined_pred = (combined_scores >= combined_threshold).astype(int)

            all_results.append(
                metrics_row(
                    sensor_set_name,
                    "SVM only",
                    holdout_subject,
                    y_test,
                    svm_pred,
                    svm_threshold,
                    None,
                )
            )

            all_results.append(
                metrics_row(
                    sensor_set_name,
                    "Personalized baseline only",
                    holdout_subject,
                    y_test,
                    dev_pred,
                    dev_threshold,
                    0.0,
                )
            )

            all_results.append(
                metrics_row(
                    sensor_set_name,
                    "Combined SVM + baseline",
                    holdout_subject,
                    y_test,
                    combined_pred,
                    combined_threshold,
                    combined_weight,
                )
            )

    results_df = pd.DataFrame(all_results)

    summary_df = (
        results_df
        .groupby(["sensor_set", "system"])
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

    best_by_system = (
        summary_df
        .sort_values("mean_stress_f1", ascending=False)
        .groupby("system")
        .head(3)
        .sort_values(["system", "mean_stress_f1"], ascending=[True, False])
    )

    print("\n===== SENSOR ABLATION SUMMARY =====")
    print(summary_df.to_string(index=False))

    print("\n===== BEST SENSOR SETS BY SYSTEM =====")
    print(best_by_system.to_string(index=False))

    results_df.to_csv("results/sensor_ablation_per_subject.csv", index=False)
    summary_df.to_csv("results/sensor_ablation_summary.csv", index=False)
    best_by_system.to_csv("results/sensor_ablation_best_by_system.csv", index=False)

    report_lines = []
    report_lines.append("MindLink Sensor Ablation Experiment")
    report_lines.append("=" * 40)
    report_lines.append("")
    report_lines.append("Research question:")
    report_lines.append("Which wearable sensor groups contribute most to MindLink stress-pattern detection?")
    report_lines.append("")
    report_lines.append("Systems tested:")
    report_lines.append("1. SVM only")
    report_lines.append("2. Personalized baseline only")
    report_lines.append("3. Combined SVM + baseline")
    report_lines.append("")
    report_lines.append("Validation:")
    report_lines.append("Outer subject-holdout testing. Thresholds and weights tuned on training subjects only.")
    report_lines.append("")
    report_lines.append("Summary:")
    report_lines.append(summary_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("Best sensor sets by system:")
    report_lines.append(best_by_system.to_string(index=False))
    report_lines.append("")
    report_lines.append("Interpretation guidance:")
    report_lines.append("- Highest stress F1 = best balanced stress detection.")
    report_lines.append("- Highest stress recall = catches more stress windows.")
    report_lines.append("- Fewer false stress alerts = less caregiver alert fatigue.")
    report_lines.append("- Fewer missed stress windows = safer caregiver-support behavior.")
    report_lines.append("")
    report_lines.append("Limitations:")
    report_lines.append("- WESAD is lab data, not original MindLink wristband data.")
    report_lines.append("- Personalized baseline uses WESAD baseline labels as simulated calibration.")
    report_lines.append("- This is not emotion detection and not medical diagnosis.")

    Path("results/sensor_ablation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nSaved:")
    print("results/sensor_ablation_per_subject.csv")
    print("results/sensor_ablation_summary.csv")
    print("results/sensor_ablation_best_by_system.csv")
    print("results/sensor_ablation_report.txt")


if __name__ == "__main__":
    main()
