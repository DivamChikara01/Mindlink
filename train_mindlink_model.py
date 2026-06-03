"""
MindLink AI Starter Model
Random Forest classifier for baseline vs stress-pattern change.

Dataset expected:
data/WESAD/S2/S2.pkl
data/WESAD/S3/S3.pkl
...

WESAD labels commonly used:
1 = baseline
2 = stress
3 = amusement
4 = meditation

For Version 1, this script only uses baseline and stress.
"""

from __future__ import annotations

import glob
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score


# WESAD timeline labels are aligned to the chest device sampling rate.
LABEL_FS = 700

# Wrist-device sampling rates in WESAD.
SENSOR_FS = {
    "ACC": 32,   # 3-axis acceleration
    "BVP": 64,   # blood volume pulse
    "EDA": 4,    # electrodermal activity
    "TEMP": 4,   # skin temperature
}

WINDOW_SEC = 60
STEP_SEC = 30


def extract_stat_features(signal_window: np.ndarray, sensor_name: str) -> Dict[str, float]:
    """
    Extract simple statistical features from one sensor window.
    Works for 1D sensors such as EDA/BVP/TEMP and 3D sensors such as ACC.
    """
    x = np.asarray(signal_window, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    features: Dict[str, float] = {}

    for col in range(x.shape[1]):
        values = x[:, col]

        if len(values) == 0:
            continue

        std = float(np.std(values))

        features[f"{sensor_name}_{col}_mean"] = float(np.mean(values))
        features[f"{sensor_name}_{col}_std"] = std
        features[f"{sensor_name}_{col}_min"] = float(np.min(values))
        features[f"{sensor_name}_{col}_max"] = float(np.max(values))
        features[f"{sensor_name}_{col}_range"] = float(np.max(values) - np.min(values))
        features[f"{sensor_name}_{col}_median"] = float(np.median(values))

        if std > 1e-8:
            features[f"{sensor_name}_{col}_skew"] = float(skew(values))
            features[f"{sensor_name}_{col}_kurtosis"] = float(kurtosis(values))
        else:
            features[f"{sensor_name}_{col}_skew"] = 0.0
            features[f"{sensor_name}_{col}_kurtosis"] = 0.0

    return features


def window_label(labels: np.ndarray) -> Optional[int]:
    """
    Convert WESAD labels to MindLink labels.

    Returns:
    0 = baseline
    1 = stress-pattern change
    None = skip mixed/other window
    """
    labels = np.asarray(labels)

    baseline_count = np.sum(labels == 1)
    stress_count = np.sum(labels == 2)
    useful_total = baseline_count + stress_count

    if useful_total == 0:
        return None

    # Only keep windows where the label is clearly mostly baseline or mostly stress.
    if baseline_count / useful_total >= 0.70:
        return 0

    if stress_count / useful_total >= 0.70:
        return 1

    return None


def process_subject(pkl_path: str) -> List[Dict[str, float]]:
    """
    Read one WESAD subject and convert it into training rows.
    """
    subject_id = Path(pkl_path).stem

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist_data = data["signal"]["wrist"]
    labels = data["label"]

    total_seconds = len(labels) // LABEL_FS
    rows: List[Dict[str, float]] = []

    for start_sec in range(0, total_seconds - WINDOW_SEC, STEP_SEC):
        end_sec = start_sec + WINDOW_SEC

        label_start = start_sec * LABEL_FS
        label_end = end_sec * LABEL_FS

        y = window_label(labels[label_start:label_end])

        if y is None:
            continue

        row: Dict[str, float] = {
            "subject": subject_id,
            "start_sec": start_sec,
            "label": y,
        }

        for sensor_name, fs in SENSOR_FS.items():
            sensor = wrist_data[sensor_name]
            sensor_start = start_sec * fs
            sensor_end = end_sec * fs
            sensor_window = sensor[sensor_start:sensor_end]

            row.update(extract_stat_features(sensor_window, sensor_name))

        rows.append(row)

    return rows


def main() -> None:
    data_files = sorted(glob.glob("data/WESAD/S*/S*.pkl"))

    if not data_files:
        raise FileNotFoundError(
            "No WESAD .pkl files found. Expected files like data/WESAD/S2/S2.pkl"
        )

    all_rows: List[Dict[str, float]] = []

    for file_path in data_files:
        print(f"Processing {file_path}")
        all_rows.extend(process_subject(file_path))

    df = pd.DataFrame(all_rows).dropna()

    if df.empty:
        raise ValueError("No usable baseline/stress windows were created.")

    print("\nDataset shape:", df.shape)
    print("\nClass counts:")
    print(df["label"].value_counts())

    X = df.drop(columns=["label", "subject", "start_sec"])
    y = df["label"]
    groups = df["subject"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # GroupKFold tests on subjects the model did not train on.
    n_splits = min(5, df["subject"].nunique())
    cv = GroupKFold(n_splits=n_splits)

    f1_scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )

    print("\nSubject-independent F1 scores:", f1_scores)
    print("Average F1:", round(float(f1_scores.mean()), 4))

    predicted = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        n_jobs=-1,
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y, predicted))

    print("\nClassification report:")
    print(classification_report(y, predicted, target_names=["baseline", "stress"]))

    # Train final model on all available data after validation.
    model.fit(X, y)

    Path("models").mkdir(exist_ok=True)

    dump(model, "models/mindlink_random_forest.joblib")
    dump(list(X.columns), "models/mindlink_feature_columns.joblib")

    print("\nSaved model files:")
    print("models/mindlink_random_forest.joblib")
    print("models/mindlink_feature_columns.joblib")


if __name__ == "__main__":
    main()
